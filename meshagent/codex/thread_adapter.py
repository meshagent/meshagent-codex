import asyncio
import logging
import uuid
from datetime import datetime, timezone
from typing import Any

from meshagent.agents.thread_adapter import ThreadAdapter, tracer
from meshagent.api import Element, RoomException

logger = logging.getLogger("codex.thread")


class CodexThreadAdapter(ThreadAdapter):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._active_events_by_key: dict[str, Element] = {}
        self._active_reasoning_by_key: dict[str, Element] = {}
        self._persisted_kinds = {
            "exec",
            "tool",
            "collab",
            "web",
            "image",
            "diff",
            "approval",
        }

    async def stop(self) -> None:
        await super().stop()
        self._active_events_by_key.clear()
        self._active_reasoning_by_key.clear()

    def _append_assistant_message(self, *, messages: Element) -> Element:
        return messages.append_child(
            tag_name="message",
            attributes={
                "text": "",
                "created_at": datetime.now(timezone.utc)
                .isoformat()
                .replace("+00:00", "Z"),
                "author_name": self._room.local_participant.get_attribute("name"),
            },
        )

    def _extract_item(self, *, event: dict) -> dict:
        params = event.get("params")
        if not isinstance(params, dict):
            return {}
        item = params.get("item")
        if not isinstance(item, dict):
            return {}
        return item

    def _item_type(self, *, item: dict) -> str:
        item_type = item.get("type")
        if not isinstance(item_type, str):
            return ""
        return item_type.strip().lower()

    def _is_agent_message(self, *, item: dict) -> bool:
        item_type = self._item_type(item=item)
        return item_type in ("agentmessage", "agent_message")

    def _to_text(self, *, value: Any) -> str:
        if isinstance(value, str):
            return value

        if isinstance(value, list):
            parts: list[str] = []
            for part in value:
                if isinstance(part, str):
                    parts.append(part)
                    continue
                if isinstance(part, dict):
                    if isinstance(part.get("text"), str):
                        parts.append(part["text"])
                    elif isinstance(part.get("value"), str):
                        parts.append(part["value"])
            return "".join(parts)

        if isinstance(value, dict):
            if isinstance(value.get("text"), str):
                return value["text"]
            if isinstance(value.get("value"), str):
                return value["value"]
            nested = value.get("content")
            if nested is not None:
                return self._to_text(value=nested)

        return ""

    def _extract_item_text(self, *, item: dict) -> str:
        for key in ("text", "content", "message"):
            if key in item:
                text = self._to_text(value=item.get(key))
                if text != "":
                    return text
        return ""

    def _extract_delta(self, *, event: dict) -> str:
        params = event.get("params")
        if not isinstance(params, dict):
            return ""

        delta = params.get("delta")
        if isinstance(delta, str):
            return delta

        nested = params.get("item")
        if isinstance(nested, dict):
            nested_delta = nested.get("delta")
            if isinstance(nested_delta, str):
                return nested_delta
            return self._extract_item_text(item=nested)

        return ""

    def _extract_task_complete_text(self, *, event: dict) -> str:
        params = event.get("params")
        if not isinstance(params, dict):
            return ""

        msg = params.get("msg")
        if not isinstance(msg, dict):
            return ""

        last_agent_message = msg.get("last_agent_message")
        if not isinstance(last_agent_message, dict):
            return ""

        return self._extract_item_text(item=last_agent_message)

    async def _process_llm_events(self) -> None:
        if self._thread is None:
            raise RoomException("thread was not opened")

        messages = self._thread.root.get_children_by_tag_name("messages")
        if len(messages) == 0:
            raise RoomException("messages element is missing from thread document")
        doc_messages = messages[0]

        updates: asyncio.Queue = asyncio.Queue()
        content_element: Element | None = None
        partial = ""

        # Coalesce partial updates to reduce sync churn.
        async def update_thread() -> None:
            changes: dict[Element, str] = {}
            try:
                while True:
                    try:
                        element, partial_text = updates.get_nowait()
                        changes[element] = partial_text
                    except asyncio.QueueEmpty:
                        for element, partial_text in changes.items():
                            element["text"] = partial_text

                        changes.clear()

                        element, partial_text = await updates.get()
                        changes[element] = partial_text
            except asyncio.QueueShutDown:
                for element, partial_text in changes.items():
                    element["text"] = partial_text
                changes.clear()

        def finish_message(*, text: str) -> None:
            nonlocal content_element, partial
            if content_element is None:
                content_element = self._append_assistant_message(messages=doc_messages)
            updates.put_nowait((content_element, text))
            content_element = None
            partial = ""
            with tracer.start_as_current_span("chatbot.thread.message") as span:
                span.set_attribute(
                    "from_participant_name",
                    self._room.local_participant.get_attribute("name"),
                )
                span.set_attribute("role", "assistant")
                span.set_attribute("text", text)

        update_thread_task = asyncio.create_task(update_thread())
        try:
            while True:
                evt = await self._llm_messages.get()
                event_type = evt.get("type")
                if event_type in ("agent.event", "codex.event"):
                    await self.handle_custom_event(messages=doc_messages, event=evt)
                    continue

                method = evt.get("method")
                if isinstance(method, str):
                    method = method.strip().lower()

                if method in ("item/started", "codex/event/item_started"):
                    item = self._extract_item(event=evt)
                    if self._is_agent_message(item=item):
                        partial = ""
                        content_element = self._append_assistant_message(
                            messages=doc_messages
                        )

                elif method in (
                    "item/agentmessage/delta",
                    "item/agentmessage/content_delta",
                    "item/agent_message/delta",
                    "item/agent_message/content_delta",
                    "codex/event/agent_message_delta",
                    "codex/event/agent_message_content_delta",
                ):
                    delta = self._extract_delta(event=evt)
                    if delta == "":
                        continue
                    partial += delta
                    if content_element is None:
                        content_element = self._append_assistant_message(
                            messages=doc_messages
                        )
                    updates.put_nowait((content_element, partial))

                elif method in ("item/completed", "codex/event/item_completed"):
                    item = self._extract_item(event=evt)
                    if not self._is_agent_message(item=item):
                        continue
                    text = self._extract_item_text(item=item)
                    if text == "":
                        text = partial
                    finish_message(text=text)

                elif method == "codex/event/task_complete":
                    text = self._extract_task_complete_text(event=evt)
                    if text != "":
                        finish_message(text=text)

                else:
                    await self.handle_custom_event(messages=doc_messages, event=evt)
        except asyncio.QueueShutDown:
            pass
        finally:
            updates.shutdown()

        await update_thread_task

    def _is_active_state(self, *, state: str) -> bool:
        normalized = state.strip().lower()
        return normalized in ("in_progress", "queued", "running")

    def _normalized_token(self, *, value: str) -> str:
        return "".join(ch for ch in value.lower() if ch.isalnum())

    def _is_reasoning_delta_method(self, *, method: str) -> bool:
        normalized = self._normalized_token(value=method)
        return "summarytextdelta" in normalized

    def _is_reasoning_done_method(self, *, method: str) -> bool:
        normalized = self._normalized_token(value=method)
        return (
            "summarypartdone" in normalized
            or "summarytextdone" in normalized
            or normalized.endswith("done")
        )

    def _reasoning_text(
        self,
        *,
        summary: str,
        headline: str,
        details: str,
        method: str,
        is_delta: bool,
    ) -> str:
        candidates = (
            [details, summary, headline]
            if is_delta
            else [
                summary,
                details,
                headline,
            ]
        )
        method_normalized = method.strip().lower()

        for candidate in candidates:
            if not isinstance(candidate, str):
                continue
            text = candidate if is_delta else candidate.strip()
            if text == "":
                continue
            normalized = text.strip().lower()
            if normalized == method_normalized:
                continue
            if normalized in (
                "reasoning",
                "reasoned",
                "reasoning failed",
                "reasoning cancelled",
            ):
                continue
            return text

        return ""

    def _reasoning_key(
        self,
        *,
        correlation_key: str | None,
        item_id: str,
        method: str,
    ) -> str | None:
        if correlation_key is not None:
            return correlation_key
        if item_id.strip() != "":
            return f"item:{item_id.strip()}"
        method_key = method.strip().lower()
        if method_key != "":
            return f"method:{method_key}"
        return None

    def _upsert_reasoning(
        self,
        *,
        messages: Element,
        state: str,
        method: str,
        summary: str,
        headline: str,
        details: str,
        item_id: str,
        correlation_key: str | None,
        drop_correlation_keys: list[str],
    ) -> None:
        now = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
        reasoning_key = self._reasoning_key(
            correlation_key=correlation_key,
            item_id=item_id,
            method=method,
        )
        is_delta = self._is_reasoning_delta_method(method=method)
        text = self._reasoning_text(
            summary=summary,
            headline=headline,
            details=details,
            method=method,
            is_delta=is_delta,
        )
        in_progress = self._is_active_state(state=state)
        is_done = self._is_reasoning_done_method(method=method)

        reasoning_element: Element | None = None
        if reasoning_key is not None:
            reasoning_element = self._active_reasoning_by_key.get(reasoning_key)

        if reasoning_element is None and (text != "" or in_progress):
            reasoning_element = messages.append_child(
                tag_name="reasoning",
                attributes={
                    "summary": "" if is_delta else (text if text != "" else ""),
                    "created_at": now,
                },
            )

        if reasoning_element is not None and text != "":
            if is_delta:
                prior = reasoning_element.get_attribute("summary")
                if not isinstance(prior, str):
                    prior = ""
                reasoning_element.set_attribute("summary", f"{prior}{text}")
            else:
                reasoning_element.set_attribute("summary", text)

        if reasoning_key is not None:
            if in_progress and reasoning_element is not None:
                self._active_reasoning_by_key[reasoning_key] = reasoning_element
            elif state in ("completed", "failed", "cancelled") or is_done:
                self._active_reasoning_by_key.pop(reasoning_key, None)

        for key in drop_correlation_keys:
            self._active_reasoning_by_key.pop(key, None)

    async def handle_custom_event(
        self,
        *,
        messages: Element,
        event: dict,
    ) -> None:
        event_type = event.get("type")
        if event_type not in ("agent.event", "codex.event"):
            return

        method = event.get("method")
        if not isinstance(method, str) or method.strip() == "":
            method = "agent/event"

        method = method.strip()

        source = event.get("source")
        if not isinstance(source, str) or source.strip() == "":
            source = "codex" if event_type == "codex.event" else "agent"
        source = source.strip()

        raw_summary = event.get("summary")
        raw_headline = event.get("headline")

        raw_details = event.get("details")

        data = event.get("data")
        if not isinstance(data, str):
            data = ""

        event_name = event.get("name")
        if not isinstance(event_name, str) or event_name.strip() == "":
            event_name = event.get("event_type")
        if not isinstance(event_name, str) or event_name.strip() == "":
            event_name = method.replace("/", ".")
        event_name = event_name.strip()

        kind = event.get("kind")
        if not isinstance(kind, str) or kind.strip() == "":
            return
        kind = kind.strip().lower()

        details: str
        if isinstance(raw_details, list):
            if kind == "reasoning":
                detail_lines = [line for line in raw_details if isinstance(line, str)]
                details = "\n".join(detail_lines)
            else:
                detail_lines = [
                    line.strip() for line in raw_details if isinstance(line, str)
                ]
                details = "\n".join(line for line in detail_lines if line != "")
        elif isinstance(raw_details, str):
            details = raw_details if kind == "reasoning" else raw_details.strip()
        else:
            details = ""

        if isinstance(raw_summary, str):
            summary = raw_summary if kind == "reasoning" else raw_summary.strip()
        else:
            summary = ""
        if summary == "":
            summary = method

        if isinstance(raw_headline, str):
            headline = raw_headline if kind == "reasoning" else raw_headline.strip()
        else:
            headline = ""

        state = event.get("state")
        if not isinstance(state, str) or state.strip() == "":
            state = "info"
        state = state.strip().lower()

        in_progress = self._is_active_state(state=state)
        retain_correlation = event.get("retain_correlation") is True
        drop_correlation_keys = event.get("drop_correlation_keys")
        if isinstance(drop_correlation_keys, list):
            drop_correlation_keys = [
                key.strip()
                for key in drop_correlation_keys
                if isinstance(key, str) and key.strip() != ""
            ]
        else:
            drop_correlation_keys = []

        correlation_key = event.get("correlation_key")
        if not isinstance(correlation_key, str) or correlation_key.strip() == "":
            correlation_key = event.get("event_key")
        if not isinstance(correlation_key, str) or correlation_key.strip() == "":
            correlation_key = None
        else:
            correlation_key = correlation_key.strip()

        item_id = event.get("item_id")
        if not isinstance(item_id, str):
            item_id = ""
        item_type = event.get("item_type")
        if not isinstance(item_type, str):
            item_type = ""

        if kind == "reasoning":
            self._upsert_reasoning(
                messages=messages,
                state=state,
                method=method,
                summary=summary,
                headline=headline,
                details=details,
                item_id=item_id,
                correlation_key=correlation_key,
                drop_correlation_keys=drop_correlation_keys,
            )
            for key in drop_correlation_keys:
                self._active_events_by_key.pop(key, None)
            return

        if kind not in self._persisted_kinds:
            for key in drop_correlation_keys:
                self._active_events_by_key.pop(key, None)
            return

        now = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")

        event_element: Element | None = None
        if correlation_key is not None:
            event_element = self._active_events_by_key.get(correlation_key)

        try:
            if event_element is None:
                event_element = messages.append_child(
                    tag_name="event",
                    attributes={
                        "id": str(uuid.uuid4()),
                        "source": source,
                        "name": event_name,
                        "kind": kind,
                        "state": state,
                        "method": method,
                        "item_id": item_id,
                        "item_type": item_type,
                        "summary": summary,
                        "headline": headline,
                        "details": details,
                        "data": data,
                        "created_at": now,
                        "updated_at": now,
                    },
                )
            else:
                event_element.set_attribute("source", source)
                event_element.set_attribute("name", event_name)
                event_element.set_attribute("kind", kind)
                event_element.set_attribute("state", state)
                event_element.set_attribute("method", method)
                event_element.set_attribute("item_id", item_id)
                event_element.set_attribute("item_type", item_type)
                event_element.set_attribute("summary", summary)
                event_element.set_attribute("headline", headline)
                if details != "" or event_element.get_attribute("details") in (
                    None,
                    "",
                ):
                    event_element.set_attribute("details", details)
                event_element.set_attribute("data", data)
                event_element.set_attribute("updated_at", now)

            if correlation_key is not None:
                if in_progress or retain_correlation:
                    self._active_events_by_key[correlation_key] = event_element
                else:
                    self._active_events_by_key.pop(correlation_key, None)

            for key in drop_correlation_keys:
                self._active_events_by_key.pop(key, None)

        except Exception as ex:
            logger.error(f"unable to add message to thread {ex}", exc_info=ex)
