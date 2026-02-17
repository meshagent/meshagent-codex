import uuid
from datetime import datetime, timezone

from meshagent.agents.thread_adapter import ThreadAdapter
from meshagent.api import Element
import logging

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
