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

        summary = event.get("summary")
        if not isinstance(summary, str) or summary.strip() == "":
            summary = method
        summary = summary.strip()

        headline = event.get("headline")
        if not isinstance(headline, str):
            headline = ""
        headline = headline.strip()

        details = event.get("details")
        if isinstance(details, list):
            detail_lines = [line.strip() for line in details if isinstance(line, str)]
            details = "\n".join(line for line in detail_lines if line != "")
        elif isinstance(details, str):
            details = details.strip()
        else:
            details = ""

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
