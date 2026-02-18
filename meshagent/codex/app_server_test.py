import asyncio
import os
import shutil
import uuid

import pytest
from meshagent.agents.context import AgentChatContext

from meshagent.codex.app_server import (
    CodexAppServerError,
    _CodexAppServerBackend,
    _CodexJsonRpcSession,
)
from meshagent.codex.thread_adapter import CodexThreadAdapter


class _FakeSession:
    def __init__(self, *, notifications: list[dict]):
        self._notifications: asyncio.Queue[dict] = asyncio.Queue()
        for notification in notifications:
            self._notifications.put_nowait(notification)
        self._requests: list[tuple[str, dict]] = []
        self._room = None

    def set_room(self, *, room) -> None:
        self._room = room

    async def start(self, *, room) -> None:
        self._room = room

    async def request(self, *, method: str, params: dict) -> dict:
        self._requests.append((method, params))
        if method == "turn/start":
            return {"turn": {"id": "turn-1"}}
        raise AssertionError(f"unexpected request: {method}")

    async def next_notification(self) -> dict:
        return await self._notifications.get()

    async def close(self) -> None:
        return


class _FakeElement:
    def __init__(self, *, tag_name: str, attributes: dict | None = None):
        self.tag_name = tag_name
        self._attributes = dict(attributes or {})
        self.children: list[_FakeElement] = []

    def append_child(self, *, tag_name: str, attributes: dict | None = None):
        element = _FakeElement(tag_name=tag_name, attributes=attributes)
        self.children.append(element)
        return element

    def get_attribute(self, name: str):
        return self._attributes.get(name)

    def set_attribute(self, name: str, value) -> None:
        self._attributes[name] = value


def _notification(
    *,
    method: str,
    thread_id: str,
    turn_id: str,
    item: dict | None = None,
    delta: str | None = None,
    turn_status: str | None = None,
) -> dict:
    params: dict = {
        "threadId": thread_id,
        "turnId": turn_id,
    }
    if item is not None:
        params["item"] = item
    if delta is not None:
        params["delta"] = delta
    if turn_status is not None:
        params["turn"] = {"id": turn_id, "status": turn_status}
    return {"method": method, "params": params}


@pytest.mark.asyncio
async def test_codex_next_streamed_delta_text_matches_final_message() -> None:
    thread_id = "thread-1"
    turn_id = "turn-1"
    final_text = "hello world"
    notifications = [
        _notification(
            method="item/started",
            thread_id=thread_id,
            turn_id=turn_id,
            item={"id": "item-1", "type": "agent_message"},
        ),
        _notification(
            method="item/agentmessage/delta",
            thread_id=thread_id,
            turn_id=turn_id,
            delta="hello ",
        ),
        # Some app-server versions emit both variants for the same chunk.
        _notification(
            method="item/agentmessage/content_delta",
            thread_id=thread_id,
            turn_id=turn_id,
            delta="hello ",
        ),
        _notification(
            method="item/agentmessage/delta",
            thread_id=thread_id,
            turn_id=turn_id,
            delta="world",
        ),
        _notification(
            method="item/completed",
            thread_id=thread_id,
            turn_id=turn_id,
            item={"id": "item-1", "type": "agent_message", "text": final_text},
        ),
        _notification(
            method="turn/completed",
            thread_id=thread_id,
            turn_id=turn_id,
            turn_status="completed",
        ),
    ]

    backend = _CodexAppServerBackend()
    backend._session = _FakeSession(notifications=notifications)

    context = AgentChatContext()
    await backend._set_thread_state(
        thread_key="thread:test",
        thread_id=thread_id,
        context=context,
    )

    emitted_events: list[dict] = []

    def _event_handler(event: dict) -> None:
        emitted_events.append(event)

    try:
        result = await backend.next(
            thread_key="thread:test",
            message="hi",
            room=object(),
            toolkits=[],
            event_handler=_event_handler,
        )
    finally:
        await backend.close()

    streamed_text = "".join(
        event.get("delta", "")
        for event in emitted_events
        if event.get("type") == "response.output_text.delta"
    )
    done_events = [
        event.get("text", "")
        for event in emitted_events
        if event.get("type") == "response.output_text.done"
    ]

    assert result == final_text
    assert streamed_text == final_text
    assert done_events == [final_text]
    assert context.messages[-1]["content"] == final_text


@pytest.mark.asyncio
async def test_reasoning_delta_keeps_whitespace_when_streaming() -> None:
    adapter = object.__new__(CodexThreadAdapter)
    adapter._active_events_by_key = {}
    adapter._active_reasoning_by_key = {}
    adapter._persisted_kinds = {
        "exec",
        "tool",
        "collab",
        "web",
        "image",
        "diff",
        "approval",
    }
    messages = _FakeElement(tag_name="messages")

    await adapter.handle_custom_event(
        messages=messages,
        event={
            "type": "codex.event",
            "kind": "reasoning",
            "state": "in_progress",
            "method": "response/reasoning_summary_text_delta",
            "item_id": "reason-1",
            "summary": "",
            "details": ["hello "],
        },
    )
    await adapter.handle_custom_event(
        messages=messages,
        event={
            "type": "codex.event",
            "kind": "reasoning",
            "state": "in_progress",
            "method": "response/reasoning_summary_text_delta",
            "item_id": "reason-1",
            "summary": "",
            "details": ["world"],
        },
    )

    assert len(messages.children) == 1
    reasoning = messages.children[0]
    assert reasoning.tag_name == "reasoning"
    assert reasoning.get_attribute("summary") == "hello world"


@pytest.mark.asyncio
async def test_codex_next_live_delta_build_matches_done_output() -> None:
    if os.getenv("MESHAGENT_CODEX_LIVE_TEST") != "1":
        pytest.skip("set MESHAGENT_CODEX_LIVE_TEST=1 to run live codex integration")
    if shutil.which("codex") is None:
        pytest.skip("codex executable not found on PATH")

    backend = _CodexAppServerBackend(forward_stdout=False, forward_stderr=False)
    thread_key = f"thread:live:{uuid.uuid4()}"
    context = AgentChatContext()
    emitted_events: list[dict] = []

    try:
        await backend.on_thread_open(
            thread_key=thread_key,
            room=None,  # type: ignore[arg-type]
            context=context,
        )
        result = await backend.next(
            thread_key=thread_key,
            message=(
                "Reply with one short sentence about streams and include at least "
                "five words."
            ),
            room=None,  # type: ignore[arg-type]
            toolkits=[],
            event_handler=emitted_events.append,
        )
    finally:
        try:
            await backend.on_thread_close(thread_key=thread_key)
        finally:
            await backend.close()

    streamed_text = "".join(
        event.get("delta", "")
        for event in emitted_events
        if event.get("type") == "response.output_text.delta"
    )
    done_events = [
        event.get("text", "")
        for event in emitted_events
        if event.get("type") == "response.output_text.done"
    ]

    assert streamed_text != ""
    assert done_events
    assert streamed_text == done_events[-1]
    assert result == done_events[-1]


@pytest.mark.asyncio
async def test_session_start_reports_missing_executable_details(monkeypatch) -> None:
    async def _raise_file_not_found(*args, **kwargs):
        del args, kwargs
        raise FileNotFoundError(2, "No such file or directory", "codex")

    monkeypatch.setattr(asyncio, "create_subprocess_exec", _raise_file_not_found)

    session = _CodexJsonRpcSession(command="codex app-server", env={"PATH": ""})
    with pytest.raises(CodexAppServerError) as exc_info:
        await session.start()

    message = str(exc_info.value)
    assert "unable to launch codex app-server with command: codex app-server" in message
    assert "missing_path=codex" in message
    assert "executable 'codex' not found on PATH" in message
