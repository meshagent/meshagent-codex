import asyncio
import json
import os
import shutil
import uuid

import pytest
from meshagent.agents.context import AgentSessionContext

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
        if method == "turn/steer":
            return {"turnId": params.get("expectedTurnId")}
        raise AssertionError(f"unexpected request: {method}")

    async def next_notification(self) -> dict:
        return await self._notifications.get()

    async def close(self) -> None:
        return


class _FakeThreadOpenSession(_FakeSession):
    def __init__(
        self,
        *,
        resume_thread_ids: set[str] | None = None,
        start_thread_id: str = "thread-started",
    ):
        super().__init__(notifications=[])
        self._resume_thread_ids = resume_thread_ids or set()
        self._start_thread_id = start_thread_id

    async def request(self, *, method: str, params: dict) -> dict:
        self._requests.append((method, params))
        if method == "thread/resume":
            thread_id = params.get("threadId")
            if isinstance(thread_id, str) and thread_id in self._resume_thread_ids:
                return {"thread": {"id": thread_id}}
            raise RuntimeError("thread not found")
        if method == "thread/start":
            return {"thread": {"id": self._start_thread_id}}
        raise AssertionError(f"unexpected request: {method}")


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


def _streamed_text_from_events(events: list[dict]) -> str:
    parts: list[str] = []
    for event in events:
        method = event.get("method")
        if not isinstance(method, str):
            continue

        method = method.lower()
        if method not in (
            "item/agentmessage/delta",
            "item/agentmessage/content_delta",
            "item/agent_message/delta",
            "item/agent_message/content_delta",
            "codex/event/agent_message_delta",
            "codex/event/agent_message_content_delta",
        ):
            continue

        params = event.get("params")
        if not isinstance(params, dict):
            continue

        delta = params.get("delta")
        if isinstance(delta, str):
            parts.append(delta)

    return "".join(parts)


def _completed_texts_from_events(events: list[dict]) -> list[str]:
    texts: list[str] = []

    for event in events:
        method = event.get("method")
        if not isinstance(method, str):
            continue

        method = method.lower()
        params = event.get("params")
        if not isinstance(params, dict):
            continue

        if method in ("item/completed", "codex/event/item_completed"):
            item = params.get("item")
            if isinstance(item, dict):
                text = item.get("text")
                if isinstance(text, str):
                    texts.append(text)
            continue

        if method == "codex/event/task_complete":
            msg = params.get("msg")
            if not isinstance(msg, dict):
                continue
            last_message = msg.get("last_agent_message")
            if not isinstance(last_message, dict):
                continue
            text = last_message.get("text")
            if isinstance(text, str):
                texts.append(text)

    return texts


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

    context = AgentSessionContext()
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

    streamed_text = _streamed_text_from_events(emitted_events)
    done_events = _completed_texts_from_events(emitted_events)

    assert result == final_text
    assert streamed_text == final_text
    assert done_events == [final_text]
    assert context.messages[-1]["content"] == final_text


@pytest.mark.asyncio
async def test_codex_next_returns_empty_string_for_cancelled_turn_without_output() -> (
    None
):
    thread_id = "thread-1"
    turn_id = "turn-1"
    notifications = [
        _notification(
            method="turn/completed",
            thread_id=thread_id,
            turn_id=turn_id,
            turn_status="cancelled",
        ),
    ]

    backend = _CodexAppServerBackend()
    backend._session = _FakeSession(notifications=notifications)

    context = AgentSessionContext()
    await backend._set_thread_state(
        thread_key="thread:test",
        thread_id=thread_id,
        context=context,
    )

    try:
        result = await backend.next(
            thread_key="thread:test",
            message="hi",
            room=object(),
            toolkits=[],
            event_handler=None,
        )
    finally:
        await backend.close()

    assert result == ""
    assert context.messages == []


@pytest.mark.asyncio
async def test_codex_next_task_complete_finishes_before_late_turn_completed() -> None:
    thread_id = "thread-1"
    turn_id = "turn-1"
    final_text = "done"
    notifications = [
        _notification(
            method="turn/diff/updated",
            thread_id=thread_id,
            turn_id=turn_id,
        ),
        {
            "method": "codex/event/task_complete",
            "params": {
                "threadId": thread_id,
                "turnId": turn_id,
                "msg": {"last_agent_message": {"text": final_text}},
            },
        },
        _notification(
            method="turn/completed",
            thread_id=thread_id,
            turn_id=turn_id,
            turn_status="completed",
        ),
    ]

    backend = _CodexAppServerBackend()
    backend._session = _FakeSession(notifications=notifications)

    context = AgentSessionContext()
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
        active_turn = await backend._active_turn_id_for_thread(
            thread_key="thread:test",
            thread_id=thread_id,
        )
    finally:
        await backend.close()

    status_events = [
        event for event in emitted_events if event.get("type") == "agent.event"
    ]
    raw_methods = [
        method
        for method in (event.get("method") for event in emitted_events)
        if isinstance(method, str)
    ]

    assert result == final_text
    assert context.messages[-1]["content"] == final_text
    assert active_turn is None
    assert any(
        event.get("kind") == "diff" and event.get("state") == "in_progress"
        for event in status_events
    )
    assert "codex/event/task_complete" in raw_methods
    assert "turn/completed" not in raw_methods


@pytest.mark.asyncio
async def test_next_turn_notification_has_no_inactivity_timeout() -> None:
    backend = _CodexAppServerBackend(request_timeout_s=0.01)
    turn_queue: asyncio.Queue[dict] = asyncio.Queue()

    async def enqueue_later() -> None:
        await asyncio.sleep(0.05)
        turn_queue.put_nowait({"method": "turn/completed", "params": {}})

    notifier = asyncio.create_task(enqueue_later())
    try:
        notification = await backend._next_turn_notification(turn_queue=turn_queue)
    finally:
        await notifier
        await backend.close()

    assert notification["method"] == "turn/completed"


@pytest.mark.asyncio
async def test_on_thread_open_resumes_persisted_external_thread_id() -> None:
    backend = _CodexAppServerBackend()
    fake_session = _FakeThreadOpenSession(resume_thread_ids={"thread-persisted"})
    backend._session = fake_session
    context = AgentSessionContext()

    try:
        thread_id = await backend.on_thread_open(
            thread_key="thread:test",
            room=object(),  # type: ignore[arg-type]
            context=context,
            external_thread_id="thread-persisted",
        )
        state = await backend._get_thread_state(thread_key="thread:test")
    finally:
        await backend.close()

    assert thread_id == "thread-persisted"
    assert fake_session._requests[0] == (
        "thread/resume",
        {"threadId": "thread-persisted"},
    )
    assert all(method != "thread/start" for method, _params in fake_session._requests)
    assert state is not None
    assert state.thread_id == "thread-persisted"


@pytest.mark.asyncio
async def test_on_thread_open_starts_new_thread_when_persisted_resume_fails() -> None:
    backend = _CodexAppServerBackend()
    fake_session = _FakeThreadOpenSession(start_thread_id="thread-new")
    backend._session = fake_session
    context = AgentSessionContext()

    try:
        thread_id = await backend.on_thread_open(
            thread_key="thread:test",
            room=object(),  # type: ignore[arg-type]
            context=context,
            external_thread_id="thread-missing",
        )
        state = await backend._get_thread_state(thread_key="thread:test")
    finally:
        await backend.close()

    methods = [method for method, _params in fake_session._requests]
    assert methods[:2] == ["thread/resume", "thread/start"]
    assert thread_id == "thread-new"
    assert state is not None
    assert state.thread_id == "thread-new"


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


def test_build_status_event_exec_file_write_sets_path_and_preview() -> None:
    backend = _CodexAppServerBackend()

    event = backend._build_status_event(
        method="item/completed",
        params={
            "item": {
                "id": "item-1",
                "type": "command_execution",
                "command": "/bin/bash -lc \"cat > /website/rollup.config.js <<'EOF'\nexport default {};\nEOF\"",
            }
        },
    )

    assert event["kind"] == "exec"
    assert event["headline"] == "Created /website/rollup.config.js"
    assert event["path"] == "/website/rollup.config.js"
    assert event["preview"] == "export default {};"


def test_build_status_event_diff_sets_path_and_structured_preview() -> None:
    backend = _CodexAppServerBackend()

    event = backend._build_status_event(
        method="item/completed",
        params={
            "item": {
                "id": "item-1",
                "type": "file_change",
                "changes": [
                    {
                        "path": "src/main.tsx",
                        "kind": {"type": "edit"},
                        "diff": "--- a/src/main.tsx\n+++ b/src/main.tsx\n@@ -1 +1 @@\n-console.log('old');\n+console.log('new');",
                    }
                ],
            }
        },
    )

    assert event["kind"] == "diff"
    assert event["path"] == "src/main.tsx"

    preview = json.loads(event["preview"])
    assert preview == {
        "changes": [
            {
                "path": "src/main.tsx",
                "kind": "edit",
                "diff": "--- a/src/main.tsx\n+++ b/src/main.tsx\n@@ -1 +1 @@\n-console.log('old');\n+console.log('new');",
            }
        ]
    }


@pytest.mark.asyncio
async def test_handle_custom_event_persists_preview_and_path_without_raw_data() -> None:
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
    preview = json.dumps(
        {
            "changes": [
                {
                    "path": "src/main.tsx",
                    "diff": "--- a/src/main.tsx\n+++ b/src/main.tsx\n@@ -1 +1 @@\n-console.log('old');\n+console.log('new');",
                }
            ]
        }
    )

    await adapter.handle_custom_event(
        messages=messages,
        event={
            "type": "agent.event",
            "kind": "diff",
            "state": "completed",
            "method": "turn/diff/completed",
            "item_id": "diff-1",
            "path": "src/main.tsx",
            "preview": preview,
            "headline": "Edited src/main.tsx (+1 -1)",
            "data": '{"raw":"payload"}',
        },
    )

    assert len(messages.children) == 1
    event_element = messages.children[0]
    assert event_element.tag_name == "event"
    assert event_element.get_attribute("path") == "src/main.tsx"
    assert event_element.get_attribute("preview") == preview
    assert event_element.get_attribute("data") is None


@pytest.mark.asyncio
async def test_codex_next_live_delta_build_matches_done_output() -> None:
    if os.getenv("MESHAGENT_CODEX_LIVE_TEST") != "1":
        pytest.skip("set MESHAGENT_CODEX_LIVE_TEST=1 to run live codex integration")
    if shutil.which("codex") is None:
        pytest.skip("codex executable not found on PATH")

    backend = _CodexAppServerBackend(forward_stdout=False, forward_stderr=False)
    thread_key = f"thread:live:{uuid.uuid4()}"
    context = AgentSessionContext()
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

    streamed_text = _streamed_text_from_events(emitted_events)
    done_events = _completed_texts_from_events(emitted_events)

    assert streamed_text != ""
    assert done_events
    assert streamed_text == done_events[-1]
    assert result == done_events[-1]


@pytest.mark.asyncio
async def test_codex_steer_sends_turn_steer_for_active_turn() -> None:
    backend = _CodexAppServerBackend()
    fake_session = _FakeSession(notifications=[])
    backend._session = fake_session

    context = AgentSessionContext()
    await backend._set_thread_state(
        thread_key="thread:test",
        thread_id="thread-1",
        context=context,
    )
    await backend._track_active_turn(
        thread_key="thread:test",
        thread_id="thread-1",
        turn_id="turn-1",
    )

    await backend.steer(thread_key="thread:test", message="keep going")

    assert fake_session._requests[-1][0] == "turn/steer"
    assert fake_session._requests[-1][1] == {
        "threadId": "thread-1",
        "expectedTurnId": "turn-1",
        "input": [{"type": "text", "text": "keep going"}],
    }


@pytest.mark.asyncio
async def test_codex_steer_fails_when_no_active_turn() -> None:
    backend = _CodexAppServerBackend()
    fake_session = _FakeSession(notifications=[])
    backend._session = fake_session

    context = AgentSessionContext()
    await backend._set_thread_state(
        thread_key="thread:test",
        thread_id="thread-1",
        context=context,
    )

    with pytest.raises(CodexAppServerError, match="has no active turn to steer"):
        await backend.steer(thread_key="thread:test", message="nudge")


def test_resolve_subprocess_argv_prefers_absolute_executable(monkeypatch) -> None:
    session = _CodexJsonRpcSession(command="codex app-server", env={"PATH": "/tmp/bin"})

    def _which(executable: str, *, path: str | None = None):
        assert executable == "codex"
        assert path == "/tmp/bin"
        return "/tmp/bin/codex"

    monkeypatch.setattr(shutil, "which", _which)

    launch_argv, resolved_executable = session._resolve_subprocess_argv(
        argv=["codex", "app-server"]
    )
    assert launch_argv == ["/tmp/bin/codex", "app-server"]
    assert resolved_executable == "/tmp/bin/codex"


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


@pytest.mark.asyncio
async def test_session_dispatch_matches_non_numeric_string_response_id() -> None:
    session = _CodexJsonRpcSession(command="codex app-server")
    loop = asyncio.get_running_loop()
    future = loop.create_future()
    session._pending["turn:steer:1"] = future

    await session._dispatch_message({"id": "turn:steer:1", "result": {"ok": True}})

    assert future.done()
    assert future.result() == {"ok": True}


@pytest.mark.asyncio
async def test_session_dispatch_matches_integral_float_response_id() -> None:
    session = _CodexJsonRpcSession(command="codex app-server")
    loop = asyncio.get_running_loop()
    future = loop.create_future()
    session._pending[7] = future

    await session._dispatch_message({"id": 7.0, "result": "ok"})

    assert future.done()
    assert future.result() == "ok"
