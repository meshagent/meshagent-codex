import asyncio

import pytest
from meshagent.agents import AgentSessionContext
from meshagent.agents.chat import ChatThreadContext
from meshagent.api.specs.service import ContainerMountSpec, RoomStorageMountSpec

from meshagent.codex.app_server import CodexAppServerError
from meshagent.codex.chatbot import CodexChatBot


@pytest.mark.asyncio
async def test_codex_cancel_keeps_thread_worker_running() -> None:
    bot = CodexChatBot(name="codex-test")
    bot._thread_tasks["/threads/test"] = asyncio.create_task(asyncio.sleep(3600))
    try:
        await bot.cancel_thread_task(path="/threads/test", thread_context=None)
        assert not bot._thread_tasks["/threads/test"].cancelled()
    finally:
        bot._thread_tasks["/threads/test"].cancel()


@pytest.mark.asyncio
async def test_clear_thread_status_ignores_stale_nowait_update(monkeypatch) -> None:
    bot = CodexChatBot(name="codex-test")
    path = "/threads/test"
    updates: list[tuple[str, str | None]] = []

    async def _capture_set(
        *, path: str, status: str | None, mode: str | None = None
    ) -> None:
        del mode
        updates.append((path, status))

    monkeypatch.setattr(bot, "set_thread_status", _capture_set)

    bot._set_thread_status_nowait(path=path, status="Applying diff")
    await bot.clear_thread_status(path=path)
    await asyncio.sleep(0)
    await asyncio.sleep(0)

    assert updates == [(path, None)]


@pytest.mark.asyncio
async def test_diff_status_events_do_not_restore_status_after_turn_clear(
    monkeypatch,
) -> None:
    bot = CodexChatBot(name="codex-test")
    path = "/threads/test"
    updates: list[tuple[str, str | None]] = []

    async def _capture_set(
        *, path: str, status: str | None, mode: str | None = None
    ) -> None:
        del mode
        updates.append((path, status))

    monkeypatch.setattr(bot, "set_thread_status", _capture_set)

    bot._update_thread_status_from_event(
        path=path,
        event={
            "type": "agent.event",
            "kind": "diff",
            "state": "in_progress",
            "correlation_key": "turn.diff:turn-1",
            "headline": "Applying diff",
        },
    )
    bot._update_thread_status_from_event(
        path=path,
        event={
            "type": "agent.event",
            "kind": "diff",
            "state": "completed",
            "correlation_key": "turn.diff:turn-1",
            "headline": "Diff updated",
        },
    )

    # Mirror turn teardown behavior after task completion.
    await bot.clear_thread_status(path=path)
    await asyncio.sleep(0)
    await asyncio.sleep(0)

    assert updates == [(path, None)]


def test_codex_status_completion_keeps_fallback_status(monkeypatch) -> None:
    bot = CodexChatBot(name="codex-test")
    path = "/threads/test"
    updates: list[tuple[str, str]] = []
    cleared: list[str] = []

    def _capture_set(*, path: str, status: str | None) -> None:
        if status is not None:
            updates.append((path, status))

    def _capture_clear(*, path: str) -> None:
        cleared.append(path)

    monkeypatch.setattr(bot, "_set_thread_status_nowait", _capture_set)
    monkeypatch.setattr(bot, "_clear_thread_status_nowait", _capture_clear)

    bot._update_thread_status_from_event(
        path=path,
        event={
            "type": "codex.event",
            "kind": "tool",
            "state": "in_progress",
            "event_key": "tool-1",
            "headline": "Running tool",
        },
    )
    bot._update_thread_status_from_event(
        path=path,
        event={
            "type": "codex.event",
            "kind": "tool",
            "state": "completed",
            "event_key": "tool-1",
        },
    )

    assert updates == [(path, "Running tool"), (path, "Thinking")]
    assert cleared == []
    assert bot._thread_status_keys.get(path) is None


def test_codex_status_completion_without_key_keeps_fallback_status(monkeypatch) -> None:
    bot = CodexChatBot(name="codex-test")
    path = "/threads/test"
    updates: list[tuple[str, str]] = []
    cleared: list[str] = []

    def _capture_set(*, path: str, status: str | None) -> None:
        if status is not None:
            updates.append((path, status))

    def _capture_clear(*, path: str) -> None:
        cleared.append(path)

    monkeypatch.setattr(bot, "_set_thread_status_nowait", _capture_set)
    monkeypatch.setattr(bot, "_clear_thread_status_nowait", _capture_clear)

    bot._update_thread_status_from_event(
        path=path,
        event={
            "type": "codex.event",
            "kind": "tool",
            "state": "completed",
            "headline": "done",
        },
    )

    assert updates == [(path, "Thinking")]
    assert cleared == []


@pytest.mark.asyncio
async def test_message_to_turn_input_uses_local_image_with_default_room_mount() -> None:
    bot = CodexChatBot(name="codex-test")

    turn_input = await bot._message_to_turn_input(
        message={
            "text": "check this",
            "attachments": [{"path": "test.jpg"}],
        }
    )

    assert turn_input == [
        {"type": "text", "text": "check this"},
        {"type": "localImage", "path": "/data/test.jpg"},
    ]


@pytest.mark.asyncio
async def test_message_to_turn_input_adds_text_for_non_image_attachment() -> None:
    bot = CodexChatBot(name="codex-test")

    turn_input = await bot._message_to_turn_input(
        message={
            "text": "",
            "attachments": [{"path": "docs/report.pdf"}],
        }
    )

    assert turn_input == [
        {"type": "text", "text": "file attached /data/docs/report.pdf"}
    ]


@pytest.mark.asyncio
async def test_message_to_turn_input_respects_room_mount_subpath() -> None:
    bot = CodexChatBot(
        name="codex-test",
        mounts=ContainerMountSpec(
            room=[
                RoomStorageMountSpec(path="/data"),
                RoomStorageMountSpec(path="/images", subpath="assets"),
            ]
        ),
    )

    turn_input = await bot._message_to_turn_input(
        message={
            "text": "",
            "attachments": [{"path": "assets/photo.png"}],
        }
    )

    assert turn_input == [{"type": "localImage", "path": "/images/photo.png"}]


class _FakeMessagesElement:
    def __init__(self):
        self._attrs: dict[str, str] = {}

    def get_attribute(self, key: str):
        return self._attrs.get(key)

    def set_attribute(self, key: str, value):
        if isinstance(value, str):
            self._attrs[key] = value
        elif value is None:
            self._attrs.pop(key, None)


class _FakeRoot:
    def __init__(self, messages: _FakeMessagesElement):
        self._messages = messages

    def get_children_by_tag_name(self, tag_name: str):
        if tag_name == "messages":
            return [self._messages]
        return []


class _FakeThread:
    def __init__(self, root: _FakeRoot):
        self.root = root


class _FakeBackend:
    def __init__(self):
        self.cleared: list[tuple[str, AgentSessionContext]] = []

    async def on_thread_clear(self, *, thread_key: str, context: AgentSessionContext):
        self.cleared.append((thread_key, context))


class _FakeSteerBackend:
    def __init__(self, *, error: Exception | None = None):
        self.error = error
        self.steer_calls: list[tuple[str, object]] = []

    async def steer(self, *, thread_key: str, message):
        self.steer_calls.append((thread_key, message))
        if self.error is not None:
            raise self.error


class _FakeParticipant:
    def __init__(self, name: str):
        self._name = name

    def get_attribute(self, key: str):
        if key == "name":
            return self._name
        return None


@pytest.mark.asyncio
async def test_on_thread_clear_resets_external_thread_id() -> None:
    bot = CodexChatBot(name="codex-test")
    fake_backend = _FakeBackend()
    bot._codex_backend = fake_backend

    messages = _FakeMessagesElement()
    messages.set_attribute("external_thread_id", "thread-123")
    thread_context = ChatThreadContext(
        path="/threads/test",
        thread=_FakeThread(root=_FakeRoot(messages=messages)),  # type: ignore[arg-type]
        participants=[],
        session=AgentSessionContext(),
    )

    await bot.on_thread_clear(thread_context=thread_context)

    assert messages.get_attribute("external_thread_id") == ""
    assert bot._external_thread_id_from_thread(thread_context=thread_context) is None
    assert fake_backend.cleared == [("/threads/test", thread_context.session)]


@pytest.mark.asyncio
async def test_on_thread_steer_falls_back_to_chat_when_no_active_turn(
    monkeypatch,
) -> None:
    bot = CodexChatBot(name="codex-test")
    fake_backend = _FakeSteerBackend(
        error=CodexAppServerError(
            "codex thread '/threads/test' has no active turn to steer"
        )
    )
    bot._codex_backend = fake_backend

    thread_context = ChatThreadContext(
        path="/threads/test",
        thread=_FakeThread(root=_FakeRoot(messages=_FakeMessagesElement())),  # type: ignore[arg-type]
        participants=[],
        session=AgentSessionContext(),
    )
    participant = _FakeParticipant("tester@example.com")
    payload = {"text": "continue", "attachments": []}
    fallback_calls: list[dict] = []

    async def _fake_chat_received(*, thread_context, from_participant, message):
        fallback_calls.append(
            {
                "path": thread_context.path,
                "name": from_participant.get_attribute("name"),
                "message": message,
            }
        )
        return "ok"

    monkeypatch.setattr(bot, "on_chat_received", _fake_chat_received)

    await bot.on_thread_steer(
        thread_context=thread_context,
        from_participant=participant,  # type: ignore[arg-type]
        message=payload,
    )

    assert fake_backend.steer_calls == [
        ("/threads/test", [{"type": "text", "text": "continue"}])
    ]
    assert fallback_calls == [
        {
            "path": "/threads/test",
            "name": "tester@example.com",
            "message": payload,
        }
    ]


@pytest.mark.asyncio
async def test_on_thread_steer_does_not_fallback_when_active_turn_exists(
    monkeypatch,
) -> None:
    bot = CodexChatBot(name="codex-test")
    fake_backend = _FakeSteerBackend()
    bot._codex_backend = fake_backend

    thread_context = ChatThreadContext(
        path="/threads/test",
        thread=_FakeThread(root=_FakeRoot(messages=_FakeMessagesElement())),  # type: ignore[arg-type]
        participants=[],
        session=AgentSessionContext(),
    )
    participant = _FakeParticipant("tester@example.com")
    payload = {"text": "continue", "attachments": []}
    fallback_calls: list[dict] = []

    async def _fake_chat_received(*, thread_context, from_participant, message):
        fallback_calls.append(
            {
                "path": thread_context.path,
                "name": from_participant.get_attribute("name"),
                "message": message,
            }
        )
        return "ok"

    monkeypatch.setattr(bot, "on_chat_received", _fake_chat_received)

    await bot.on_thread_steer(
        thread_context=thread_context,
        from_participant=participant,  # type: ignore[arg-type]
        message=payload,
    )

    assert fake_backend.steer_calls == [
        ("/threads/test", [{"type": "text", "text": "continue"}])
    ]
    assert fallback_calls == []
