import asyncio

import pytest
from meshagent.agents import AgentSessionContext
from meshagent.agents.chat import ChatThreadContext
from meshagent.api.specs.service import ContainerMountSpec, RoomStorageMountSpec

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
