import asyncio

import pytest

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
