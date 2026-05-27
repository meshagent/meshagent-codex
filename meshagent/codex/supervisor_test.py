from __future__ import annotations

import pytest

from meshagent.agents.messages import (
    AGENT_MESSAGE_THREAD_RENAME,
    AGENT_MESSAGE_THREAD_START,
    RenameThread,
    StartThread,
)
from meshagent.api import Participant

from .supervisor import CodexAgentSupervisor


class _FakeDatasetThreadStorage:
    upsert_calls: list[tuple[object, str, str, str]] = []
    rename_calls: list[tuple[object, str, str, str]] = []

    @staticmethod
    async def upsert_thread(*, room, thread_dir: str, path: str, name: str) -> None:
        _FakeDatasetThreadStorage.upsert_calls.append((room, thread_dir, path, name))

    @staticmethod
    async def rename_thread(*, room, thread_dir: str, path: str, name: str) -> None:
        _FakeDatasetThreadStorage.rename_calls.append((room, thread_dir, path, name))


@pytest.fixture(autouse=True)
def _clear_fake_dataset_storage() -> None:
    _FakeDatasetThreadStorage.upsert_calls.clear()
    _FakeDatasetThreadStorage.rename_calls.clear()


@pytest.mark.asyncio
async def test_dataset_thread_started_returns_complete_thread_list_entry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import meshagent.agents.dataset_thread_storage as dataset_thread_storage

    monkeypatch.setattr(
        dataset_thread_storage,
        "DatasetThreadStorage",
        _FakeDatasetThreadStorage,
    )
    room = object()
    supervisor = CodexAgentSupervisor(
        participant=Participant(id="codex", attributes={"name": "codex"}),
        thread_storage="dataset",
        thread_dir="dataset://agents/codex/threads",
        room=room,
    )

    entry = await supervisor.on_thread_started(
        thread_id="dataset://agents/codex/threads/first",
        start_thread=StartThread(
            type=AGENT_MESSAGE_THREAD_START,
            name="First Thread",
        ),
        sender=None,
    )

    assert entry is not None
    assert entry.name == "First Thread"
    assert entry.path == "dataset://agents/codex/threads/first"
    assert entry.created_at != ""
    assert entry.modified_at == entry.created_at
    assert _FakeDatasetThreadStorage.upsert_calls == [
        (
            room,
            "dataset://agents/codex/threads",
            "dataset://agents/codex/threads/first",
            "First Thread",
        )
    ]


@pytest.mark.asyncio
async def test_dataset_thread_renamed_returns_complete_thread_list_entry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import meshagent.agents.dataset_thread_storage as dataset_thread_storage

    monkeypatch.setattr(
        dataset_thread_storage,
        "DatasetThreadStorage",
        _FakeDatasetThreadStorage,
    )
    room = object()
    supervisor = CodexAgentSupervisor(
        participant=Participant(id="codex", attributes={"name": "codex"}),
        thread_storage="dataset",
        thread_dir="dataset://agents/codex/threads",
        room=room,
    )

    entry = await supervisor.on_thread_renamed(
        rename_thread=RenameThread(
            type=AGENT_MESSAGE_THREAD_RENAME,
            thread_id="dataset://agents/codex/threads/first",
            name="Renamed Thread",
        ),
        sender=None,
    )

    assert entry is not None
    assert entry.name == "Renamed Thread"
    assert entry.path == "dataset://agents/codex/threads/first"
    assert entry.created_at == ""
    assert entry.modified_at != ""
    assert _FakeDatasetThreadStorage.rename_calls == [
        (
            room,
            "dataset://agents/codex/threads",
            "dataset://agents/codex/threads/first",
            "Renamed Thread",
        )
    ]
