from __future__ import annotations

from datetime import datetime, timezone

import pytest

from meshagent.agents.messages import (
    AGENT_MESSAGE_THREAD_RENAME,
    AGENT_MESSAGE_THREAD_START,
    RenameThread,
    StartThread,
)
from meshagent.agents.thread_storage import ThreadListEntry
from meshagent.api import Participant

from .supervisor import CodexAgentSupervisor


class _FakeDatasetThreadStorage:
    upsert_calls: list[tuple[object, str, str, str]] = []
    rename_calls: list[tuple[object, str, str, str]] = []

    def __init__(
        self,
        *,
        room,
        path: str | None = None,
        thread_dir: str | None = None,
    ) -> None:
        del path
        self.room = room
        self.thread_dir = thread_dir or ""

    async def upsert_thread(self, *, path: str, name: str) -> ThreadListEntry:
        _FakeDatasetThreadStorage.upsert_calls.append(
            (self.room, self.thread_dir, path, name)
        )
        now = datetime.now(timezone.utc).isoformat()
        return ThreadListEntry(
            name=name,
            path=path,
            created_at=now,
            modified_at=now,
        )

    async def rename_thread(self, *, path: str, name: str) -> ThreadListEntry:
        _FakeDatasetThreadStorage.rename_calls.append(
            (self.room, self.thread_dir, path, name)
        )
        return ThreadListEntry(
            name=name,
            path=path,
            created_at="",
            modified_at=datetime.now(timezone.utc).isoformat(),
        )


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
