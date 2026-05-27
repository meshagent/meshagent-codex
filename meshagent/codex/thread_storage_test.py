from __future__ import annotations

from types import SimpleNamespace

import pytest

from meshagent.agents.messages import AGENT_MESSAGE_THREAD_START, StartThread

from .thread_storage import CodexThreadStorage, CodexThreadStorageRepository


class _FakeCodexThreadClient:
    def __init__(self) -> None:
        self.start_params: list[dict | None] = []
        self.archived_thread_ids: list[str] = []
        self.renamed_threads: list[tuple[str, str]] = []

    async def thread_start(self, params: dict | None = None):
        self.start_params.append(params)
        return SimpleNamespace(thread=SimpleNamespace(id="codex-thread-1"))

    async def thread_list(self, params: dict | None = None):
        return SimpleNamespace(
            data=[
                SimpleNamespace(
                    id="thread-1",
                    name="Named Thread",
                    preview="Preview",
                    created_at="2026-01-01T00:00:00Z",
                    updated_at="2026-01-02T00:00:00Z",
                ),
                SimpleNamespace(
                    id="thread-2",
                    name="",
                    preview="Preview Thread",
                    created_at="2026-01-03T00:00:00Z",
                    updated_at="2026-01-04T00:00:00Z",
                ),
            ]
        )

    async def thread_archive(self, thread_id: str):
        self.archived_thread_ids.append(thread_id)

    async def thread_set_name(self, thread_id: str, name: str):
        self.renamed_threads.append((thread_id, name))


@pytest.mark.asyncio
async def test_codex_thread_repository_creates_thread_with_model_and_metadata() -> None:
    client = _FakeCodexThreadClient()
    repository = CodexThreadStorageRepository(
        client=client,
        default_model=lambda: "gpt-default",
    )

    thread_id = await repository.create_thread_id(
        start_thread=StartThread(
            type=AGENT_MESSAGE_THREAD_START,
            name="Test Thread",
            instructions="Be concise.",
        )
    )

    assert thread_id == "codex-thread-1"
    assert client.start_params == [
        {
            "model": "gpt-default",
            "developerInstructions": "Be concise.",
            "config": {"name": "Test Thread"},
        }
    ]


@pytest.mark.asyncio
async def test_codex_thread_repository_lists_renames_and_deletes_threads() -> None:
    client = _FakeCodexThreadClient()
    repository = CodexThreadStorageRepository(
        client=client,
        default_model=lambda: "gpt-default",
    )

    page = await repository.list_threads(limit=20, offset=0)
    renamed = await repository.rename_thread(path="thread-1", name="Renamed")
    await repository.delete_thread(path="thread-2")

    assert [entry.name for entry in page.threads] == ["Named Thread", "Preview Thread"]
    assert renamed is None
    assert client.renamed_threads == [("thread-1", "Renamed")]
    assert client.archived_thread_ids == ["thread-2"]


def test_codex_thread_storage_is_non_ephemeral_noop_storage() -> None:
    storage = CodexThreadStorage(path="thread-1")

    storage.push_message(message=object(), sender=None)

    assert storage.path == "thread-1"
    assert storage.is_ephemeral is False
    assert storage.agent_messages() == []
    assert storage.unflushed_agent_messages() == []


def test_codex_thread_repository_does_not_create_thread_storage() -> None:
    repository = CodexThreadStorageRepository(
        client=_FakeCodexThreadClient(),
        default_model=lambda: "gpt-default",
    )

    assert not hasattr(repository, "create_thread_storage")
