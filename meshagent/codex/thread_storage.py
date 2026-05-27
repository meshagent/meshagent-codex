from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Callable
from typing import Any, Protocol

from meshagent.agents.context import AgentSessionContext
from meshagent.agents.messages import StartThread
from meshagent.agents.thread_storage import (
    ThreadListEntry,
    ThreadListEvent,
    ThreadListPage,
)
from meshagent.api import Participant
from meshagent.tools import Toolkit


class CodexThreadClient(Protocol):
    async def thread_start(self, params: dict | None = None) -> Any: ...

    async def thread_list(self, params: dict | None = None) -> Any: ...

    async def thread_archive(self, thread_id: str) -> Any: ...

    async def thread_set_name(self, thread_id: str, name: str) -> Any: ...


class CodexThreadStorage:
    """ThreadStorage facade for Codex-native thread persistence.

    Codex owns the durable transcript for this mode. The Meshagent storage
    facade exists so generic process code can attach thread tools and lifecycle
    hooks without mirroring messages into another store.
    """

    def __init__(self, *, path: str, is_ephemeral: bool = False) -> None:
        self._path = path
        self._is_ephemeral = is_ephemeral

    @property
    def path(self) -> str:
        return self._path

    @property
    def is_ephemeral(self) -> bool:
        return self._is_ephemeral

    async def start(self) -> None:
        return None

    async def stop(self) -> None:
        return None

    async def wait_until_ready(self) -> None:
        return None

    def unflushed_agent_messages(self) -> list[Any]:
        return []

    def agent_messages(self) -> list[Any]:
        return []

    def push_message(
        self,
        *,
        message: Any,
        sender: Participant | None = None,
    ) -> None:
        del message
        del sender

    def restore_session_context(
        self,
        *,
        context: AgentSessionContext,
        llm_adapter: Any = None,
    ) -> None:
        del context
        del llm_adapter

    async def restore_session_context_async(
        self,
        *,
        context: AgentSessionContext,
        llm_adapter: Any = None,
    ) -> None:
        del context
        del llm_adapter

    def make_toolkit(self) -> Toolkit:
        return Toolkit(name="codex_thread_storage", tools=[])


class CodexThreadStorageRepository:
    def __init__(
        self,
        *,
        client: CodexThreadClient | None = None,
        client_provider: Callable[[], CodexThreadClient] | None = None,
        default_model: Callable[[], str],
    ) -> None:
        self._client = client
        self._client_provider = client_provider
        self._default_model = default_model

    @property
    def client(self) -> CodexThreadClient:
        if self._client is not None:
            return self._client
        if self._client_provider is not None:
            return self._client_provider()
        raise RuntimeError("Codex thread storage repository has no client")

    @property
    def is_ephemeral(self) -> bool:
        return False

    def thread_list_path(self) -> str:
        return ""

    async def create_thread_id(self, *, start_thread: StartThread) -> str:
        params: dict[str, object] = {
            "model": start_thread.model or self._default_model(),
        }
        if start_thread.instructions is not None:
            params["developerInstructions"] = start_thread.instructions
        if start_thread.name is not None:
            params["config"] = {"name": start_thread.name}
        response = await self.client.thread_start(params)
        return response.thread.id

    async def rename_thread(
        self,
        *,
        path: str,
        name: str,
    ) -> ThreadListEntry | None:
        await self.client.thread_set_name(path, name)
        return None

    async def delete_thread(
        self,
        *,
        path: str,
        delete_storage: bool = True,
    ) -> None:
        del delete_storage
        await self.client.thread_archive(path)

    async def upsert_thread(
        self,
        *,
        path: str,
        name: str | None = None,
        created_at: str | None = None,
        modified_at: str | None = None,
    ) -> ThreadListEntry | None:
        del path
        del name
        del created_at
        del modified_at
        return None

    async def list_threads(self, *, limit: int, offset: int) -> ThreadListPage:
        response = await self.client.thread_list(
            {
                "limit": limit,
                "cursor": None,
                "archived": False,
            }
        )
        entries: list[ThreadListEntry] = []
        for thread in response.data:
            name = thread.name.strip() if isinstance(thread.name, str) else ""
            if name == "":
                name = thread.preview.strip()
            if name == "":
                name = thread.id
            entries.append(
                ThreadListEntry(
                    name=name,
                    path=thread.id,
                    created_at=str(thread.created_at),
                    modified_at=str(thread.updated_at),
                )
            )
        return ThreadListPage(
            threads=entries,
            total=len(entries),
            offset=offset,
            limit=limit,
        )

    async def watch_threads(
        self,
        *,
        poll_interval: float = 1.0,
    ) -> AsyncIterator[ThreadListEvent]:
        del poll_interval
        await asyncio.Event().wait()
        if False:
            yield
