from __future__ import annotations

import logging
import posixpath
import uuid
from collections.abc import Callable
from datetime import datetime, timezone
from typing import Literal, Protocol

from meshagent.agents.messages import (
    AGENT_EVENT_MODEL_CHANGED,
    AGENT_MESSAGE_CAPABILITIES_RESPONSE,
    AGENT_MESSAGE_MODELS_RESPONSE,
    AgentMessage,
    AgentModelChanged,
    AgentModelInfo,
    AgentProviderInfo,
    CapabilitiesRequest,
    CapabilitiesResponse,
    ModelsRequest,
    ModelsResponse,
    StartThread,
    ToolkitCapabilities,
)
from meshagent.agents.process import (
    AgentProcess,
    AgentSupervisor,
    Message,
    ThreadIsolationMode,
)
from meshagent.agents.thread_status_publisher import AgentMessageThreadStatusPublisher
from meshagent.agents.thread_storage import (
    ThreadListEntry,
    ThreadListPage,
    ThreadStorage,
)
from meshagent.api import Participant, RoomClient
from meshagent.api.agent_content import AgentFileContent, AgentTextContent

from .thread_storage import CodexThreadStorageRepository
from .vendor.openai_codex.async_client import AsyncAppServerClient
from .vendor.openai_codex.client import AppServerConfig
from .vendor.openai_codex.generated.v2_all import ModelListResponse, ThreadStartResponse

logger = logging.getLogger("meshagent.codex.supervisor")

DEFAULT_CODEX_MODEL = "gpt-5.5"
CodexThreadStorageMode = Literal["none", "codex", "dataset"]


class CodexClient(Protocol):
    async def start(self) -> None: ...

    async def close(self) -> None: ...

    async def initialize(self): ...

    async def thread_start(
        self,
        params: dict | None = None,
    ) -> ThreadStartResponse: ...

    async def thread_list(self, params: dict | None = None): ...

    async def thread_archive(self, thread_id: str): ...

    async def thread_set_name(self, thread_id: str, name: str): ...

    async def model_list(self, include_hidden: bool = False) -> ModelListResponse: ...


CodexClientFactory = Callable[[AppServerConfig | None], CodexClient]


class CodexAgentSupervisor(AgentSupervisor):
    """AgentSupervisor backed by the vendored Codex async app-server client."""

    def __init__(
        self,
        *,
        participant: Participant,
        config: AppServerConfig | None = None,
        client_factory: CodexClientFactory | None = None,
        default_model: str | None = None,
        provider_name: str = "codex",
        provider_friendly_name: str = "Codex",
        thread_isolation: ThreadIsolationMode = "global",
        thread_storage: CodexThreadStorageMode = "codex",
        thread_dir: str | None = None,
        room: RoomClient | None = None,
    ) -> None:
        super().__init__(thread_isolation=thread_isolation)
        self._participant = participant
        self._room = room
        self._config = config
        self._client_factory = client_factory or AsyncAppServerClient
        self._client: CodexClient | None = None
        self._default_model = default_model
        self._provider_name = provider_name
        self._provider_friendly_name = provider_friendly_name
        self._model_infos: list[AgentModelInfo] = []
        self._thread_storage = thread_storage
        self._thread_dir = thread_dir
        self._codex_thread_storage_repository: CodexThreadStorageRepository | None = (
            None
        )

    @property
    def client(self) -> CodexClient:
        if self._client is None:
            raise RuntimeError("CodexAgentSupervisor has not been started")
        return self._client

    @property
    def default_model(self) -> str:
        if self._default_model is not None and self._default_model.strip() != "":
            return self._default_model
        if len(self._model_infos) > 0:
            return self._model_infos[0].name
        return DEFAULT_CODEX_MODEL

    @property
    def provider_name(self) -> str:
        return self._provider_name

    @property
    def codex_thread_storage_repository(self) -> CodexThreadStorageRepository:
        repository = self._codex_thread_storage_repository
        if repository is None:
            raise RuntimeError("CodexAgentSupervisor has not been started")
        return repository

    def provider_info(self, current_model: str | None = None) -> AgentProviderInfo:
        active_model = current_model or self.default_model
        models = [
            model.model_copy(update={"active": model.name == active_model})
            for model in self._model_infos
        ]
        if len(models) == 0:
            models = [
                AgentModelInfo(
                    name=self.default_model,
                    friendly_name=self.default_model,
                    modalities=["text"],
                    active=True,
                )
            ]
        return AgentProviderInfo(
            name=self._provider_name,
            friendly_name=self._provider_friendly_name,
            default_model=self.default_model,
            models=models,
        )

    async def on_start(self) -> None:
        client = self._client_factory(self._config)
        await client.start()
        await client.initialize()
        self._client = client
        if self._thread_storage == "codex":
            self._codex_thread_storage_repository = CodexThreadStorageRepository(
                client=client,
                default_model=lambda: self.default_model,
            )
        await self._refresh_models()

    async def on_stop(self) -> None:
        client = self._client
        self._client = None
        self._codex_thread_storage_repository = None
        if client is not None:
            await client.close()

    async def _refresh_models(self) -> None:
        try:
            response = await self.client.model_list(include_hidden=False)
        except Exception:
            logger.debug("unable to load Codex model list", exc_info=True)
            return

        models: list[AgentModelInfo] = []
        default_model: str | None = None
        for model in response.data:
            model_name = model.model.strip() if model.model.strip() != "" else model.id
            if model.is_default and default_model is None:
                default_model = model_name
            models.append(
                AgentModelInfo(
                    name=model_name,
                    friendly_name=model.display_name,
                    description=model.description,
                    modalities=["text"],
                    supports_attachments="image" in (model.input_modalities or []),
                    accepts=["image"]
                    if "image" in (model.input_modalities or [])
                    else [],
                    active=False,
                )
            )
        self._model_infos = models
        if self._default_model is None and default_model is not None:
            self._default_model = default_model

    async def create_thread_id(
        self,
        *,
        start_thread: StartThread,
        sender: Participant | None,
    ) -> str:
        del sender
        if self._thread_storage == "dataset":
            return self._new_dataset_thread_id()
        if self._thread_storage == "none":
            return self._new_tmp_thread_id()

        return await self.codex_thread_storage_repository.create_thread_id(
            start_thread=start_thread,
        )

    def create_thread_process(self, thread_id: str) -> AgentProcess:
        from .process import CodexAgentProcess

        def publish(payload: AgentMessage) -> None:
            self.emit(sender=self._participant, payload=payload)

        return CodexAgentProcess(
            thread_id=thread_id,
            participant=self._participant,
            client=self.client,
            provider_name=self._provider_name,
            default_model=self.default_model,
            provider_info_builder=self.provider_info,
            working_dir=None if self._config is None else self._config.cwd,
            thread_storage=self._create_thread_storage(thread_id=thread_id),
            ephemeral_codex_thread=self._thread_storage != "codex",
            thread_status_publisher=AgentMessageThreadStatusPublisher(
                thread_id=thread_id,
                publish=publish,
            ),
        )

    async def on_models_request(self, message: Message) -> None:
        request = ModelsRequest.model_validate(message.data.model_dump(mode="python"))
        self.emit(
            sender=message.sender,
            payload=ModelsResponse(
                type=AGENT_MESSAGE_MODELS_RESPONSE,
                source_message_id=request.message_id,
                providers=[self.provider_info()],
            ),
        )

    async def on_thread_renamed(
        self,
        *,
        rename_thread,
        sender: Participant | None,
    ) -> ThreadListEntry | None:
        del sender
        if self._thread_storage == "dataset":
            from meshagent.agents.dataset_thread_storage import DatasetThreadStorage

            await DatasetThreadStorage.rename_thread(
                room=self._room_or_raise(),
                thread_dir=self._thread_dir_or_raise(),
                path=rename_thread.thread_id,
                name=rename_thread.name,
            )
            now = datetime.now(timezone.utc).isoformat()
            return ThreadListEntry(
                name=rename_thread.name,
                path=rename_thread.thread_id,
                created_at="",
                modified_at=now,
            )
        if self._thread_storage == "none":
            return None
        return await self.codex_thread_storage_repository.rename_thread(
            thread_id=rename_thread.thread_id,
            name=rename_thread.name,
        )

    async def on_thread_deleted(
        self,
        *,
        delete_thread,
        sender: Participant | None,
    ) -> None:
        del sender
        if self._thread_storage == "dataset":
            from meshagent.agents.dataset_thread_storage import DatasetThreadStorage

            await DatasetThreadStorage.delete_thread(
                room=self._room_or_raise(),
                thread_dir=self._thread_dir_or_raise(),
                path=delete_thread.thread_id,
            )
            return
        if self._thread_storage == "none":
            return
        await self.codex_thread_storage_repository.delete_thread(
            thread_id=delete_thread.thread_id,
        )

    async def list_threads(
        self,
        *,
        list_threads,
        sender: Participant | None,
    ) -> ThreadListPage:
        del sender
        if self._thread_storage == "dataset":
            from meshagent.agents.dataset_thread_storage import DatasetThreadStorage

            return await DatasetThreadStorage.list_threads(
                room=self._room_or_raise(),
                thread_dir=self._thread_dir_or_raise(),
                limit=list_threads.limit,
                offset=list_threads.offset,
            )
        if self._thread_storage == "none":
            return ThreadListPage(
                threads=[],
                total=0,
                offset=list_threads.offset,
                limit=list_threads.limit,
            )
        return await self.codex_thread_storage_repository.list_threads(
            limit=list_threads.limit,
            offset=list_threads.offset,
        )

    async def on_thread_started(
        self,
        *,
        thread_id: str,
        start_thread: StartThread,
        sender: Participant | None,
    ) -> ThreadListEntry | None:
        del sender
        if self._thread_storage != "dataset":
            return None
        from meshagent.agents.dataset_thread_storage import DatasetThreadStorage

        name = _thread_name_from_start_thread(start_thread)
        await DatasetThreadStorage.upsert_thread(
            room=self._room_or_raise(),
            thread_dir=self._thread_dir_or_raise(),
            path=thread_id,
            name=name,
        )
        now = datetime.now(timezone.utc).isoformat()
        return ThreadListEntry(
            name=name,
            path=thread_id,
            created_at=now,
            modified_at=now,
        )

    def _create_thread_storage(self, *, thread_id: str) -> ThreadStorage | None:
        if self._thread_storage == "codex":
            return self.codex_thread_storage_repository.create_thread_storage(
                thread_id=thread_id,
            )
        if self._thread_storage != "dataset":
            return None
        from meshagent.agents.dataset_thread_storage import DatasetThreadStorage

        return DatasetThreadStorage(room=self._room_or_raise(), path=thread_id)

    def _room_or_raise(self) -> RoomClient:
        if self._room is None:
            raise RuntimeError("dataset thread storage requires a room client")
        return self._room

    def _thread_dir_or_raise(self) -> str:
        if self._thread_dir is None or self._thread_dir.strip() == "":
            raise RuntimeError("dataset thread storage requires --thread-dir")
        return self._thread_dir

    def _new_dataset_thread_id(self) -> str:
        thread_dir = self._thread_dir_or_raise().strip().rstrip("/")
        item_id = uuid.uuid4().hex
        if thread_dir.startswith("dataset://"):
            return f"{thread_dir}/{item_id}"
        return f"dataset://{posixpath.join(thread_dir.strip('/'), item_id)}"

    def _new_tmp_thread_id(self) -> str:
        thread_dir = (self._thread_dir or "/tmp/codex/threads").strip().rstrip("/")
        return f"tmp://{posixpath.join(thread_dir.strip('/'), uuid.uuid4().hex)}"

    def _build_capabilities(self) -> list[ToolkitCapabilities]:
        return []

    async def emit_capabilities_response(
        self,
        *,
        request: CapabilitiesRequest,
        sender: Participant | None,
    ) -> None:
        self.emit(
            sender=sender,
            payload=CapabilitiesResponse(
                type=AGENT_MESSAGE_CAPABILITIES_RESPONSE,
                thread_id=request.thread_id,
                source_message_id=request.message_id,
                version="codex",
                toolkits=self._build_capabilities(),
            ),
        )

    def model_changed(
        self,
        *,
        thread_id: str,
        source_message_id: str | None = None,
        model: str | None = None,
    ) -> AgentModelChanged:
        return AgentModelChanged(
            type=AGENT_EVENT_MODEL_CHANGED,
            thread_id=thread_id,
            source_message_id=source_message_id,
            provider=self._provider_name,
            model=model or self.default_model,
            output_modalities=["text"],
            supports_attachments=True,
            accepts=["image"],
        )


def _thread_name_from_start_thread(start_thread: StartThread) -> str:
    if start_thread.name is not None and start_thread.name.strip() != "":
        return start_thread.name.strip()

    text_parts: list[str] = []
    attachment_count = 0
    for item in start_thread.content or []:
        if isinstance(item, AgentTextContent) and item.text.strip() != "":
            text_parts.extend(item.text.strip().split())
        elif isinstance(item, AgentFileContent):
            attachment_count += 1
    if len(text_parts) > 0:
        return " ".join(text_parts[:6])
    if attachment_count > 0:
        return "Attachment Thread"
    return "New Chat"


def new_thread_id() -> str:
    return str(uuid.uuid4())
