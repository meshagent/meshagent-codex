from __future__ import annotations

import logging
import posixpath
import uuid
from collections.abc import Callable
from typing import Literal, Protocol

from meshagent.agents.messages import (
    AGENT_EVENT_MODEL_CHANGED,
    AGENT_MESSAGE_CAPABILITIES_RESPONSE,
    AgentError,
    AgentMessage,
    AgentModelChanged,
    AgentModelInfo,
    AgentProviderInfo,
    CapabilitiesRequest,
    CapabilitiesResponse,
    StartThread,
    ToolkitCapabilities,
    TurnStart,
)
from meshagent.agents.process import (
    AgentProcess,
    AgentSupervisor,
    ThreadIsolationMode,
)
from meshagent.agents.thread_status_publisher import AgentMessageThreadStatusPublisher
from meshagent.agents.thread_storage import (
    NoopThreadStorageRepository,
    ThreadStorage,
    ThreadStorageRepository,
)
from meshagent.api import Participant, RoomClient

from .thread_storage import CodexThreadStorage, CodexThreadStorageRepository
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


class CodexBackend:
    def __init__(
        self,
        *,
        participant: Participant,
        config: AppServerConfig | None,
        client_factory: CodexClientFactory,
        default_model: str | None,
        provider_name: str,
        provider_friendly_name: str,
        thread_storage: CodexThreadStorageMode,
        thread_dir: str | None,
        room: RoomClient | None,
    ) -> None:
        self._participant = participant
        self._config = config
        self._client_factory = client_factory
        self._client: CodexClient | None = None
        self._default_model = default_model
        self._provider_name = provider_name
        self._provider_friendly_name = provider_friendly_name
        self._model_infos: list[AgentModelInfo] = []
        self._thread_storage = thread_storage
        self._thread_dir = thread_dir
        self._room = room

    @property
    def name(self) -> str:
        return "codex"

    @property
    def client(self) -> CodexClient:
        if self._client is None:
            raise RuntimeError("Codex backend has not been started")
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

    async def on_start(self) -> None:
        client = self._client_factory(self._config)
        await client.start()
        await client.initialize()
        self._client = client
        await self._refresh_models()

    async def on_stop(self) -> None:
        client = self._client
        self._client = None
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
            backend=self.name,
            default_model=self.default_model,
            models=models,
        )

    def model_providers(
        self,
        *,
        current_backend: str | None,
        current_provider: str | None,
        current_model: str | None,
    ) -> list[AgentProviderInfo]:
        del current_backend
        del current_provider
        return [self.provider_info(current_model=current_model)]

    async def validate_turn_start(self, turn_start: TurnStart) -> AgentError | None:
        if (
            turn_start.backend is not None
            and turn_start.backend.strip() != ""
            and turn_start.backend != self.name
        ):
            return AgentError(
                message=f"unknown backend {turn_start.backend!r}",
                code="unknown_backend",
            )
        unsupported = [
            output
            for output in (turn_start.output_modalities or [])
            if output != "text"
        ]
        if len(unsupported) > 0:
            unsupported_text = ", ".join(repr(item) for item in unsupported)
            return AgentError(
                message=f"Codex does not support {unsupported_text} output modalities",
                code="unsupported_modality",
            )
        requested_model = turn_start.model
        if (
            requested_model is not None
            and requested_model.strip() != ""
            and len(self._model_infos) > 0
            and not any(model.name == requested_model for model in self._model_infos)
        ):
            names = ", ".join(model.name for model in self._model_infos)
            return AgentError(
                message=(
                    f"unknown model {requested_model!r} for provider "
                    f"{self._provider_name!r}; available models: {names}"
                ),
                code="unknown_model",
            )
        return None

    async def create_realtime_connection(
        self,
        *,
        supervisor: AgentSupervisor,
        thread_id: str,
        start_thread: StartThread,
        sender: Participant | None,
    ):
        del supervisor
        del thread_id
        del start_thread
        del sender
        return None

    async def create_thread_id(
        self,
        *,
        supervisor: AgentSupervisor,
        start_thread: StartThread,
        sender: Participant | None,
    ) -> str:
        del supervisor
        del sender
        if self._thread_storage == "dataset":
            return self._new_dataset_thread_id()
        if self._thread_storage == "none":
            return self._new_tmp_thread_id()
        return await CodexThreadStorageRepository(
            client_provider=lambda: self.client,
            default_model=lambda: self.default_model,
        ).create_thread_id(start_thread=start_thread)

    def create_thread_process(
        self,
        *,
        supervisor: AgentSupervisor,
        thread_id: str,
    ) -> AgentProcess:
        from .process import CodexAgentProcess

        def publish(payload: AgentMessage) -> None:
            supervisor.emit(sender=self._participant, payload=payload)

        thread_storage: ThreadStorage | None
        if self._thread_storage == "codex":
            thread_storage = CodexThreadStorage(path=thread_id)
        elif self._thread_storage == "dataset":
            from meshagent.agents.dataset_thread_storage import DatasetThreadStorage

            thread_storage = DatasetThreadStorage(
                room=self._room_or_raise(),
                path=thread_id,
            )
        else:
            thread_storage = None

        return CodexAgentProcess(
            thread_id=thread_id,
            participant=self._participant,
            client=self.client,
            provider_name=self._provider_name,
            default_model=self.default_model,
            provider_info_builder=self.provider_info,
            backend_name=self.name,
            working_dir=None if self._config is None else self._config.cwd,
            thread_storage=thread_storage,
            ephemeral_codex_thread=self._thread_storage != "codex",
            thread_status_publisher=AgentMessageThreadStatusPublisher(
                thread_id=thread_id,
                publish=publish,
            ),
        )

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
        codex_backend = CodexBackend(
            participant=participant,
            config=config,
            client_factory=client_factory or AsyncAppServerClient,
            default_model=default_model,
            provider_name=provider_name,
            provider_friendly_name=provider_friendly_name,
            thread_storage=thread_storage,
            thread_dir=thread_dir,
            room=room,
        )

        thread_storage_repository: ThreadStorageRepository
        codex_thread_storage_repository: CodexThreadStorageRepository | None = None
        if thread_storage == "codex":
            codex_thread_storage_repository = CodexThreadStorageRepository(
                client_provider=lambda: codex_backend.client,
                default_model=lambda: codex_backend.default_model,
            )
            thread_storage_repository = codex_thread_storage_repository
        elif thread_storage == "dataset":
            from meshagent.agents import dataset_thread_storage

            if room is None:
                raise RuntimeError("dataset thread storage requires a room client")
            if thread_dir is None or thread_dir.strip() == "":
                raise RuntimeError("dataset thread storage requires --thread-dir")
            thread_storage_repository = dataset_thread_storage.DatasetThreadStorage(
                room=room,
                thread_dir=thread_dir,
            )
        else:
            thread_storage_repository = NoopThreadStorageRepository()

        super().__init__(
            thread_isolation=thread_isolation,
            thread_storage_repository=thread_storage_repository,
            agent_backends=[codex_backend],
        )
        self._codex_backend = codex_backend
        self._codex_thread_storage_repository = codex_thread_storage_repository

    @property
    def client(self) -> CodexClient:
        return self._codex_backend.client

    @property
    def default_model(self) -> str:
        return self._codex_backend.default_model

    @property
    def provider_name(self) -> str:
        return self._codex_backend.provider_name

    @property
    def codex_thread_storage_repository(self) -> CodexThreadStorageRepository:
        repository = self._codex_thread_storage_repository
        if repository is None:
            raise RuntimeError("CodexAgentSupervisor has not been started")
        return repository

    def provider_info(self, current_model: str | None = None) -> AgentProviderInfo:
        return self._codex_backend.provider_info(current_model=current_model)

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
            provider=self.provider_name,
            backend=self._codex_backend.name,
            model=model or self.default_model,
            output_modalities=["text"],
            supports_attachments=True,
            accepts=["image"],
        )


def new_thread_id() -> str:
    return str(uuid.uuid4())
