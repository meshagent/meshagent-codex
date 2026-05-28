from __future__ import annotations

import asyncio
import contextlib
import inspect
import os
import shlex
import shutil
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

import pytest

from meshagent.agents.messages import (
    AGENT_EVENT_MODEL_CHANGED,
    AGENT_EVENT_TEXT_CONTENT_DELTA,
    AGENT_EVENT_THREAD_LOADED,
    AGENT_EVENT_THREAD_STATUS,
    AGENT_EVENT_TURN_ENDED,
    AGENT_EVENT_TURN_INTERRUPTED,
    AGENT_EVENT_TURN_INTERRUPT_ACCEPTED,
    AGENT_EVENT_TURN_START_ACCEPTED,
    AGENT_EVENT_TURN_START_REJECTED,
    AGENT_EVENT_TURN_STARTED,
    AGENT_EVENT_TURN_STEER_ACCEPTED,
    AGENT_EVENT_TURN_STEER_REJECTED,
    AGENT_EVENT_TURN_STEERED,
    AGENT_MESSAGE_THREAD_CLOSE,
    AGENT_MESSAGE_CAPABILITIES_REQUEST,
    AGENT_MESSAGE_CAPABILITIES_RESPONSE,
    AGENT_MESSAGE_MODEL_CHANGE,
    AGENT_MESSAGE_MODELS_REQUEST,
    AGENT_MESSAGE_MODELS_RESPONSE,
    AGENT_MESSAGE_THREAD_OPEN,
    AGENT_MESSAGE_TURN_INTERRUPT,
    AGENT_MESSAGE_TURN_START,
    AGENT_MESSAGE_TURN_STEER,
    AgentMessage,
    AgentProviderInfo,
    AgentTextContent,
    AgentThreadMessage,
    ChangeModel,
    CapabilitiesRequest,
    ModelsRequest,
    OpenThread,
    TurnEnded,
    TurnInterrupt,
    TurnStartRejected,
    TurnStart,
    TurnSteer,
    TurnSteerRejected,
)
from meshagent.agents.process import AgentSupervisor, Message
from meshagent.agents.thread_status_publisher import AgentMessageThreadStatusPublisher

from .process import CodexAgentProcess
from .vendor.openai_codex.async_client import AsyncAppServerClient
from .vendor.openai_codex.client import AppServerConfig
from .vendor.openai_codex.generated.v2_all import (
    ModelListResponse,
    TurnCompletedNotification,
    TurnStatus,
)


def _codex_bin_from_path() -> str | None:
    return shutil.which("codex")


def _should_run_codex_e2e_tests() -> bool:
    return (
        os.getenv("RUN_CODEX_E2E_TESTS") == "1" and _codex_bin_from_path() is not None
    )


pytestmark = pytest.mark.skipif(
    not _should_run_codex_e2e_tests(),
    reason="set RUN_CODEX_E2E_TESTS=1 and install codex on PATH to run live Codex tests",
)


class _CollectingSupervisor(AgentSupervisor):
    def __init__(self) -> None:
        super().__init__()
        self.messages: list[AgentMessage] = []
        self._messages: asyncio.Queue[AgentMessage] = asyncio.Queue()

    def send(self, message: Message) -> None:
        self.messages.append(message.data)
        self._messages.put_nowait(message.data)

    async def wait_for_type(
        self,
        *message_types: str,
        timeout: float = 120,
    ) -> AgentMessage:
        expected = set(message_types)
        deadline = asyncio.get_running_loop().time() + timeout
        while True:
            remaining = deadline - asyncio.get_running_loop().time()
            if remaining <= 0:
                seen = ", ".join(message.type for message in self.messages)
                raise AssertionError(
                    f"timed out waiting for {sorted(expected)}; seen: {seen}"
                )
            message = await asyncio.wait_for(self._messages.get(), timeout=remaining)
            if message.type in expected:
                return message


class _LiveCodexHarness:
    def __init__(
        self,
        *,
        client: AsyncAppServerClient,
        process: CodexAgentProcess,
        supervisor: _CollectingSupervisor,
        model: str,
        thread_id: str,
        status_messages: list[AgentThreadMessage],
    ) -> None:
        self.client = client
        self.process = process
        self.supervisor = supervisor
        self.model = model
        self.thread_id = thread_id
        self.status_messages = status_messages

    def send(self, data: AgentMessage) -> None:
        self.process.send(Message(data=data))

    async def wait_for_type(
        self,
        *message_types: str,
        timeout: float = 120,
    ) -> AgentMessage:
        return await self.supervisor.wait_for_type(*message_types, timeout=timeout)

    async def wait_for_turn_end(self, *, timeout: float = 180) -> TurnEnded:
        message = await self.wait_for_type(AGENT_EVENT_TURN_ENDED, timeout=timeout)
        return TurnEnded.model_validate(message.model_dump(mode="python"))


def _config_overrides_from_env() -> tuple[str, ...]:
    raw = os.getenv("CODEX_E2E_CONFIG", "").strip()
    return tuple(shlex.split(raw)) if raw != "" else ()


def _live_test_model(response: ModelListResponse) -> str:
    configured = os.getenv("CODEX_E2E_MODEL", "").strip()
    if configured != "":
        return configured
    for model in response.data:
        if model.is_default:
            return model.model.strip() if model.model.strip() != "" else model.id
    if len(response.data) > 0:
        model = response.data[0]
        return model.model.strip() if model.model.strip() != "" else model.id
    return "gpt-5.5"


async def _materialize_live_thread(harness: "_LiveCodexHarness") -> None:
    started = await harness.client.turn_start(
        harness.thread_id,
        [{"type": "text", "text": "Reply with exactly: seeded"}],
        {"model": harness.model},
    )
    try:
        deadline = asyncio.get_running_loop().time() + 180
        while True:
            remaining = deadline - asyncio.get_running_loop().time()
            if remaining <= 0:
                raise AssertionError("timed out waiting for seed turn completion")
            notification = await asyncio.wait_for(
                harness.client.next_turn_notification(started.turn.id),
                timeout=remaining,
            )
            if isinstance(notification.payload, TurnCompletedNotification):
                if notification.payload.turn.status != TurnStatus.completed:
                    raise AssertionError(
                        "seed turn failed with status "
                        f"{notification.payload.turn.status.value}"
                    )
                return
    finally:
        harness.client.unregister_turn_notifications(started.turn.id)


async def _materialize_client_thread(
    *,
    client: AsyncAppServerClient,
    thread_id: str,
    model: str,
) -> None:
    started = await client.turn_start(
        thread_id,
        [{"type": "text", "text": "Reply with exactly: seeded"}],
        {"model": model},
    )
    try:
        deadline = asyncio.get_running_loop().time() + 180
        while True:
            remaining = deadline - asyncio.get_running_loop().time()
            if remaining <= 0:
                raise AssertionError("timed out waiting for seed turn completion")
            notification = await asyncio.wait_for(
                client.next_turn_notification(started.turn.id),
                timeout=remaining,
            )
            if isinstance(notification.payload, TurnCompletedNotification):
                if notification.payload.turn.status != TurnStatus.completed:
                    raise AssertionError(
                        "seed turn failed with status "
                        f"{notification.payload.turn.status.value}"
                    )
                return
    finally:
        client.unregister_turn_notifications(started.turn.id)


@asynccontextmanager
async def _live_codex_harness(tmp_path: Path) -> AsyncIterator[_LiveCodexHarness]:
    codex_bin = _codex_bin_from_path()
    if codex_bin is None:
        pytest.skip("codex binary is not on PATH")

    client = AsyncAppServerClient(
        AppServerConfig(
            codex_bin=codex_bin,
            cwd=str(tmp_path),
            config_overrides=_config_overrides_from_env(),
            client_name="meshagent_codex_e2e",
            client_title="MeshAgent Codex E2E",
        )
    )
    process: CodexAgentProcess | None = None
    supervisor: _CollectingSupervisor | None = None
    try:
        await client.start()
        await client.initialize()
        model = _live_test_model(await client.model_list(include_hidden=False))
        thread = await client.thread_start(
            {
                "model": model,
                "ephemeral": False,
                "cwd": str(tmp_path),
                "developerInstructions": (
                    "You are running inside a MeshAgent live e2e test. "
                    "Keep responses short and follow the user's requested exact text."
                ),
            }
        )
        thread_id = thread.thread.id
        supervisor = _CollectingSupervisor()
        status_messages: list[AgentThreadMessage] = []
        process = CodexAgentProcess(
            thread_id=thread_id,
            participant=None,
            client=client,
            provider_name="codex",
            default_model=model,
            provider_info_builder=lambda current_model: AgentProviderInfo(
                name="codex",
                friendly_name="Codex",
                default_model=current_model or model,
                models=[],
            ),
            thread_status_publisher=AgentMessageThreadStatusPublisher(
                thread_id=thread_id,
                publish=status_messages.append,
            ),
        )
        await process.start(supervisor)
        yield _LiveCodexHarness(
            client=client,
            process=process,
            supervisor=supervisor,
            model=model,
            thread_id=thread_id,
            status_messages=status_messages,
        )
    finally:
        if process is not None and supervisor is not None:
            with contextlib.suppress(Exception):
                await process.stop(supervisor)
        await client.close()


@pytest.mark.asyncio
async def test_codex_live_e2e_thread_loading_models_and_capabilities(
    tmp_path: Path,
) -> None:
    async with _live_codex_harness(tmp_path) as harness:
        harness.send(
            OpenThread(type=AGENT_MESSAGE_THREAD_OPEN, thread_id=harness.thread_id)
        )
        assert (
            await harness.wait_for_type(AGENT_EVENT_THREAD_LOADED)
        ).thread_id == harness.thread_id
        model_changed = await harness.wait_for_type(AGENT_EVENT_MODEL_CHANGED)
        assert model_changed.thread_id == harness.thread_id
        assert model_changed.model == harness.model

        harness.send(ModelsRequest(type=AGENT_MESSAGE_MODELS_REQUEST))
        models = await harness.wait_for_type(AGENT_MESSAGE_MODELS_RESPONSE)
        assert len(models.providers) == 1
        assert models.providers[0].name == "codex"

        harness.send(
            CapabilitiesRequest(
                type=AGENT_MESSAGE_CAPABILITIES_REQUEST,
                thread_id=harness.thread_id,
            )
        )
        capabilities = await harness.wait_for_type(AGENT_MESSAGE_CAPABILITIES_RESPONSE)
        assert capabilities.thread_id == harness.thread_id
        assert capabilities.version == "codex"

        harness.send(
            ChangeModel(
                type=AGENT_MESSAGE_MODEL_CHANGE,
                thread_id=harness.thread_id,
                model=harness.model,
            )
        )
        changed = await harness.wait_for_type(AGENT_EVENT_MODEL_CHANGED)
        assert changed.model == harness.model


@pytest.mark.asyncio
async def test_codex_live_e2e_turn_streams_status_and_text(tmp_path: Path) -> None:
    async with _live_codex_harness(tmp_path) as harness:
        harness.send(
            TurnStart(
                type=AGENT_MESSAGE_TURN_START,
                thread_id=harness.thread_id,
                content=[
                    AgentTextContent(
                        type="text",
                        text="Reply with exactly this text and no punctuation: meshagent e2e ok",
                    )
                ],
                model=harness.model,
            )
        )
        accepted = await harness.wait_for_type(AGENT_EVENT_TURN_START_ACCEPTED)
        assert accepted.thread_id == harness.thread_id
        assert accepted.turn_id is not None
        started = await harness.wait_for_type(AGENT_EVENT_TURN_STARTED)
        assert started.turn_id == accepted.turn_id
        ended = await harness.wait_for_turn_end()
        assert ended.turn_id == accepted.turn_id
        assert ended.error is None

        text = "".join(
            message.text
            for message in harness.supervisor.messages
            if message.type == AGENT_EVENT_TEXT_CONTENT_DELTA
        )
        assert "meshagent" in text.casefold()
        assert any(
            message.type == AGENT_EVENT_THREAD_STATUS
            and message.turn_id == accepted.turn_id
            for message in harness.status_messages
        )
        assert harness.status_messages[-1].status is None


@pytest.mark.asyncio
async def test_codex_live_e2e_loaded_thread_accepts_followup_turn(
    tmp_path: Path,
) -> None:
    async with _live_codex_harness(tmp_path) as harness:
        await _materialize_live_thread(harness)
        harness.send(
            OpenThread(
                type=AGENT_MESSAGE_THREAD_OPEN,
                thread_id=harness.thread_id,
                load=True,
            )
        )
        loaded = await harness.wait_for_type(AGENT_EVENT_THREAD_LOADED)
        assert loaded.thread_id == harness.thread_id

        harness.send(
            TurnStart(
                type=AGENT_MESSAGE_TURN_START,
                thread_id=harness.thread_id,
                content=[
                    AgentTextContent(
                        type="text",
                        text=(
                            "Reply with exactly this text and no punctuation: "
                            "meshagent loaded followup ok"
                        ),
                    )
                ],
                model=harness.model,
            )
        )
        accepted = await harness.wait_for_type(AGENT_EVENT_TURN_START_ACCEPTED)
        ended = await harness.wait_for_turn_end()
        assert ended.turn_id == accepted.turn_id
        assert ended.error is None

        text = "".join(
            message.text
            for message in harness.supervisor.messages
            if message.type == AGENT_EVENT_TEXT_CONTENT_DELTA
        )
        assert "meshagent" in text.casefold()


@pytest.mark.asyncio
async def test_codex_live_e2e_failed_terminal_turn_reports_error(
    tmp_path: Path,
) -> None:
    async with _live_codex_harness(tmp_path) as harness:
        harness.send(
            TurnStart(
                type=AGENT_MESSAGE_TURN_START,
                thread_id=harness.thread_id,
                content=[AgentTextContent(type="text", text="hello")],
                model="definitely-not-a-real-codex-model",
            )
        )
        accepted = await harness.wait_for_type(AGENT_EVENT_TURN_START_ACCEPTED)
        ended = await harness.wait_for_turn_end()
        assert ended.turn_id == accepted.turn_id
        assert ended.error is not None
        assert ended.error.code == "codex_turn_failed"
        assert ended.error.message != ""
        assert harness.status_messages[-1].status is None


@pytest.mark.asyncio
async def test_codex_live_e2e_turn_steering(tmp_path: Path) -> None:
    async with _live_codex_harness(tmp_path) as harness:
        harness.send(
            TurnStart(
                type=AGENT_MESSAGE_TURN_START,
                thread_id=harness.thread_id,
                content=[
                    AgentTextContent(
                        type="text",
                        text=(
                            "Wait for one more user message before giving the final "
                            "answer. Do not finish until the user says CONTINUE."
                        ),
                    )
                ],
                model=harness.model,
            )
        )
        accepted = await harness.wait_for_type(AGENT_EVENT_TURN_START_ACCEPTED)
        harness.send(
            TurnSteer(
                type=AGENT_MESSAGE_TURN_STEER,
                thread_id=harness.thread_id,
                turn_id=accepted.turn_id,
                content=[
                    AgentTextContent(
                        type="text",
                        text="CONTINUE. Reply with exactly: meshagent steer ok",
                    )
                ],
            )
        )
        steer_accepted = await harness.wait_for_type(
            AGENT_EVENT_TURN_STEER_ACCEPTED,
            timeout=60,
        )
        assert steer_accepted.turn_id == accepted.turn_id
        steered = await harness.wait_for_type(AGENT_EVENT_TURN_STEERED, timeout=60)
        assert steered.turn_id == accepted.turn_id
        ended = await harness.wait_for_turn_end()
        assert ended.turn_id == accepted.turn_id


@pytest.mark.asyncio
async def test_codex_live_e2e_turn_interruption(tmp_path: Path) -> None:
    async with _live_codex_harness(tmp_path) as harness:
        harness.send(
            TurnStart(
                type=AGENT_MESSAGE_TURN_START,
                thread_id=harness.thread_id,
                content=[
                    AgentTextContent(
                        type="text",
                        text=(
                            "Think silently for a long time before answering. "
                            "Do not provide a final answer yet."
                        ),
                    )
                ],
                model=harness.model,
            )
        )
        accepted = await harness.wait_for_type(AGENT_EVENT_TURN_START_ACCEPTED)
        harness.send(
            TurnInterrupt(
                type=AGENT_MESSAGE_TURN_INTERRUPT,
                thread_id=harness.thread_id,
                turn_id=accepted.turn_id,
            )
        )
        interrupt_accepted = await harness.wait_for_type(
            AGENT_EVENT_TURN_INTERRUPT_ACCEPTED,
            timeout=60,
        )
        assert interrupt_accepted.turn_id == accepted.turn_id
        ended = await harness.wait_for_turn_end()
        assert ended.turn_id == accepted.turn_id
        assert any(
            message.type in {AGENT_EVENT_TURN_INTERRUPTED, AGENT_EVENT_TURN_ENDED}
            and message.turn_id == accepted.turn_id
            for message in harness.supervisor.messages
        )


@pytest.mark.asyncio
async def test_codex_live_e2e_run_tui_loaded_thread_followups(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from meshagent.agents.chat_client import ChatThreadSession
    from meshagent.cli import ask as ask_module
    from meshagent.cli import process as process_cli

    codex_bin = _codex_bin_from_path()
    if codex_bin is None:
        pytest.skip("codex binary is not on PATH")

    working_dir = str(tmp_path)
    seed_client = AsyncAppServerClient(
        AppServerConfig(
            codex_bin=codex_bin,
            cwd=working_dir,
            config_overrides=_config_overrides_from_env(),
            client_name="meshagent_codex_e2e_seed",
            client_title="MeshAgent Codex E2E Seed",
        )
    )
    await seed_client.start()
    try:
        await seed_client.initialize()
        model = _live_test_model(await seed_client.model_list(include_hidden=False))
        thread = await seed_client.thread_start(
            {
                "model": model,
                "ephemeral": False,
                "cwd": working_dir,
                "developerInstructions": (
                    "You are running inside a MeshAgent live e2e test. "
                    "Keep responses short and follow the user's requested exact text."
                ),
            }
        )
        thread_id = thread.thread.id
        await _materialize_client_thread(
            client=seed_client,
            thread_id=thread_id,
            model=model,
        )
    finally:
        await seed_client.close()

    def trace_message(message: Message, *, prefix: str) -> str:
        payload = message.data
        details = [prefix, payload.type]
        for field_name in ("message_id", "source_message_id", "thread_id", "turn_id"):
            value = payload.model_dump(mode="python").get(field_name)
            if isinstance(value, str) and value.strip() != "":
                details.append(f"{field_name}={value}")
        if isinstance(payload, (TurnStartRejected, TurnSteerRejected)):
            details.append(f"error_code={payload.error.code}")
            details.append(f"error_message={payload.error.message}")
        return " ".join(details)

    async def wait_for_turn_end(
        session: ChatThreadSession,
        *,
        seen: list[str],
        case_name: str,
    ) -> None:
        session_events: list[str] = []
        deadline = asyncio.get_running_loop().time() + 180
        while True:
            remaining = deadline - asyncio.get_running_loop().time()
            if remaining <= 0:
                pending = [
                    f"{item.message_type}:{item.message_id}:{item.label}"
                    for item in session.pending_inputs
                ]
                message_types = [message.type for message in session.messages]
                raise AssertionError(
                    f"{case_name} timed out waiting for turn end; "
                    f"session_events={session_events}; pending={pending}; "
                    f"message_types={message_types}; status={session.thread_status_text}; "
                    f"seen={seen}"
                )
            event = await asyncio.wait_for(session.receive(), timeout=remaining)
            event_type = event.get("type")
            session_events.append(str(event_type))
            if event_type == AGENT_EVENT_TURN_START_REJECTED:
                rejected = TurnStartRejected.model_validate(event)
                raise AssertionError(
                    f"{case_name} turn start rejected: "
                    f"{rejected.error.code}: {rejected.error.message}; "
                    f"session_events={session_events}; seen={seen}"
                )
            if event_type == AGENT_EVENT_TURN_STEER_REJECTED:
                rejected = TurnSteerRejected.model_validate(event)
                raise AssertionError(
                    f"{case_name} turn steer rejected: "
                    f"{rejected.error.code}: {rejected.error.message}; "
                    f"session_events={session_events}; seen={seen}"
                )
            if event_type == AGENT_EVENT_TURN_ENDED:
                ended = TurnEnded.model_validate(event)
                if ended.error is not None:
                    raise AssertionError(
                        f"{case_name} turn ended with error: "
                        f"{ended.error.code}: {ended.error.message}; "
                        f"session_events={session_events}; seen={seen}"
                    )
                return

    async def run_loaded_startup_prompt(**kwargs) -> None:
        session = kwargs["session_provider"]()
        assert isinstance(session, ChatThreadSession)
        assert session.thread_path == thread_id
        await ask_module._send_chat_thread_prompt(
            session=session,
            prompt=(
                "Reply with exactly this text and no punctuation: "
                "meshagent loaded startup ok"
            ),
        )
        await wait_for_turn_end(
            session,
            seen=kwargs["trace_seen"],
            case_name="startup loaded thread follow-up",
        )

    async def run_sidebar_loaded_prompt(**kwargs) -> None:
        await asyncio.sleep(0)
        handled = False
        deadline = asyncio.get_running_loop().time() + 30
        while not handled:
            handler_result = kwargs["side_panel_mouse_handler"](0, 4)
            if inspect.isawaitable(handler_result):
                handler_result = await handler_result
            handled = handler_result is True
            if handled:
                break
            if asyncio.get_running_loop().time() >= deadline:
                raise AssertionError("thread sidebar did not open selected thread")
            await asyncio.sleep(0.05)

        session = kwargs["session_provider"]()
        assert isinstance(session, ChatThreadSession)
        assert session.thread_path == thread_id
        await ask_module._send_chat_thread_prompt(
            session=session,
            prompt=(
                "Reply with exactly this text and no punctuation: "
                "meshagent loaded sidebar ok"
            ),
        )
        await wait_for_turn_end(
            session,
            seen=kwargs["trace_seen"],
            case_name="sidebar loaded thread follow-up",
        )

    async def run_case(
        *,
        thread_path: str | None,
        ask_runner,
    ) -> list[str]:
        class _LocalParticipant:
            id = "codex"
            attributes = {"name": "codex"}

            def get_attribute(self, name: str):
                return self.attributes.get(name)

            async def set_attribute(self, name: str, value) -> None:
                self.attributes[name] = value

        class _Room:
            local_participant = _LocalParticipant()

        agent_cls = process_cli.build_process_agent(
            model=f"codex/{model}",
            rule=[],
            toolkit=[],
            schema=[],
            require_table_read=[],
            require_table_write=[],
            channels=[],
            thread_storage="codex",
            working_dir=working_dir,
        )
        agent = agent_cls()
        await agent.start(room=_Room())
        supervisor = agent._supervisor
        event_queue = supervisor.subscribe_local_events()
        seen: list[str] = []
        original_send = supervisor.send

        def record_send(message: Message) -> None:
            seen.append(trace_message(message, prefix="send"))
            original_send(message)

        supervisor.send = record_send

        async def traced_ask_runner(**kwargs) -> None:
            await ask_runner(**kwargs, trace_seen=seen)

        monkeypatch.setattr(ask_module, "_run_ask_tui", traced_ask_runner)
        try:
            trace_task = asyncio.create_task(trace_local_events(event_queue, seen))
            try:
                await asyncio.wait_for(
                    process_cli._run_process_run_tui(
                        bot=agent,
                        room=None,
                        model=f"codex/{model}",
                        thread_path=thread_path,
                        thread_storage="codex",
                        agent_name="codex",
                        thread_dir=None,
                        threading_mode="none",
                        message=None,
                        working_dir=working_dir,
                    ),
                    timeout=240,
                )
            finally:
                trace_task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await trace_task
        finally:
            supervisor.unsubscribe_local_events(event_queue)
            await agent.stop()
        return seen

    async def trace_local_events(
        queue: asyncio.Queue[Message],
        seen: list[str],
    ) -> None:
        while True:
            message = await queue.get()
            seen.append(trace_message(message, prefix="local"))

    startup_seen = await run_case(
        thread_path=thread_id,
        ask_runner=run_loaded_startup_prompt,
    )
    sidebar_seen = await run_case(
        thread_path=None,
        ask_runner=run_sidebar_loaded_prompt,
    )

    for seen in (startup_seen, sidebar_seen):
        seen_types = [entry.split()[1] for entry in seen]
        assert AGENT_MESSAGE_THREAD_OPEN in seen_types, seen
        assert AGENT_EVENT_THREAD_LOADED in seen_types, seen
        assert AGENT_MESSAGE_TURN_START in seen_types, seen
        assert AGENT_EVENT_TURN_START_ACCEPTED in seen_types, seen
        assert AGENT_EVENT_TURN_ENDED in seen_types, seen
        assert AGENT_MESSAGE_THREAD_CLOSE in seen_types, seen
