from __future__ import annotations

import asyncio
import contextlib
import logging
from dataclasses import dataclass
from typing import Any, Callable, Protocol

from pydantic import BaseModel

from meshagent.agents.messages import (
    AGENT_EVENT_MODEL_CHANGED,
    AGENT_EVENT_TEXT_CONTENT_DELTA,
    AGENT_EVENT_TEXT_CONTENT_ENDED,
    AGENT_EVENT_TEXT_CONTENT_STARTED,
    AGENT_EVENT_THREAD_EVENT,
    AGENT_EVENT_THREAD_LOADED,
    AGENT_EVENT_TOOL_CALL_ENDED,
    AGENT_EVENT_TOOL_CALL_STARTED,
    AGENT_EVENT_USAGE_UPDATED,
    AGENT_EVENT_TURN_ENDED,
    AGENT_EVENT_TURN_INTERRUPTED,
    AGENT_EVENT_TURN_INTERRUPT_ACCEPTED,
    AGENT_EVENT_TURN_STARTED,
    AGENT_EVENT_TURN_START_ACCEPTED,
    AGENT_EVENT_TURN_START_REJECTED,
    AGENT_EVENT_TURN_STEERED,
    AGENT_EVENT_TURN_STEER_ACCEPTED,
    AGENT_EVENT_TURN_STEER_REJECTED,
    AGENT_MESSAGE_CAPABILITIES_REQUEST,
    AGENT_MESSAGE_CAPABILITIES_RESPONSE,
    AGENT_MESSAGE_MODEL_CHANGE,
    AGENT_MESSAGE_MODELS_REQUEST,
    AGENT_MESSAGE_MODELS_RESPONSE,
    AGENT_MESSAGE_THREAD_CLOSE,
    AGENT_MESSAGE_THREAD_OPEN,
    AGENT_MESSAGE_TURN_INTERRUPT,
    AGENT_MESSAGE_TURN_START,
    AGENT_MESSAGE_TURN_STEER,
    AgentError,
    AgentMessage,
    AgentModelChanged,
    AgentProviderInfo,
    AgentTextContentDelta,
    AgentTextContentEnded,
    AgentTextContentStarted,
    AgentThreadEvent,
    AgentThreadMessage,
    AgentToolCallEnded,
    AgentToolCallStarted,
    AgentContextWindowUsage,
    AgentUsageUpdated,
    CapabilitiesRequest,
    CapabilitiesResponse,
    ChangeModel,
    ModelsRequest,
    ModelsResponse,
    OpenThread,
    ThreadLoaded,
    TurnEnded,
    TurnInterrupt,
    TurnInterrupted,
    TurnInterruptAccepted,
    TurnStart,
    TurnStartAccepted,
    TurnStarted,
    TurnStartRejected,
    TurnSteer,
    TurnSteerAccepted,
    TurnSteered,
    TurnSteerRejected,
)
from meshagent.agents.process import AgentProcess, Message
from meshagent.agents.thread_status_publisher import ThreadStatusPublisher
from meshagent.agents.thread_storage import ThreadStorage
from meshagent.api import Participant
from meshagent.api.agent_content import AgentFileContent, AgentTextContent
from meshagent.openai.tools import OpenAIResponsesAdapter

from .vendor.openai_codex.generated.v2_all import (
    AgentMessageDeltaNotification,
    AgentMessageThreadItem,
    ErrorNotification,
    ImageUserInput,
    ItemCompletedNotification,
    ItemStartedNotification,
    LocalImageUserInput,
    ReasoningSummaryTextDeltaNotification,
    ReasoningTextDeltaNotification,
    TextUserInput,
    ThreadReadResponse,
    ThreadResumeResponse,
    ThreadStartResponse,
    ThreadTokenUsageUpdatedNotification,
    TokenUsageBreakdown,
    TurnCompletedNotification,
    TurnDiffUpdatedNotification,
    TurnStartedNotification,
    TurnStartResponse,
    TurnStatus,
    UserMessageThreadItem,
)
from .vendor.openai_codex.models import Notification, UnknownNotification

logger = logging.getLogger("meshagent.codex.process")


class CodexTurnClient(Protocol):
    async def thread_start(
        self,
        params: dict[str, Any] | None = None,
    ) -> ThreadStartResponse: ...

    async def thread_inject_items(
        self,
        thread_id: str,
        items: list,
    ) -> Any: ...

    async def turn_start(
        self,
        thread_id: str,
        input_items: list[dict[str, Any]] | dict[str, Any] | str,
        params: dict[str, Any] | None = None,
    ) -> TurnStartResponse: ...

    async def turn_steer(
        self,
        thread_id: str,
        expected_turn_id: str,
        input_items: list[dict[str, Any]] | dict[str, Any] | str,
    ): ...

    async def turn_interrupt(self, thread_id: str, turn_id: str): ...

    async def thread_read(
        self,
        thread_id: str,
        include_turns: bool = False,
    ) -> ThreadReadResponse: ...

    async def thread_resume(
        self,
        thread_id: str,
        params: dict[str, Any] | None = None,
    ) -> ThreadResumeResponse: ...

    async def next_turn_notification(self, turn_id: str) -> Notification: ...

    def unregister_turn_notifications(self, turn_id: str) -> None: ...


ProviderInfoBuilder = Callable[[str | None], AgentProviderInfo]


@dataclass(slots=True)
class _ActiveTurn:
    codex_turn_id: str
    source_message_id: str
    sender: Participant | None
    text_item_started: set[str]
    text_item_delta_seen: set[str]
    text_item_ended: set[str]
    diff_tool_item_ids: set[str]


@dataclass(frozen=True, slots=True)
class _CodexDiffToolNotification:
    thread_id: str
    turn_id: str
    item_id: str
    tool: str
    diff: str
    completed: bool


class CodexAgentProcess(AgentProcess):
    """Per-thread process for one Codex AsyncThread/turn stream."""

    def __init__(
        self,
        *,
        thread_id: str,
        participant: Participant,
        client: CodexTurnClient,
        provider_name: str,
        default_model: str,
        provider_info_builder: ProviderInfoBuilder,
        backend_name: str = "codex",
        thread_status_publisher: ThreadStatusPublisher | None = None,
        working_dir: str | None = None,
        thread_storage: ThreadStorage | None = None,
        ephemeral_codex_thread: bool = False,
    ) -> None:
        super().__init__(
            thread_id=thread_id,
            thread_storage=thread_storage,
            backend=backend_name,
        )
        self._participant = participant
        self._client = client
        self._provider_name = provider_name
        self._backend_name = backend_name
        self._current_model = default_model
        self._provider_info_builder = provider_info_builder
        self._thread_status_publisher = thread_status_publisher
        self._working_dir = working_dir
        self._active_turn: _ActiveTurn | None = None
        self._active_turn_task: asyncio.Task[None] | None = None
        self._steer_tasks: set[asyncio.Task[None]] = set()
        self._ephemeral_codex_thread = ephemeral_codex_thread
        self._codex_thread_id = None if ephemeral_codex_thread else thread_id
        self._thread_storage_injected = False
        self._handlers = {
            AGENT_MESSAGE_THREAD_OPEN: self.on_thread_open,
            AGENT_MESSAGE_THREAD_CLOSE: self.on_thread_close,
            AGENT_MESSAGE_TURN_START: self.on_turn_start,
            AGENT_MESSAGE_TURN_STEER: self.on_turn_steer,
            AGENT_MESSAGE_TURN_INTERRUPT: self.on_turn_interrupt,
            AGENT_MESSAGE_MODELS_REQUEST: self.on_models_request,
            AGENT_MESSAGE_CAPABILITIES_REQUEST: self.on_capabilities_request,
            AGENT_MESSAGE_MODEL_CHANGE: self.on_model_change,
        }

    def emit(self, *, sender: Participant | None, payload: AgentMessage) -> None:
        thread_storage = self.thread_storage
        if (
            thread_storage is not None
            and isinstance(payload, AgentThreadMessage)
            and payload.thread_id == self.thread_id
        ):
            thread_storage.push_message(message=payload, sender=sender)
        super().emit(sender=sender, payload=payload)

    @property
    def turn_id(self) -> str | None:
        active_turn = self._active_turn
        if active_turn is None:
            return None
        return active_turn.codex_turn_id

    def handles(self, message: Message) -> bool:
        return message.data.type in self._handlers

    async def on_message(self, message: Message) -> None:
        handler = self._handlers.get(message.data.type)
        if handler is None:
            return
        await handler(message)

    async def on_stop(self) -> None:
        active_turn = self._active_turn
        if active_turn is not None:
            with contextlib.suppress(Exception):
                await self._client.turn_interrupt(
                    self._codex_thread_id_or_raise(), active_turn.codex_turn_id
                )
        task = self._active_turn_task
        if task is not None:
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task
        steer_tasks = list(self._steer_tasks)
        for steer_task in steer_tasks:
            steer_task.cancel()
        for steer_task in steer_tasks:
            with contextlib.suppress(asyncio.CancelledError):
                await steer_task

    async def on_thread_open(self, message: Message) -> None:
        request = OpenThread.model_validate(message.data.model_dump(mode="python"))
        if request.load is True:
            if self._ephemeral_codex_thread:
                await self._ensure_codex_thread()
                stored_messages = await self._messages_from_thread_storage(
                    since_turn=request.since_turn
                )
            else:
                response = await self._client.thread_resume(
                    self._codex_thread_id_or_raise(),
                    params=self._thread_resume_params(),
                )
                if response.model.strip() != "":
                    self._current_model = response.model.strip()
                stored_messages = self._messages_from_thread_read(
                    response=response,
                    since_turn=request.since_turn,
                )
            for stored_message in stored_messages:
                self.emit(sender=message.sender, payload=stored_message)
        self.emit(
            sender=message.sender,
            payload=ThreadLoaded(
                type=AGENT_EVENT_THREAD_LOADED,
                thread_id=request.thread_id,
                source_message_id=request.message_id,
                since_turn=request.since_turn,
            ),
        )
        self.emit(
            sender=message.sender,
            payload=self._model_changed(
                thread_id=request.thread_id,
                source_message_id=request.message_id,
            ),
        )

    def _messages_from_thread_read(
        self,
        *,
        response: ThreadReadResponse | ThreadResumeResponse,
        since_turn: str | None,
    ) -> list[AgentMessage]:
        messages: list[AgentMessage] = []
        thread_id = self._thread_id_or_raise()
        normalized_since_turn = _normalized_non_empty_string(since_turn)
        include = normalized_since_turn is None
        for turn in response.thread.turns:
            if turn.id == normalized_since_turn:
                include = True
            turn_messages: list[AgentMessage] = []
            for item in turn.items:
                thread_item = item.root
                if isinstance(thread_item, UserMessageThreadItem):
                    content = _agent_input_content_from_codex_user_input(
                        thread_item.content
                    )
                    if len(content) == 0:
                        continue
                    turn_messages.append(
                        TurnStartAccepted(
                            type=AGENT_EVENT_TURN_START_ACCEPTED,
                            message_id=thread_item.id,
                            thread_id=thread_id,
                            turn_id=turn.id,
                            source_message_id=thread_item.id,
                            content=content,
                        )
                    )
                    continue
                if isinstance(thread_item, AgentMessageThreadItem):
                    if thread_item.text.strip() == "":
                        continue
                    turn_messages.append(
                        AgentTextContentDelta(
                            type=AGENT_EVENT_TEXT_CONTENT_DELTA,
                            thread_id=thread_id,
                            turn_id=turn.id,
                            item_id=thread_item.id,
                            provider=self._provider_name,
                            model=self._current_model,
                            text=thread_item.text,
                        )
                    )
            if any(
                message.message_id == normalized_since_turn for message in turn_messages
            ):
                include = True
            if include:
                messages.extend(turn_messages)
                if len(turn_messages) > 0:
                    messages.append(
                        TurnEnded(
                            type=AGENT_EVENT_TURN_ENDED,
                            thread_id=thread_id,
                            turn_id=turn.id,
                        )
                    )
        return messages

    async def _messages_from_thread_storage(
        self,
        *,
        since_turn: str | None,
    ) -> list[AgentMessage]:
        thread_storage = self.thread_storage
        if thread_storage is None:
            return []
        await thread_storage.wait_until_ready()
        messages = thread_storage.agent_messages()
        normalized_since_turn = _normalized_non_empty_string(since_turn)
        if normalized_since_turn is None:
            return messages
        for index, stored_message in enumerate(messages):
            if stored_message.message_id == normalized_since_turn:
                return messages[index:]
            if stored_message.turn_id == normalized_since_turn:
                return messages[index:]
        return []

    async def on_thread_close(self, message: Message) -> None:
        del message

    async def on_models_request(self, message: Message) -> None:
        request = ModelsRequest.model_validate(message.data.model_dump(mode="python"))
        self.emit(
            sender=message.sender,
            payload=ModelsResponse(
                type=AGENT_MESSAGE_MODELS_RESPONSE,
                source_message_id=request.message_id,
                providers=[self._provider_info_builder(self._current_model)],
            ),
        )

    async def on_capabilities_request(self, message: Message) -> None:
        request = CapabilitiesRequest.model_validate(
            message.data.model_dump(mode="python")
        )
        self.emit(
            sender=message.sender,
            payload=CapabilitiesResponse(
                type=AGENT_MESSAGE_CAPABILITIES_RESPONSE,
                thread_id=request.thread_id,
                source_message_id=request.message_id,
                version="codex",
                toolkits=[],
            ),
        )

    async def on_model_change(self, message: Message) -> None:
        request = ChangeModel.model_validate(message.data.model_dump(mode="python"))
        if request.model is not None and request.model.strip() != "":
            self._current_model = request.model.strip()
        self.emit(
            sender=message.sender,
            payload=self._model_changed(
                thread_id=request.thread_id,
                source_message_id=request.message_id,
            ),
        )

    async def on_turn_start(self, message: Message) -> None:
        turn_start = TurnStart.model_validate(message.data.model_dump(mode="python"))
        if self._active_turn is not None:
            self.emit(
                sender=message.sender,
                payload=TurnStartRejected(
                    type=AGENT_EVENT_TURN_START_REJECTED,
                    thread_id=turn_start.thread_id,
                    source_message_id=turn_start.message_id,
                    error=AgentError(
                        message="turn is already in progress",
                        code="turn_in_progress",
                    ),
                ),
            )
            return

        if turn_start.model is not None and turn_start.model.strip() != "":
            self._current_model = turn_start.model.strip()

        try:
            codex_thread_id = await self._ensure_codex_thread()
            response = await self._client.turn_start(
                codex_thread_id,
                _codex_input_items(turn_start.content),
                params=self._turn_params(turn_start),
            )
        except Exception as exc:
            self.emit(
                sender=message.sender,
                payload=TurnStartRejected(
                    type=AGENT_EVENT_TURN_START_REJECTED,
                    thread_id=turn_start.thread_id,
                    source_message_id=turn_start.message_id,
                    error=AgentError(message=str(exc), code="codex_turn_start_failed"),
                ),
            )
            return

        codex_turn_id = response.turn.id
        active_turn = _ActiveTurn(
            codex_turn_id=codex_turn_id,
            source_message_id=turn_start.message_id,
            sender=message.sender,
            text_item_started=set(),
            text_item_delta_seen=set(),
            text_item_ended=set(),
            diff_tool_item_ids=set(),
        )
        self._active_turn = active_turn
        await self._set_status(status="Queued", turn_id=codex_turn_id)
        self.emit(
            sender=message.sender,
            payload=TurnStartAccepted(
                type=AGENT_EVENT_TURN_START_ACCEPTED,
                thread_id=turn_start.thread_id,
                turn_id=codex_turn_id,
                source_message_id=turn_start.message_id,
                content=turn_start.content,
                sender_name=turn_start.sender_name,
            ),
        )
        self.emit(
            sender=message.sender,
            payload=TurnStarted(
                type=AGENT_EVENT_TURN_STARTED,
                thread_id=turn_start.thread_id,
                turn_id=codex_turn_id,
                source_message_id=turn_start.message_id,
            ),
        )
        self._active_turn_task = asyncio.create_task(
            self._run_turn_notifications(active_turn=active_turn)
        )

    async def on_turn_steer(self, message: Message) -> None:
        turn_steer = TurnSteer.model_validate(message.data.model_dump(mode="python"))
        active_turn = self._active_turn
        if active_turn is None or active_turn.codex_turn_id != turn_steer.turn_id:
            self.emit(
                sender=message.sender,
                payload=TurnSteerRejected(
                    type=AGENT_EVENT_TURN_STEER_REJECTED,
                    thread_id=turn_steer.thread_id,
                    turn_id=turn_steer.turn_id,
                    source_message_id=turn_steer.message_id,
                    error=AgentError(
                        message="turn is not in progress",
                        code="turn_not_in_progress",
                    ),
                ),
            )
            return
        self.emit(
            sender=message.sender,
            payload=TurnSteerAccepted(
                type=AGENT_EVENT_TURN_STEER_ACCEPTED,
                thread_id=turn_steer.thread_id,
                turn_id=active_turn.codex_turn_id,
                source_message_id=turn_steer.message_id,
                content=turn_steer.content,
                sender_name=turn_steer.sender_name,
            ),
        )
        self.emit(
            sender=message.sender,
            payload=TurnSteered(
                type=AGENT_EVENT_TURN_STEERED,
                thread_id=turn_steer.thread_id,
                turn_id=active_turn.codex_turn_id,
                source_message_id=turn_steer.message_id,
            ),
        )
        task = asyncio.create_task(
            self._run_turn_steer_handoff(
                turn_steer=turn_steer,
                codex_turn_id=active_turn.codex_turn_id,
                sender=message.sender,
            )
        )
        self._steer_tasks.add(task)
        task.add_done_callback(self._steer_tasks.discard)

    async def _run_turn_steer_handoff(
        self,
        *,
        turn_steer: TurnSteer,
        codex_turn_id: str,
        sender: Participant | None,
    ) -> None:
        try:
            await self._client.turn_steer(
                self._codex_thread_id_or_raise(),
                codex_turn_id,
                _codex_input_items(turn_steer.content),
            )
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            self.emit(
                sender=sender,
                payload=TurnSteerRejected(
                    type=AGENT_EVENT_TURN_STEER_REJECTED,
                    thread_id=turn_steer.thread_id,
                    turn_id=turn_steer.turn_id,
                    source_message_id=turn_steer.message_id,
                    error=AgentError(message=str(exc), code="codex_turn_steer_failed"),
                ),
            )

    async def on_turn_interrupt(self, message: Message) -> None:
        turn_interrupt = TurnInterrupt.model_validate(
            message.data.model_dump(mode="python")
        )
        active_turn = self._active_turn
        if active_turn is None or active_turn.codex_turn_id != turn_interrupt.turn_id:
            return
        self.emit(
            sender=message.sender,
            payload=TurnInterruptAccepted(
                type=AGENT_EVENT_TURN_INTERRUPT_ACCEPTED,
                thread_id=turn_interrupt.thread_id,
                turn_id=turn_interrupt.turn_id,
                source_message_id=turn_interrupt.message_id,
            ),
        )
        await self._client.turn_interrupt(
            self._codex_thread_id_or_raise(), turn_interrupt.turn_id
        )

    async def _run_turn_notifications(self, *, active_turn: _ActiveTurn) -> None:
        error: AgentError | None = None
        try:
            while True:
                notification = await self._client.next_turn_notification(
                    active_turn.codex_turn_id
                )
                await self._handle_notification(
                    notification=notification,
                    active_turn=active_turn,
                )
                if _is_turn_terminal(notification):
                    error = _turn_error(notification)
                    break
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.exception("Codex turn notification stream failed")
            error = AgentError(message=str(exc), code="codex_notification_failed")
        finally:
            self._client.unregister_turn_notifications(active_turn.codex_turn_id)
            await self._set_status(status=None, turn_id=None)
            self._emit_started_diff_tools_ended(active_turn=active_turn, error=error)
            if error is not None and error.code == "turn_interrupted":
                self.emit(
                    sender=active_turn.sender,
                    payload=TurnInterrupted(
                        type=AGENT_EVENT_TURN_INTERRUPTED,
                        thread_id=self._thread_id_or_raise(),
                        turn_id=active_turn.codex_turn_id,
                        source_message_id=active_turn.source_message_id,
                    ),
                )
            self.emit(
                sender=active_turn.sender,
                payload=TurnEnded(
                    type=AGENT_EVENT_TURN_ENDED,
                    thread_id=self._thread_id_or_raise(),
                    turn_id=active_turn.codex_turn_id,
                    error=None
                    if error is not None and error.code == "turn_interrupted"
                    else error,
                ),
            )
            if self._active_turn is active_turn:
                self._active_turn = None
                self._active_turn_task = None

    async def _handle_notification(
        self,
        *,
        notification: Notification,
        active_turn: _ActiveTurn,
    ) -> None:
        await self._publish_status_from_notification(notification=notification)
        payload = notification.payload
        if isinstance(payload, TurnStartedNotification):
            await self._set_status(status="Thinking", turn_id=active_turn.codex_turn_id)
            return
        if isinstance(payload, AgentMessageDeltaNotification):
            self._emit_text_delta(payload=payload, active_turn=active_turn)
            return
        if isinstance(payload, ItemStartedNotification):
            item = payload.item.root
            if item.type == "agentMessage":
                self._emit_text_started(
                    thread_id=self._thread_id_or_raise(),
                    turn_id=payload.turn_id,
                    item_id=item.id,
                    active_turn=active_turn,
                )
            return
        if isinstance(payload, ItemCompletedNotification):
            item = payload.item.root
            if item.type == "agentMessage":
                if item.id not in active_turn.text_item_started:
                    self._emit_text_started(
                        thread_id=self._thread_id_or_raise(),
                        turn_id=payload.turn_id,
                        item_id=item.id,
                        active_turn=active_turn,
                    )
                if item.text != "" and item.id not in active_turn.text_item_delta_seen:
                    self._emit_text_delta(
                        payload=AgentMessageDeltaNotification(
                            delta=item.text,
                            itemId=item.id,
                            threadId=self._thread_id_or_raise(),
                            turnId=payload.turn_id,
                        ),
                        active_turn=active_turn,
                    )
                self._emit_text_ended(
                    thread_id=self._thread_id_or_raise(),
                    turn_id=payload.turn_id,
                    item_id=item.id,
                    active_turn=active_turn,
                )
            return
        if isinstance(payload, ErrorNotification):
            return
        if isinstance(
            payload,
            (ReasoningSummaryTextDeltaNotification, ReasoningTextDeltaNotification),
        ):
            self.emit(
                sender=active_turn.sender,
                payload=AgentThreadEvent(
                    type=AGENT_EVENT_THREAD_EVENT,
                    thread_id=self._thread_id_or_raise(),
                    provider=self._provider_name,
                    model=self._current_model,
                    event={
                        "type": "codex_reasoning_delta",
                        "turn_id": payload.turn_id,
                        "item_id": payload.item_id,
                        "text": payload.delta,
                    },
                ),
            )
            return
        if isinstance(payload, ThreadTokenUsageUpdatedNotification):
            usage = _usage_update_from_codex_token_usage(payload=payload)
            self.emit(
                sender=active_turn.sender,
                payload=usage.model_copy(
                    update={"thread_id": self._thread_id_or_raise()}
                ),
            )
            return
        diff_tool = _codex_diff_tool_from_notification(notification=notification)
        if diff_tool is not None:
            self._emit_codex_diff_tool(tool=diff_tool, active_turn=active_turn)
            return

    def _emit_codex_diff_tool(
        self,
        *,
        tool: _CodexDiffToolNotification,
        active_turn: _ActiveTurn,
    ) -> None:
        arguments = {"diff": tool.diff}
        active_turn.diff_tool_item_ids.add(tool.item_id)
        self.emit(
            sender=active_turn.sender,
            payload=AgentToolCallStarted(
                type=AGENT_EVENT_TOOL_CALL_STARTED,
                thread_id=self._thread_id_or_raise(),
                turn_id=tool.turn_id,
                item_id=tool.item_id,
                namespace="codex",
                call_id=tool.item_id,
                toolkit="codex",
                tool=tool.tool,
                arguments=arguments,
            ),
        )
        if not tool.completed:
            return
        self._emit_codex_diff_tool_ended(
            thread_id=tool.thread_id,
            turn_id=tool.turn_id,
            item_id=tool.item_id,
            sender=active_turn.sender,
            error=None,
        )
        active_turn.diff_tool_item_ids.discard(tool.item_id)

    def _emit_started_diff_tools_ended(
        self,
        *,
        active_turn: _ActiveTurn,
        error: AgentError | None,
    ) -> None:
        for item_id in sorted(active_turn.diff_tool_item_ids):
            self._emit_codex_diff_tool_ended(
                thread_id=self._thread_id_or_raise(),
                turn_id=active_turn.codex_turn_id,
                item_id=item_id,
                sender=active_turn.sender,
                error=error,
            )
        active_turn.diff_tool_item_ids.clear()

    def _emit_codex_diff_tool_ended(
        self,
        *,
        thread_id: str,
        turn_id: str,
        item_id: str,
        sender: Participant | None,
        error: AgentError | None,
    ) -> None:
        self.emit(
            sender=sender,
            payload=AgentToolCallEnded(
                type=AGENT_EVENT_TOOL_CALL_ENDED,
                thread_id=self._thread_id_or_raise(),
                turn_id=turn_id,
                item_id=item_id,
                namespace="codex",
                call_id=item_id,
                toolkit="codex",
                tool=_codex_diff_tool_from_item_id(item_id=item_id),
                error=error,
            ),
        )

    def _emit_text_started(
        self,
        *,
        thread_id: str,
        turn_id: str,
        item_id: str,
        active_turn: _ActiveTurn,
    ) -> None:
        if item_id in active_turn.text_item_started:
            return
        active_turn.text_item_started.add(item_id)
        self.emit(
            sender=active_turn.sender,
            payload=AgentTextContentStarted(
                type=AGENT_EVENT_TEXT_CONTENT_STARTED,
                thread_id=thread_id,
                turn_id=turn_id,
                item_id=item_id,
                provider=self._provider_name,
                model=self._current_model,
                phase="final_answer",
            ),
        )

    def _emit_text_delta(
        self,
        *,
        payload: AgentMessageDeltaNotification,
        active_turn: _ActiveTurn,
    ) -> None:
        active_turn.text_item_delta_seen.add(payload.item_id)
        self._emit_text_started(
            thread_id=self._thread_id_or_raise(),
            turn_id=payload.turn_id,
            item_id=payload.item_id,
            active_turn=active_turn,
        )
        self.emit(
            sender=active_turn.sender,
            payload=AgentTextContentDelta(
                type=AGENT_EVENT_TEXT_CONTENT_DELTA,
                thread_id=self._thread_id_or_raise(),
                turn_id=payload.turn_id,
                item_id=payload.item_id,
                provider=self._provider_name,
                model=self._current_model,
                text=payload.delta,
                phase="final_answer",
            ),
        )

    def _emit_text_ended(
        self,
        *,
        thread_id: str,
        turn_id: str,
        item_id: str,
        active_turn: _ActiveTurn,
    ) -> None:
        if item_id in active_turn.text_item_ended:
            return
        active_turn.text_item_ended.add(item_id)
        self.emit(
            sender=active_turn.sender,
            payload=AgentTextContentEnded(
                type=AGENT_EVENT_TEXT_CONTENT_ENDED,
                thread_id=thread_id,
                turn_id=turn_id,
                item_id=item_id,
                provider=self._provider_name,
                model=self._current_model,
                phase="final_answer",
            ),
        )

    async def _publish_status_from_notification(
        self, *, notification: Notification
    ) -> None:
        status = _status_from_notification(notification)
        if status is None:
            return
        payload = notification.payload
        pending_item_id = None
        if isinstance(
            payload,
            (
                AgentMessageDeltaNotification,
                ItemCompletedNotification,
                ItemStartedNotification,
                ReasoningSummaryTextDeltaNotification,
                ReasoningTextDeltaNotification,
            ),
        ):
            if isinstance(
                payload, (ItemStartedNotification, ItemCompletedNotification)
            ):
                pending_item_id = payload.item.root.id
            else:
                pending_item_id = payload.item_id
        await self._set_status(status=status, pending_item_id=pending_item_id)

    async def _set_status(
        self,
        *,
        status: str | None,
        turn_id: str | None = None,
        pending_item_id: str | None = None,
    ) -> None:
        publisher = self._thread_status_publisher
        if publisher is None:
            return
        if turn_id is not None:
            await publisher.set_thread_turn_id(turn_id=turn_id)
        elif status is None:
            await publisher.set_thread_turn_id(turn_id=None)
        await publisher.set_thread_status(
            status=status,
            mode="steerable" if status is not None else None,
            pending_item_id=pending_item_id,
        )

    def _model_changed(
        self,
        *,
        thread_id: str,
        source_message_id: str | None,
    ) -> AgentModelChanged:
        return AgentModelChanged(
            type=AGENT_EVENT_MODEL_CHANGED,
            thread_id=thread_id,
            source_message_id=source_message_id,
            provider=self._provider_name,
            backend=self._backend_name,
            model=self._current_model,
            output_modalities=["text"],
            supports_attachments=True,
            accepts=["image"],
        )

    def _turn_params(self, turn_start: TurnStart) -> dict[str, Any]:
        params: dict[str, Any] = {"model": self._current_model}
        if self._working_dir is not None:
            params["cwd"] = self._working_dir
        return params

    def _thread_resume_params(self) -> dict[str, Any]:
        params: dict[str, Any] = {"model": self._current_model}
        if self._working_dir is not None:
            params["cwd"] = self._working_dir
        return params

    async def _ensure_codex_thread(self) -> str:
        if self._codex_thread_id is not None:
            return self._codex_thread_id
        params: dict[str, Any] = {"model": self._current_model, "ephemeral": True}
        if self._working_dir is not None:
            params["cwd"] = self._working_dir
        response = await self._client.thread_start(params)
        self._codex_thread_id = response.thread.id
        await self._inject_thread_storage()
        return self._codex_thread_id

    async def _inject_thread_storage(self) -> None:
        if self._thread_storage_injected:
            return
        self._thread_storage_injected = True
        thread_storage = self.thread_storage
        if thread_storage is None:
            return
        await thread_storage.wait_until_ready()
        items = _responses_items_from_agent_messages(thread_storage.agent_messages())
        if len(items) == 0:
            return
        await self._client.thread_inject_items(
            self._codex_thread_id_or_raise(),
            items,
        )

    def _thread_id_or_raise(self) -> str:
        if self.thread_id is None:
            raise RuntimeError("CodexAgentProcess requires a thread id")
        return self.thread_id

    def _codex_thread_id_or_raise(self) -> str:
        if self._codex_thread_id is None:
            raise RuntimeError("CodexAgentProcess requires a Codex thread id")
        return self._codex_thread_id


def _codex_input_items(content: list[Any]) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    for part in content:
        if isinstance(part, AgentTextContent):
            items.append({"type": "text", "text": part.text})
        elif isinstance(part, AgentFileContent):
            items.append({"type": "text", "text": part.url})
    if len(items) == 0:
        items.append({"type": "text", "text": ""})
    return items


def _responses_items_from_agent_messages(
    messages: list[AgentMessage],
) -> list[dict[str, Any]]:
    response_messages: list[dict[str, Any]] = []
    reader = OpenAIResponsesAdapter().make_agent_event_reader(
        emit_message=response_messages.append
    )
    for message in messages:
        reader.consume(message)
    reader.finalize()

    items: list[dict[str, Any]] = []
    for message in response_messages:
        item = _responses_item_from_adapter_message(message)
        if item is not None:
            items.append(item)
    return items


def _responses_item_from_adapter_message(
    message: dict[str, Any],
) -> dict[str, Any] | None:
    message_type = message.get("type")
    if message_type == "conversation.item.create":
        item = message.get("item")
        return dict(item) if isinstance(item, dict) else None

    role = message.get("role")
    if role in {"user", "assistant"}:
        content = message.get("content")
        if not isinstance(content, list) or len(content) == 0:
            return None
        item = dict(message)
        item.setdefault("type", "message")
        return item

    if isinstance(message_type, str) and message_type.strip() != "":
        return dict(message)
    return None


def _agent_input_content_from_codex_user_input(
    content: list[Any],
) -> list[AgentTextContent | AgentFileContent]:
    parts: list[AgentTextContent | AgentFileContent] = []
    for item in content:
        user_input = item.root
        if isinstance(user_input, TextUserInput):
            if user_input.text.strip() != "":
                parts.append(AgentTextContent(type="text", text=user_input.text))
            continue
        if isinstance(user_input, ImageUserInput):
            parts.append(AgentFileContent(type="file", url=user_input.url))
            continue
        if isinstance(user_input, LocalImageUserInput):
            parts.append(AgentFileContent(type="file", url=user_input.path))
            continue
    return parts


def _usage_update_from_codex_token_usage(
    *,
    payload: ThreadTokenUsageUpdatedNotification,
) -> AgentUsageUpdated:
    total = payload.token_usage.total
    return AgentUsageUpdated(
        type=AGENT_EVENT_USAGE_UPDATED,
        thread_id=payload.thread_id,
        turn_id=payload.turn_id,
        usage=_usage_breakdown_values(total),
        context_window=AgentContextWindowUsage(
            used_tokens=total.total_tokens,
            total_tokens=payload.token_usage.model_context_window,
        ),
    )


def _usage_breakdown_values(usage: TokenUsageBreakdown) -> dict[str, float]:
    return {
        "input_tokens": float(usage.input_tokens),
        "cached_input_tokens": float(usage.cached_input_tokens),
        "output_tokens": float(usage.output_tokens),
        "reasoning_output_tokens": float(usage.reasoning_output_tokens),
        "total_tokens": float(usage.total_tokens),
    }


def _codex_diff_tool_from_notification(
    *, notification: Notification
) -> _CodexDiffToolNotification | None:
    tool = _codex_diff_tool_from_method(method=notification.method)
    if tool is None:
        return None

    payload = notification.payload
    if isinstance(payload, TurnDiffUpdatedNotification):
        return _CodexDiffToolNotification(
            thread_id=payload.thread_id,
            turn_id=payload.turn_id,
            item_id=_codex_diff_tool_item_id(turn_id=payload.turn_id, tool=tool),
            tool=tool,
            diff=payload.diff,
            completed=_codex_diff_tool_is_terminal(tool=tool),
        )

    if not isinstance(payload, UnknownNotification):
        return None

    thread_id = _json_string(payload.params.get("threadId"))
    turn_id = _json_string(payload.params.get("turnId"))
    if thread_id is None or turn_id is None:
        return None

    diff = _json_string(payload.params.get("diff")) or ""
    return _CodexDiffToolNotification(
        thread_id=thread_id,
        turn_id=turn_id,
        item_id=_codex_diff_tool_item_id(turn_id=turn_id, tool=tool),
        tool=tool,
        diff=diff,
        completed=_codex_diff_tool_is_terminal(tool=tool),
    )


def _codex_diff_tool_from_method(*, method: str) -> str | None:
    normalized = method.strip().replace(".", "/")
    if normalized.startswith("turn/"):
        normalized = normalized[len("turn/") :]
    tool = normalized.replace("/", "_").strip("_")
    if not tool.startswith("diff"):
        return None
    return tool


def _codex_diff_tool_item_id(*, turn_id: str, tool: str) -> str:
    return f"{turn_id}:codex-{tool}"


def _codex_diff_tool_from_item_id(*, item_id: str) -> str:
    marker = ":codex-"
    if marker not in item_id:
        return "diff_updated"
    return item_id.rsplit(marker, 1)[1] or "diff_updated"


def _codex_diff_tool_is_terminal(*, tool: str) -> bool:
    return (
        tool.endswith("_completed")
        or tool.endswith("_failed")
        or tool.endswith("_cancelled")
    )


def _json_string(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    normalized = value.strip()
    return normalized if normalized != "" else None


def _normalized_non_empty_string(value: str | None) -> str | None:
    if value is None:
        return None
    normalized = value.strip()
    return normalized if normalized != "" else None


def _is_turn_terminal(notification: Notification) -> bool:
    return isinstance(notification.payload, TurnCompletedNotification)


def _turn_error(notification: Notification) -> AgentError | None:
    payload = notification.payload
    if not isinstance(payload, TurnCompletedNotification):
        return None
    if payload.turn.status == TurnStatus.completed:
        return None
    if payload.turn.status == TurnStatus.interrupted:
        return AgentError(message="turn interrupted", code="turn_interrupted")
    if payload.turn.error is not None:
        return AgentError(
            message=payload.turn.error.message,
            code="codex_turn_failed",
        )
    return AgentError(
        message=f"turn ended with status {payload.turn.status.value}",
        code="codex_turn_failed",
    )


def _status_from_notification(notification: Notification) -> str | None:
    payload = notification.payload
    if isinstance(payload, TurnStartedNotification):
        return "Thinking"
    if isinstance(payload, TurnCompletedNotification):
        if payload.turn.status == TurnStatus.completed:
            return "Wrapping up"
        if payload.turn.status == TurnStatus.interrupted:
            return "Cancelled"
        return "Failed"
    if isinstance(payload, AgentMessageDeltaNotification):
        return "Responding"
    if isinstance(
        payload, (ReasoningSummaryTextDeltaNotification, ReasoningTextDeltaNotification)
    ):
        return "Thinking"
    if isinstance(payload, ItemStartedNotification):
        return _item_status(payload.item.root.type, started=True)
    if isinstance(payload, ItemCompletedNotification):
        return _item_status(payload.item.root.type, started=False)
    if isinstance(payload, ErrorNotification):
        return "Failed"
    if isinstance(payload, ThreadTokenUsageUpdatedNotification):
        return None
    if isinstance(payload, TurnDiffUpdatedNotification):
        return "Writing file"
    if _codex_diff_tool_from_notification(notification=notification) is not None:
        return "Writing file"
    if isinstance(payload, UnknownNotification):
        return None
    if isinstance(payload, BaseModel):
        return notification.method.replace("/", " ")
    return None


def _item_status(item_type: str, *, started: bool) -> str:
    del started
    if item_type == "agentMessage":
        return "Responding"
    if item_type == "commandExecution":
        return "Running command"
    if item_type == "fileChange":
        return "Writing file"
    if item_type == "reasoning":
        return "Thinking"
    return "Working"
