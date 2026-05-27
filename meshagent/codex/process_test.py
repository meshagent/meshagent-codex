from __future__ import annotations

import asyncio
from typing import Any

import pytest

from meshagent.agents.chat_client import BaseChatClient
from meshagent.agents.messages import (
    AGENT_EVENT_TEXT_CONTENT_DELTA,
    AGENT_EVENT_THREAD_LOADED,
    AGENT_EVENT_THREAD_STATUS,
    AGENT_EVENT_TOOL_CALL_ENDED,
    AGENT_EVENT_TOOL_CALL_STARTED,
    AGENT_EVENT_USAGE_UPDATED,
    AGENT_EVENT_TURN_ENDED,
    AGENT_EVENT_TURN_STARTED,
    AGENT_EVENT_TURN_START_ACCEPTED,
    AGENT_EVENT_TURN_STEERED,
    AGENT_EVENT_TURN_STEER_ACCEPTED,
    AGENT_MESSAGE_THREAD_OPEN,
    AGENT_MESSAGE_TURN_START,
    AGENT_MESSAGE_TURN_STEER,
    AgentMessage,
    AgentProviderInfo,
    AgentTextContent,
    OpenThread,
    TurnStart,
    TurnSteer,
)
from meshagent.agents.thread_status_publisher import AgentMessageThreadStatusPublisher

from .process import CodexAgentProcess
from .vendor.openai_codex.generated.v2_all import (
    AgentMessageDeltaNotification,
    AgentMessageThreadItem,
    ErrorNotification,
    ItemCompletedNotification,
    TextUserInput,
    ThreadTokenUsage,
    ThreadTokenUsageUpdatedNotification,
    ThreadItem,
    TokenUsageBreakdown,
    Turn,
    TurnCompletedNotification,
    TurnDiffUpdatedNotification,
    TurnError,
    TurnStartedNotification,
    TurnStartResponse,
    TurnStatus,
    UserInput,
    UserMessageThreadItem,
)
from .vendor.openai_codex.models import Notification, UnknownNotification


class _FakeCodexClient:
    def __init__(self) -> None:
        self.turn_start_calls: list[tuple[str, Any, dict[str, Any] | None]] = []
        self.turn_steer_calls: list[tuple[str, str, Any]] = []
        self.notifications: asyncio.Queue[Notification] = asyncio.Queue()
        self.unregistered_turns: list[str] = []
        self.thread_read_response: Any = None
        self.thread_resume_calls: list[tuple[str, dict[str, Any] | None]] = []

    async def turn_start(
        self,
        thread_id: str,
        input_items: list[dict[str, Any]] | dict[str, Any] | str,
        params: dict[str, Any] | None = None,
    ) -> TurnStartResponse:
        self.turn_start_calls.append((thread_id, input_items, params))
        return TurnStartResponse(
            turn=Turn(id="codex-turn-1", items=[], status=TurnStatus.in_progress)
        )

    async def turn_steer(
        self,
        thread_id: str,
        expected_turn_id: str,
        input_items: list[dict[str, Any]] | dict[str, Any] | str,
    ) -> None:
        self.turn_steer_calls.append((thread_id, expected_turn_id, input_items))

    async def turn_interrupt(self, thread_id: str, turn_id: str) -> None:
        del thread_id
        del turn_id

    async def thread_read(self, thread_id: str, include_turns: bool = False):
        del thread_id
        del include_turns
        return self.thread_read_response

    async def thread_resume(
        self,
        thread_id: str,
        params: dict[str, Any] | None = None,
    ):
        self.thread_resume_calls.append((thread_id, params))
        return self.thread_read_response

    async def next_turn_notification(self, turn_id: str) -> Notification:
        del turn_id
        return await self.notifications.get()

    def unregister_turn_notifications(self, turn_id: str) -> None:
        self.unregistered_turns.append(turn_id)


def _provider_info(current_model: str | None) -> AgentProviderInfo:
    return AgentProviderInfo(
        name="codex",
        friendly_name="Codex",
        default_model=current_model or "gpt-5.5",
        models=[],
    )


class _CollectingCodexAgentProcess(CodexAgentProcess):
    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.messages: list[AgentMessage] = []

    def emit(self, *, sender, payload: AgentMessage) -> None:
        del sender
        self.messages.append(payload)


class _NoopChatClient(BaseChatClient):
    async def _start_transport(self) -> None:
        pass

    async def _stop_transport(self) -> None:
        pass

    async def _send_agent_message(self, payload: AgentMessage) -> None:
        del payload


@pytest.mark.asyncio
async def test_codex_agent_process_emits_agent_messages_and_status() -> None:
    client = _FakeCodexClient()
    emitted: list[AgentMessage] = []
    process = _CollectingCodexAgentProcess(
        thread_id="thread-1",
        participant=None,
        client=client,
        provider_name="codex",
        default_model="gpt-5.5",
        provider_info_builder=_provider_info,
        thread_status_publisher=AgentMessageThreadStatusPublisher(
            thread_id="thread-1",
            publish=emitted.append,
        ),
    )

    await process.on_turn_start(
        message=type(
            "Message",
            (),
            {
                "sender": None,
                "data": TurnStart(
                    type=AGENT_MESSAGE_TURN_START,
                    thread_id="thread-1",
                    content=[AgentTextContent(type="text", text="hello")],
                    model="gpt-5.5",
                ),
            },
        )()
    )
    await client.notifications.put(
        Notification(
            method="turn/started",
            payload=TurnStartedNotification(
                threadId="thread-1",
                turn=Turn(id="codex-turn-1", items=[], status=TurnStatus.in_progress),
            ),
        )
    )
    await client.notifications.put(
        Notification(
            method="item/agentMessage/delta",
            payload=AgentMessageDeltaNotification(
                delta="hi",
                itemId="item-1",
                threadId="thread-1",
                turnId="codex-turn-1",
            ),
        )
    )
    await client.notifications.put(
        Notification(
            method="thread/tokenUsage/updated",
            payload=ThreadTokenUsageUpdatedNotification(
                threadId="thread-1",
                turnId="codex-turn-1",
                tokenUsage=ThreadTokenUsage(
                    last=TokenUsageBreakdown(
                        cachedInputTokens=10,
                        inputTokens=100,
                        outputTokens=20,
                        reasoningOutputTokens=5,
                        totalTokens=120,
                    ),
                    total=TokenUsageBreakdown(
                        cachedInputTokens=25,
                        inputTokens=1000,
                        outputTokens=250,
                        reasoningOutputTokens=75,
                        totalTokens=1250,
                    ),
                    modelContextWindow=128000,
                ),
            ),
        )
    )
    await client.notifications.put(
        Notification(
            method="turn/diff/updated",
            payload=TurnDiffUpdatedNotification(
                diff=(
                    "diff --git a/example.txt b/example.txt\n"
                    "--- a/example.txt\n"
                    "+++ b/example.txt\n"
                    "@@ -1 +1 @@\n"
                    "-old\n"
                    "+new\n"
                ),
                threadId="thread-1",
                turnId="codex-turn-1",
            ),
        )
    )
    await client.notifications.put(
        Notification(
            method="turn/diff/completed",
            payload=UnknownNotification(
                params={
                    "diff": "*** Begin Patch\n*** Update File: example.txt\n+done\n",
                    "threadId": "thread-1",
                    "turnId": "codex-turn-1",
                }
            ),
        )
    )
    await client.notifications.put(
        Notification(
            method="turn/completed",
            payload=TurnCompletedNotification(
                threadId="thread-1",
                turn=Turn(id="codex-turn-1", items=[], status=TurnStatus.completed),
            ),
        )
    )

    task = process._active_turn_task
    assert task is not None
    await task

    assert client.turn_start_calls == [
        (
            "thread-1",
            [{"type": "text", "text": "hello"}],
            {"model": "gpt-5.5"},
        )
    ]
    assert client.unregistered_turns == ["codex-turn-1"]
    status_texts = [
        message.status
        for message in emitted
        if message.type == AGENT_EVENT_THREAD_STATUS and message.status is not None
    ]
    assert "Wrapping up" in status_texts
    assert "Writing file" in status_texts
    assert "Completed" not in status_texts
    usage_updates = [
        message
        for message in process.messages
        if message.type == AGENT_EVENT_USAGE_UPDATED
    ]
    assert len(usage_updates) == 1
    assert usage_updates[0].usage == {
        "input_tokens": 1000.0,
        "cached_input_tokens": 25.0,
        "output_tokens": 250.0,
        "reasoning_output_tokens": 75.0,
        "total_tokens": 1250.0,
    }
    assert usage_updates[0].context_window.used_tokens == 1250
    assert usage_updates[0].context_window.total_tokens == 128000
    diff_started = [
        message
        for message in process.messages
        if message.type == AGENT_EVENT_TOOL_CALL_STARTED
        and message.toolkit == "codex"
        and message.tool == "diff_updated"
    ]
    assert len(diff_started) == 1
    assert diff_started[0].item_id == "codex-turn-1:codex-diff_updated"
    assert diff_started[0].namespace == "codex"
    assert diff_started[0].arguments["diff"].startswith("diff --git")
    diff_ended = [
        message
        for message in process.messages
        if message.type == AGENT_EVENT_TOOL_CALL_ENDED
        and message.toolkit == "codex"
        and message.tool == "diff_updated"
    ]
    assert len(diff_ended) == 1
    assert diff_ended[0].item_id == "codex-turn-1:codex-diff_updated"
    diff_completed_started = [
        message
        for message in process.messages
        if message.type == AGENT_EVENT_TOOL_CALL_STARTED
        and message.toolkit == "codex"
        and message.tool == "diff_completed"
    ]
    assert len(diff_completed_started) == 1
    assert diff_completed_started[0].arguments["diff"].startswith("*** Begin Patch")
    diff_completed_ended = [
        message
        for message in process.messages
        if message.type == AGENT_EVENT_TOOL_CALL_ENDED
        and message.toolkit == "codex"
        and message.tool == "diff_completed"
    ]
    assert len(diff_completed_ended) == 1
    assert diff_completed_ended[0].item_id == "codex-turn-1:codex-diff_completed"
    assert [
        message.type
        for message in process.messages
        if message.type
        in {
            AGENT_EVENT_TURN_START_ACCEPTED,
            AGENT_EVENT_TURN_STARTED,
            AGENT_EVENT_TEXT_CONTENT_DELTA,
            AGENT_EVENT_TURN_ENDED,
        }
    ] == [
        AGENT_EVENT_TURN_START_ACCEPTED,
        AGENT_EVENT_TURN_STARTED,
        AGENT_EVENT_TEXT_CONTENT_DELTA,
        AGENT_EVENT_TURN_ENDED,
    ]
    assert any(message.type == AGENT_EVENT_THREAD_STATUS for message in emitted)


@pytest.mark.asyncio
async def test_codex_agent_process_does_not_duplicate_completed_item_text_after_delta() -> (
    None
):
    client = _FakeCodexClient()
    process = _CollectingCodexAgentProcess(
        thread_id="thread-1",
        participant=None,
        client=client,
        provider_name="codex",
        default_model="gpt-5.5",
        provider_info_builder=_provider_info,
    )

    await process.on_turn_start(
        message=type(
            "Message",
            (),
            {
                "sender": None,
                "data": TurnStart(
                    type=AGENT_MESSAGE_TURN_START,
                    thread_id="thread-1",
                    content=[AgentTextContent(type="text", text="hello")],
                    model="gpt-5.5",
                ),
            },
        )()
    )
    await client.notifications.put(
        Notification(
            method="item/agentMessage/delta",
            payload=AgentMessageDeltaNotification(
                delta="codex run turn ok",
                itemId="item-1",
                threadId="thread-1",
                turnId="codex-turn-1",
            ),
        )
    )
    await client.notifications.put(
        Notification(
            method="item/completed",
            payload=ItemCompletedNotification(
                completedAtMs=1,
                threadId="thread-1",
                turnId="codex-turn-1",
                item=ThreadItem(
                    root=AgentMessageThreadItem(
                        id="item-1",
                        text="codex run turn ok",
                        type="agentMessage",
                    )
                ),
            ),
        )
    )
    await client.notifications.put(
        Notification(
            method="turn/completed",
            payload=TurnCompletedNotification(
                threadId="thread-1",
                turn=Turn(id="codex-turn-1", items=[], status=TurnStatus.completed),
            ),
        )
    )

    task = process._active_turn_task
    assert task is not None
    await task

    text = "".join(
        message.text
        for message in process.messages
        if message.type == AGENT_EVENT_TEXT_CONTENT_DELTA
    )
    assert text == "codex run turn ok"


@pytest.mark.asyncio
async def test_codex_agent_process_loads_thread_messages_on_open() -> None:
    client = _FakeCodexClient()
    client.thread_read_response = type(
        "ThreadReadResponse",
        (),
        {
            "model": "gpt-5.5",
            "thread": type(
                "Thread",
                (),
                {
                    "id": "thread-1",
                    "turns": [
                        Turn(
                            id="turn-1",
                            items=[
                                ThreadItem(
                                    root=UserMessageThreadItem(
                                        id="user-item-1",
                                        type="userMessage",
                                        content=[
                                            UserInput(
                                                root=TextUserInput(
                                                    type="text",
                                                    text="loaded hello",
                                                )
                                            )
                                        ],
                                    )
                                ),
                                ThreadItem(
                                    root=AgentMessageThreadItem(
                                        id="agent-item-1",
                                        text="loaded response",
                                        type="agentMessage",
                                    )
                                ),
                            ],
                            status=TurnStatus.completed,
                        )
                    ],
                },
            )(),
        },
    )()
    process = _CollectingCodexAgentProcess(
        thread_id="thread-1",
        participant=None,
        client=client,
        provider_name="codex",
        default_model="gpt-5.5",
        provider_info_builder=_provider_info,
    )

    await process.on_thread_open(
        message=type(
            "Message",
            (),
            {
                "sender": None,
                "data": OpenThread(
                    type=AGENT_MESSAGE_THREAD_OPEN,
                    thread_id="thread-1",
                    load=True,
                ),
            },
        )()
    )

    assert [message.type for message in process.messages[:4]] == [
        AGENT_EVENT_TURN_START_ACCEPTED,
        AGENT_EVENT_TEXT_CONTENT_DELTA,
        AGENT_EVENT_TURN_ENDED,
        AGENT_EVENT_THREAD_LOADED,
    ]
    assert client.thread_resume_calls == [("thread-1", {"model": "gpt-5.5"})]
    assert process.messages[0].content == [
        AgentTextContent(type="text", text="loaded hello")
    ]
    assert process.messages[1].text == "loaded response"

    session = _NoopChatClient()._create_thread_session(thread_path="thread-1")
    for agent_message in process.messages:
        session.add_agent_message(agent_message)
    assert session.queued_message_labels == ()
    assert session.active_turn_id is None
    assert session.last_completed_turn_id == "turn-1"


@pytest.mark.asyncio
async def test_chat_thread_session_marks_steer_queued_immediately() -> None:
    client = _NoopChatClient()
    session = client._create_thread_session(thread_path="thread-1")
    session._local_agent_message_ids.add("input-1")
    client._handle_agent_payload(
        {
            "type": AGENT_EVENT_TURN_START_ACCEPTED,
            "thread_id": "thread-1",
            "turn_id": "turn-1",
            "source_message_id": "input-1",
        }
    )

    message_id = session.steer(prompt="adjust this")

    assert message_id is not None
    assert session.queued_message_labels == ("user: adjust this",)


@pytest.mark.asyncio
async def test_codex_agent_process_accepts_steer_before_handoff_completes() -> None:
    class _BlockingSteerClient(_FakeCodexClient):
        def __init__(self) -> None:
            super().__init__()
            self.steer_called = asyncio.Event()
            self.release_steer = asyncio.Event()

        async def turn_steer(
            self,
            thread_id: str,
            expected_turn_id: str,
            input_items: list[dict[str, Any]] | dict[str, Any] | str,
        ) -> None:
            self.turn_steer_calls.append((thread_id, expected_turn_id, input_items))
            self.steer_called.set()
            await self.release_steer.wait()

    client = _BlockingSteerClient()
    process = _CollectingCodexAgentProcess(
        thread_id="thread-1",
        participant=None,
        client=client,
        provider_name="codex",
        default_model="gpt-5.5",
        provider_info_builder=_provider_info,
    )

    await process.on_turn_start(
        message=type(
            "Message",
            (),
            {
                "sender": None,
                "data": TurnStart(
                    type=AGENT_MESSAGE_TURN_START,
                    thread_id="thread-1",
                    content=[AgentTextContent(type="text", text="hello")],
                    model="gpt-5.5",
                ),
            },
        )()
    )
    await process.on_turn_steer(
        message=type(
            "Message",
            (),
            {
                "sender": None,
                "data": TurnSteer(
                    type=AGENT_MESSAGE_TURN_STEER,
                    thread_id="thread-1",
                    turn_id="codex-turn-1",
                    content=[AgentTextContent(type="text", text="steer")],
                ),
            },
        )()
    )
    await asyncio.wait_for(client.steer_called.wait(), timeout=1)

    assert any(
        message.type == AGENT_EVENT_TURN_STEER_ACCEPTED for message in process.messages
    )
    assert any(message.type == AGENT_EVENT_TURN_STEERED for message in process.messages)

    client.release_steer.set()
    while len(process._steer_tasks) > 0:
        await asyncio.sleep(0)

    assert client.turn_steer_calls == [
        ("thread-1", "codex-turn-1", [{"type": "text", "text": "steer"}])
    ]

    task = process._active_turn_task
    assert task is not None
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task


@pytest.mark.asyncio
async def test_codex_agent_process_handles_failed_terminal_turn() -> None:
    client = _FakeCodexClient()
    process = _CollectingCodexAgentProcess(
        thread_id="thread-1",
        participant=None,
        client=client,
        provider_name="codex",
        default_model="gpt-5.5",
        provider_info_builder=_provider_info,
    )

    await process.on_turn_start(
        message=type(
            "Message",
            (),
            {
                "sender": None,
                "data": TurnStart(
                    type=AGENT_MESSAGE_TURN_START,
                    thread_id="thread-1",
                    content=[AgentTextContent(type="text", text="hello")],
                    model="gpt-5.5",
                ),
            },
        )()
    )
    await client.notifications.put(
        Notification(
            method="turn/completed",
            payload=TurnCompletedNotification(
                threadId="thread-1",
                turn=Turn(
                    id="codex-turn-1",
                    items=[],
                    status=TurnStatus.failed,
                    error=TurnError(message="authentication failed"),
                ),
            ),
        )
    )

    task = process._active_turn_task
    assert task is not None
    await task

    ended = next(
        message
        for message in process.messages
        if message.type == AGENT_EVENT_TURN_ENDED
    )
    assert ended.error is not None
    assert ended.error.message == "authentication failed"
    assert ended.error.code == "codex_turn_failed"


@pytest.mark.asyncio
async def test_codex_agent_process_reports_error_notification_on_failed_turn_end() -> (
    None
):
    client = _FakeCodexClient()
    process = _CollectingCodexAgentProcess(
        thread_id="thread-1",
        participant=None,
        client=client,
        provider_name="codex",
        default_model="gpt-5.5",
        provider_info_builder=_provider_info,
    )

    await process.on_turn_start(
        message=type(
            "Message",
            (),
            {
                "sender": None,
                "data": TurnStart(
                    type=AGENT_MESSAGE_TURN_START,
                    thread_id="thread-1",
                    content=[AgentTextContent(type="text", text="hello")],
                    model="gpt-5.5",
                ),
            },
        )()
    )
    await client.notifications.put(
        Notification(
            method="error",
            payload=ErrorNotification(
                error=TurnError(message="model not supported"),
                threadId="thread-1",
                turnId="codex-turn-1",
                willRetry=False,
            ),
        )
    )
    await client.notifications.put(
        Notification(
            method="turn/completed",
            payload=TurnCompletedNotification(
                threadId="thread-1",
                turn=Turn(
                    id="codex-turn-1",
                    items=[],
                    status=TurnStatus.failed,
                    error=TurnError(message="model not supported"),
                ),
            ),
        )
    )

    task = process._active_turn_task
    assert task is not None
    await task

    ended = next(
        message
        for message in process.messages
        if message.type == AGENT_EVENT_TURN_ENDED
    )
    assert ended.error is not None
    assert ended.error.message == "model not supported"
    assert ended.error.code == "codex_turn_failed"
