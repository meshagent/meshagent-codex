from __future__ import annotations

import asyncio
import inspect

import pytest

from meshagent.agents.chat_client import LocalChatClient
from meshagent.agents.messages import (
    AGENT_EVENT_MODEL_CHANGED,
    AGENT_EVENT_THREAD_LISTED,
    AGENT_EVENT_THREAD_LOADED,
    AGENT_EVENT_THREAD_STARTED,
    AGENT_EVENT_THREAD_STATUS,
    AGENT_EVENT_TURN_ENDED,
    AGENT_MESSAGE_THREAD_LIST,
    AGENT_MESSAGE_THREAD_START,
    AgentRealtimeConnectionInfo,
    AgentTextContent,
    AgentTextContentDelta,
    AgentThreadListEntry,
    AgentThreadStatus,
    ListThreads,
    StartThread,
    ThreadStarted,
    ThreadsListed,
)
from meshagent.agents.process import Message
from meshagent.api import Participant
from meshagent.cli import process as process_cli

from .supervisor import CodexAgentSupervisor
from .vendor.openai_codex.client import AppServerConfig


class _LocalEventCodexSupervisor(CodexAgentSupervisor):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self._local_event_queues: list[asyncio.Queue[Message]] = []

    def subscribe_local_events(self) -> asyncio.Queue[Message]:
        queue: asyncio.Queue[Message] = asyncio.Queue()
        self._local_event_queues.append(queue)
        return queue

    def unsubscribe_local_events(self, queue: asyncio.Queue[Message]) -> None:
        if queue in self._local_event_queues:
            self._local_event_queues.remove(queue)

    def _send_to_local_event_queues(self, message: Message) -> None:
        for queue in [*self._local_event_queues]:
            queue.put_nowait(message)

    def send(self, message: Message) -> None:
        if message.source is not None:
            self._send_to_local_event_queues(message)
        super().send(message)

    def _send_to_channels(self, message: Message) -> None:
        if isinstance(message.data, ThreadsListed):
            self._send_to_local_event_queues(message)
        elif message.source is None and message.data.type in {
            AGENT_EVENT_MODEL_CHANGED,
            AGENT_EVENT_THREAD_STATUS,
        }:
            self._send_to_local_event_queues(message)
        super()._send_to_channels(message)

    def _emit_thread_started(
        self,
        *,
        start_thread: StartThread,
        sender: Participant | None,
        thread_id: str,
        realtime_connection: AgentRealtimeConnectionInfo | None = None,
    ) -> None:
        thread_started = ThreadStarted(
            type=AGENT_EVENT_THREAD_STARTED,
            source_message_id=start_thread.message_id,
            thread_id=thread_id,
            realtime_connection=realtime_connection,
        )
        self._send_to_local_event_queues(Message(data=thread_started, sender=sender))
        self._send_to_channels(Message(data=thread_started, sender=sender))

    async def _route(self, message: Message) -> None:
        if message.data.type == AGENT_MESSAGE_THREAD_LIST:
            list_threads = ListThreads.model_validate(
                message.data.model_dump(mode="python")
            )
            page = await self.list_threads(
                list_threads=list_threads, sender=message.sender
            )
            response = ThreadsListed(
                type=AGENT_EVENT_THREAD_LISTED,
                source_message_id=list_threads.message_id,
                threads=[
                    AgentThreadListEntry(
                        path=entry.path,
                        name=entry.name,
                        created_at=entry.created_at,
                        modified_at=entry.modified_at,
                    )
                    for entry in page.threads
                ],
                total=page.total,
                offset=page.offset,
                limit=page.limit,
            )
            response_message = Message(data=response, sender=message.sender)
            self._send_to_local_event_queues(response_message)
            super()._send_to_channels(response_message)
            return
        await super()._route(message)


def _codex_config() -> AppServerConfig:
    return AppServerConfig(
        codex_bin="/tmp/codex",
        cwd=None,
        config_overrides=(),
        client_name="meshagent_codex_test",
        client_title="MeshAgent Codex Test",
    )


@pytest.mark.asyncio
async def test_codex_supervisor_forwards_thread_started_to_local_events() -> None:
    supervisor = _LocalEventCodexSupervisor(
        participant=Participant(id="codex", attributes={"name": "codex"}),
        config=_codex_config(),
        default_model="gpt-5.5",
    )
    events = supervisor.subscribe_local_events()
    participant = Participant(id="client", attributes={"name": "client"})

    supervisor._emit_thread_started(
        start_thread=StartThread(
            type=AGENT_MESSAGE_THREAD_START,
            message_id="message-1",
            content=[AgentTextContent(type="text", text="hello")],
        ),
        sender=participant,
        thread_id="thread-1",
    )

    event = await asyncio.wait_for(events.get(), timeout=1)
    assert isinstance(event.data, ThreadStarted)
    assert event.sender == participant
    assert event.data.thread_id == "thread-1"


@pytest.mark.asyncio
async def test_codex_supervisor_forwards_thread_list_response_locally() -> None:
    class _FakeCodexClient:
        async def start(self) -> None:
            return None

        async def close(self) -> None:
            return None

        async def initialize(self):
            return None

        async def model_list(self, include_hidden: bool = False):
            del include_hidden
            return type("ModelListResponse", (), {"data": []})()

        async def thread_list(self, params: dict | None = None):
            del params
            return type(
                "ThreadListResponse",
                (),
                {
                    "data": [
                        type(
                            "Thread",
                            (),
                            {
                                "id": "codex-thread-1",
                                "name": None,
                                "preview": "Codex Thread Preview",
                                "created_at": 1,
                                "updated_at": 2,
                            },
                        )()
                    ]
                },
            )()

    supervisor = _LocalEventCodexSupervisor(
        participant=Participant(id="codex", attributes={"name": "codex"}),
        config=_codex_config(),
        client_factory=lambda _config: _FakeCodexClient(),
        default_model="gpt-5.5",
    )
    events = supervisor.subscribe_local_events()
    client = LocalChatClient(
        thread_path=None,
        send_message=supervisor.send,
        events=events,
        on_close=lambda: supervisor.unsubscribe_local_events(events),
    )
    try:
        await supervisor.start()
        await client.start()

        response = await asyncio.wait_for(
            client.thread_session.list_threads(limit=100, offset=0),
            timeout=1,
        )
    finally:
        await client.close()
        await supervisor.stop()

    assert [(thread.name, thread.path) for thread in response.threads] == [
        ("Codex Thread Preview", "codex-thread-1")
    ]


@pytest.mark.asyncio
async def test_codex_supervisor_forwards_source_less_status_locally() -> None:
    supervisor = _LocalEventCodexSupervisor(
        participant=Participant(id="codex", attributes={"name": "codex"}),
        config=_codex_config(),
        default_model="gpt-5.5",
    )
    events = supervisor.subscribe_local_events()
    status = AgentThreadStatus(
        type=AGENT_EVENT_THREAD_STATUS,
        thread_id="thread-1",
        turn_id="turn-1",
        status="Thinking",
        mode="steerable",
    )

    supervisor._send_to_channels(Message(data=status))

    event = await asyncio.wait_for(events.get(), timeout=1)
    assert event.data == status


@pytest.mark.asyncio
async def test_codex_loaded_thread_accepts_followup_turn() -> None:
    from .vendor.openai_codex.generated.v2_all import (
        AgentMessageDeltaNotification,
        AgentMessageThreadItem,
        TextUserInput,
        ThreadItem,
        Turn,
        TurnCompletedNotification,
        TurnStartResponse,
        TurnStatus,
        UserInput,
        UserMessageThreadItem,
    )
    from .vendor.openai_codex.models import Notification

    class _FakeCodexClient:
        def __init__(self) -> None:
            self.notifications: asyncio.Queue[Notification] = asyncio.Queue()
            self.turn_start_calls: list[tuple[str, object, object]] = []

        async def start(self) -> None:
            return None

        async def close(self) -> None:
            return None

        async def initialize(self):
            return None

        async def model_list(self, include_hidden: bool = False):
            del include_hidden
            return type("ModelListResponse", (), {"data": []})()

        async def thread_read(self, thread_id: str, include_turns: bool = False):
            del include_turns
            return type(
                "ThreadReadResponse",
                (),
                {
                    "thread": type(
                        "Thread",
                        (),
                        {
                            "id": thread_id,
                            "turns": [
                                Turn(
                                    id="loaded-turn-1",
                                    items=[
                                        ThreadItem(
                                            root=UserMessageThreadItem(
                                                id="loaded-user-1",
                                                type="userMessage",
                                                content=[
                                                    UserInput(
                                                        root=TextUserInput(
                                                            type="text",
                                                            text="loaded prompt",
                                                        )
                                                    )
                                                ],
                                            )
                                        ),
                                        ThreadItem(
                                            root=AgentMessageThreadItem(
                                                id="loaded-agent-1",
                                                type="agentMessage",
                                                text="loaded response",
                                            )
                                        ),
                                    ],
                                    status=TurnStatus.completed,
                                )
                            ],
                        },
                    )()
                },
            )()

        async def thread_resume(self, thread_id: str, params: dict | None = None):
            del params
            response = await self.thread_read(thread_id=thread_id, include_turns=True)
            response.model = "gpt-5.5"
            return response

        async def turn_start(self, thread_id, input_items, params=None):
            self.turn_start_calls.append((thread_id, input_items, params))
            await self.notifications.put(
                Notification(
                    method="item/agentMessage/delta",
                    payload=AgentMessageDeltaNotification(
                        delta="followup response",
                        itemId="followup-item-1",
                        threadId=thread_id,
                        turnId="followup-turn-1",
                    ),
                )
            )
            await self.notifications.put(
                Notification(
                    method="turn/completed",
                    payload=TurnCompletedNotification(
                        threadId=thread_id,
                        turn=Turn(
                            id="followup-turn-1",
                            items=[],
                            status=TurnStatus.completed,
                        ),
                    ),
                )
            )
            return TurnStartResponse(
                turn=Turn(
                    id="followup-turn-1",
                    items=[],
                    status=TurnStatus.in_progress,
                )
            )

        async def turn_steer(self, thread_id, expected_turn_id, input_items) -> None:
            del thread_id
            del expected_turn_id
            del input_items

        async def turn_interrupt(self, thread_id, turn_id) -> None:
            del thread_id
            del turn_id

        async def next_turn_notification(self, turn_id: str) -> Notification:
            del turn_id
            return await self.notifications.get()

        def unregister_turn_notifications(self, turn_id: str) -> None:
            del turn_id

    fake_client = _FakeCodexClient()
    supervisor = _LocalEventCodexSupervisor(
        participant=Participant(id="codex", attributes={"name": "codex"}),
        config=_codex_config(),
        client_factory=lambda _config: fake_client,
        default_model="gpt-5.5",
    )
    events = supervisor.subscribe_local_events()
    client = LocalChatClient(
        thread_path=None,
        send_message=supervisor.send,
        events=events,
        on_close=lambda: supervisor.unsubscribe_local_events(events),
        local_participant_name="you",
    )
    try:
        await supervisor.start()
        await client.start()
        session = await client.open_thread(
            "thread-1",
            local_participant_name="you",
            close_client_on_close=False,
            load=True,
        )
        while True:
            event = await asyncio.wait_for(session.receive(), timeout=1)
            if event.get("type") == AGENT_EVENT_THREAD_LOADED:
                break

        await session.send_text(text="followup prompt")
        while True:
            event = await asyncio.wait_for(session.receive(), timeout=1)
            if event.get("type") == AGENT_EVENT_TURN_ENDED:
                break
    finally:
        await client.close()
        await supervisor.stop()

    assert fake_client.turn_start_calls == [
        (
            "thread-1",
            [{"type": "text", "text": "followup prompt"}],
            {"model": "gpt-5.5"},
        )
    ]
    text = "".join(
        message.text
        for message in session.messages
        if isinstance(message, AgentTextContentDelta)
        and message.turn_id == "followup-turn-1"
    )
    assert text == "followup response"


@pytest.mark.asyncio
async def test_process_run_message_waits_for_loaded_codex_thread_before_turn() -> None:
    from .vendor.openai_codex.generated.v2_all import (
        AgentMessageDeltaNotification,
        Turn,
        TurnCompletedNotification,
        TurnStartResponse,
        TurnStatus,
    )
    from .vendor.openai_codex.models import Notification

    class _FakeCodexClient:
        def __init__(self) -> None:
            self.notifications: asyncio.Queue[Notification] = asyncio.Queue()
            self.thread_read_completed = False
            self.turn_start_calls: list[tuple[str, object, object, bool]] = []

        async def start(self) -> None:
            return None

        async def close(self) -> None:
            return None

        async def initialize(self):
            return None

        async def model_list(self, include_hidden: bool = False):
            del include_hidden
            return type("ModelListResponse", (), {"data": []})()

        async def thread_read(self, thread_id: str, include_turns: bool = False):
            del include_turns
            self.thread_read_completed = True
            return type(
                "ThreadReadResponse",
                (),
                {
                    "thread": type(
                        "Thread",
                        (),
                        {
                            "id": thread_id,
                            "name": "Existing thread",
                            "preview": "loaded preview",
                            "created_at": 1,
                            "updated_at": 2,
                            "turns": [],
                        },
                    )()
                },
            )()

        async def thread_resume(self, thread_id: str, params: dict | None = None):
            del params
            response = await self.thread_read(thread_id=thread_id, include_turns=True)
            response.model = "gpt-5.5"
            return response

        async def turn_start(self, thread_id, input_items, params=None):
            self.turn_start_calls.append(
                (thread_id, input_items, params, self.thread_read_completed)
            )
            await self.notifications.put(
                Notification(
                    method="item/agentMessage/delta",
                    payload=AgentMessageDeltaNotification(
                        delta="followup response",
                        itemId="followup-item-1",
                        threadId=thread_id,
                        turnId="followup-turn-1",
                    ),
                )
            )
            await self.notifications.put(
                Notification(
                    method="turn/completed",
                    payload=TurnCompletedNotification(
                        threadId=thread_id,
                        turn=Turn(
                            id="followup-turn-1",
                            items=[],
                            status=TurnStatus.completed,
                        ),
                    ),
                )
            )
            return TurnStartResponse(
                turn=Turn(
                    id="followup-turn-1",
                    items=[],
                    status=TurnStatus.in_progress,
                )
            )

        async def turn_steer(self, thread_id, expected_turn_id, input_items) -> None:
            del thread_id
            del expected_turn_id
            del input_items

        async def turn_interrupt(self, thread_id, turn_id) -> None:
            del thread_id
            del turn_id

        async def next_turn_notification(self, turn_id: str) -> Notification:
            del turn_id
            return await self.notifications.get()

        def unregister_turn_notifications(self, turn_id: str) -> None:
            del turn_id

    fake_client = _FakeCodexClient()
    supervisor = _LocalEventCodexSupervisor(
        participant=Participant(id="codex", attributes={"name": "codex"}),
        config=_codex_config(),
        client_factory=lambda _config: fake_client,
        default_model="gpt-5.5",
    )
    bot = type("CodexBot", (), {"_supervisor": supervisor})()
    try:
        await supervisor.start()
        await asyncio.wait_for(
            process_cli._run_process_run_tui(
                bot=bot,
                model="codex/gpt-5.5",
                thread_path="thread-1",
                thread_storage="codex",
                agent_name="codex",
                thread_dir=None,
                threading_mode="none",
                message="followup prompt",
                working_dir=None,
            ),
            timeout=2,
        )
    finally:
        await supervisor.stop()

    assert fake_client.turn_start_calls == [
        (
            "thread-1",
            [{"type": "text", "text": "followup prompt"}],
            {"model": "gpt-5.5"},
            True,
        )
    ]


@pytest.mark.asyncio
async def test_process_run_tui_sidebar_loaded_codex_thread_accepts_followup_turn(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from meshagent.cli import ask as ask_module

    from .vendor.openai_codex.generated.v2_all import (
        AgentMessageDeltaNotification,
        Turn,
        TurnCompletedNotification,
        TurnStartResponse,
        TurnStatus,
    )
    from .vendor.openai_codex.models import Notification

    class _FakeCodexClient:
        def __init__(self) -> None:
            self.notifications: asyncio.Queue[Notification] = asyncio.Queue()
            self.thread_list_called = asyncio.Event()
            self.thread_read_completed = False
            self.turn_start_calls: list[tuple[str, object, object, bool]] = []

        async def start(self) -> None:
            return None

        async def close(self) -> None:
            return None

        async def initialize(self):
            return None

        async def model_list(self, include_hidden: bool = False):
            del include_hidden
            return type("ModelListResponse", (), {"data": []})()

        async def thread_list(self, params: dict | None = None):
            del params
            self.thread_list_called.set()
            return type(
                "ThreadListResponse",
                (),
                {
                    "data": [
                        type(
                            "Thread",
                            (),
                            {
                                "id": "thread-1",
                                "name": "Existing thread",
                                "preview": "loaded preview",
                                "created_at": 1,
                                "updated_at": 2,
                            },
                        )()
                    ]
                },
            )()

        async def thread_read(self, thread_id: str, include_turns: bool = False):
            del include_turns
            self.thread_read_completed = True
            return type(
                "ThreadReadResponse",
                (),
                {
                    "thread": type(
                        "Thread",
                        (),
                        {
                            "id": thread_id,
                            "name": "Existing thread",
                            "preview": "loaded preview",
                            "created_at": 1,
                            "updated_at": 2,
                            "turns": [],
                        },
                    )()
                },
            )()

        async def thread_resume(self, thread_id: str, params: dict | None = None):
            del params
            response = await self.thread_read(thread_id=thread_id, include_turns=True)
            response.model = "gpt-5.5"
            return response

        async def turn_start(self, thread_id, input_items, params=None):
            self.turn_start_calls.append(
                (thread_id, input_items, params, self.thread_read_completed)
            )
            await self.notifications.put(
                Notification(
                    method="item/agentMessage/delta",
                    payload=AgentMessageDeltaNotification(
                        delta="followup response",
                        itemId="followup-item-1",
                        threadId=thread_id,
                        turnId="followup-turn-1",
                    ),
                )
            )
            await self.notifications.put(
                Notification(
                    method="turn/completed",
                    payload=TurnCompletedNotification(
                        threadId=thread_id,
                        turn=Turn(
                            id="followup-turn-1",
                            items=[],
                            status=TurnStatus.completed,
                        ),
                    ),
                )
            )
            return TurnStartResponse(
                turn=Turn(
                    id="followup-turn-1",
                    items=[],
                    status=TurnStatus.in_progress,
                )
            )

        async def turn_steer(self, thread_id, expected_turn_id, input_items) -> None:
            del thread_id
            del expected_turn_id
            del input_items

        async def turn_interrupt(self, thread_id, turn_id) -> None:
            del thread_id
            del turn_id

        async def next_turn_notification(self, turn_id: str) -> Notification:
            del turn_id
            return await self.notifications.get()

        def unregister_turn_notifications(self, turn_id: str) -> None:
            del turn_id

    async def fake_run_ask_tui(**kwargs):
        await asyncio.wait_for(fake_client.thread_list_called.wait(), timeout=1)
        handled = False
        deadline = asyncio.get_running_loop().time() + 1
        while not handled:
            kwargs["side_panel_renderer"](True, width=32, height=10)
            handled_result = kwargs["side_panel_mouse_handler"](0, 1)
            if inspect.isawaitable(handled_result):
                handled_result = await handled_result
            handled = handled_result is True
            if handled:
                break
            if asyncio.get_running_loop().time() >= deadline:
                raise AssertionError("thread sidebar did not open selected thread")
            await asyncio.sleep(0.01)
        session = kwargs["session_provider"]()
        assert session.thread_path == "thread-1"
        await session.send_text(text="followup prompt")
        while True:
            event = await asyncio.wait_for(session.receive(), timeout=1)
            if event.get("type") == AGENT_EVENT_TURN_ENDED:
                return

    fake_client = _FakeCodexClient()
    supervisor = _LocalEventCodexSupervisor(
        participant=Participant(id="codex", attributes={"name": "codex"}),
        config=_codex_config(),
        client_factory=lambda _config: fake_client,
        default_model="gpt-5.5",
    )
    bot = type("CodexBot", (), {"_supervisor": supervisor})()
    monkeypatch.setattr(ask_module, "_run_ask_tui", fake_run_ask_tui)
    try:
        await supervisor.start()
        await asyncio.wait_for(
            process_cli._run_process_run_tui(
                bot=bot,
                model="codex/gpt-5.5",
                thread_path=None,
                thread_storage="codex",
                agent_name="codex",
                thread_dir="threads",
                threading_mode="default-new",
                message=None,
                working_dir=None,
            ),
            timeout=2,
        )
    finally:
        await supervisor.stop()

    assert fake_client.turn_start_calls == [
        (
            "thread-1",
            [{"type": "text", "text": "followup prompt"}],
            {"model": "gpt-5.5"},
            True,
        )
    ]
