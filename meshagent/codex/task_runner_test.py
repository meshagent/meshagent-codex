import re

import pytest
from meshagent.agents.adapter import LLMAdapter
from meshagent.agents.context import AgentSessionContext
from meshagent.agents.task_runner import TaskContext

import meshagent.codex.task_runner as task_runner_module
from meshagent.codex.task_runner import CodexTaskRunner


class _FakeParticipant:
    def __init__(self, *, name: str, participant_id: str):
        self._name = name
        self.id = participant_id

    def get_attribute(self, key: str):
        if key == "name":
            return self._name
        return None


class _FakeRoom:
    def __init__(self):
        self.local_participant = _FakeParticipant(
            name="assistant",
            participant_id="assistant-id",
        )


class _FakeBackend:
    def __init__(self):
        self.open_thread_key = None
        self.closed_thread_key = None
        self.used_event_handler = False

    async def on_thread_open(
        self,
        *,
        thread_key: str,
        room,
        context,
        model: str | None = None,
        skill_dirs: list[str] | None = None,
    ) -> None:
        del room, context, model, skill_dirs
        self.open_thread_key = thread_key

    async def next(
        self,
        *,
        thread_key: str,
        message,
        developer_instructions,
        room,
        toolkits,
        event_handler,
        model: str | None = None,
        on_behalf_of=None,
    ) -> str:
        del (
            thread_key,
            message,
            developer_instructions,
            room,
            toolkits,
            model,
            on_behalf_of,
        )
        if event_handler is not None:
            self.used_event_handler = True
            event_handler(
                {
                    "method": "item/started",
                    "params": {
                        "item": {"id": "item-1", "type": "agent_message"},
                    },
                }
            )
            event_handler(
                {
                    "method": "item/agentmessage/delta",
                    "params": {"delta": "assistant "},
                }
            )
            event_handler(
                {
                    "method": "item/completed",
                    "params": {
                        "item": {
                            "id": "item-1",
                            "type": "agent_message",
                            "text": "assistant response",
                        }
                    },
                }
            )
        return "assistant response"

    async def on_thread_close(self, *, thread_key: str) -> None:
        self.closed_thread_key = thread_key


class _FakeLLMAdapter(LLMAdapter):
    def __init__(self, *, generated_thread_name: str = "release planning"):
        self.generated_thread_name = generated_thread_name
        self.calls: list[dict] = []

    def default_model(self) -> str:
        return "test-model"

    async def next(
        self,
        *,
        context,
        room,
        toolkits,
        output_schema=None,
        event_handler=None,
        model=None,
        on_behalf_of=None,
    ):
        del event_handler
        self.calls.append(
            {
                "context": context,
                "room": room,
                "toolkits": toolkits,
                "output_schema": output_schema,
                "model": model,
                "on_behalf_of": on_behalf_of,
            }
        )

        if output_schema is not None:
            return {"thread_name": self.generated_thread_name}

        return "assistant response"


class _FakeThreadAdapter:
    instances: list["_FakeThreadAdapter"] = []

    def __init__(self, *, room, path: str):
        del room
        self.path = path
        self.started = False
        self.stopped = False
        self.appended = False
        self.writes: list[tuple[str, str]] = []
        self.events: list[dict] = []
        _FakeThreadAdapter.instances.append(self)

    async def start(self) -> None:
        self.started = True

    async def stop(self) -> None:
        self.stopped = True

    def append_messages(self, *, context: AgentSessionContext) -> None:
        del context
        self.appended = True

    def write_text_message(self, *, text: str, participant) -> None:
        if isinstance(participant, str):
            participant_name = participant
        else:
            participant_name = participant.get_attribute("name") or ""

        self.writes.append((text, participant_name))

    def push(self, *, event: dict) -> None:
        self.events.append(event)


def test_task_runner_input_schema_path_only_for_manual_mode() -> None:
    llm_adapter = _FakeLLMAdapter()
    manual_runner = CodexTaskRunner(threading_mode="manual")
    auto_runner = CodexTaskRunner(llm_adapter=llm_adapter, threading_mode="auto")
    none_runner = CodexTaskRunner(threading_mode="none")

    assert "path" in manual_runner.input_schema["properties"]
    assert "path" not in auto_runner.input_schema["properties"]
    assert "path" not in none_runner.input_schema["properties"]


def test_task_runner_fallback_thread_name_uses_timestamp() -> None:
    runner = CodexTaskRunner(threading_mode="manual")
    thread_name = runner._fallback_thread_name(prompt="any prompt")

    assert re.fullmatch(r"thread-\d{8}-\d{6}", thread_name) is not None


def test_task_runner_auto_threading_requires_llm_adapter() -> None:
    with pytest.raises(
        ValueError,
        match=re.escape("`llm_adapter` is required when `threading_mode` is 'auto'"),
    ):
        CodexTaskRunner(threading_mode="auto")


@pytest.mark.asyncio
async def test_task_runner_manual_threading_uses_selected_thread_path(
    monkeypatch,
) -> None:
    _FakeThreadAdapter.instances.clear()
    monkeypatch.setattr(task_runner_module, "CodexThreadAdapter", _FakeThreadAdapter)

    runner = CodexTaskRunner(threading_mode="manual")
    fake_backend = _FakeBackend()
    runner._codex_backend = fake_backend

    async def _no_required_toolkits(*, context):
        del context
        return []

    monkeypatch.setattr(runner, "get_required_toolkits", _no_required_toolkits)

    room = _FakeRoom()
    caller = _FakeParticipant(name="caller", participant_id="caller-id")
    context = TaskContext(
        session=AgentSessionContext(system_role=None),
        room=room,
        caller=caller,
        on_behalf_of=None,
        toolkits=[],
    )

    result = await runner.ask(
        context=context,
        arguments={"prompt": "hello", "path": "/threads/task-1"},
    )

    assert result == "assistant response"
    assert fake_backend.open_thread_key == "/threads/task-1"
    assert fake_backend.closed_thread_key == "/threads/task-1"
    assert fake_backend.used_event_handler is True

    assert len(_FakeThreadAdapter.instances) == 1
    adapter = _FakeThreadAdapter.instances[0]
    assert adapter.path == "/threads/task-1"
    assert adapter.started
    assert adapter.appended
    assert adapter.stopped
    assert adapter.writes == [("hello", "caller")]
    assert [event["method"] for event in adapter.events] == [
        "item/started",
        "item/agentmessage/delta",
        "item/completed",
    ]


@pytest.mark.asyncio
async def test_task_runner_auto_threading_generates_path(monkeypatch) -> None:
    _FakeThreadAdapter.instances.clear()
    monkeypatch.setattr(task_runner_module, "CodexThreadAdapter", _FakeThreadAdapter)

    llm_adapter = _FakeLLMAdapter(generated_thread_name="Release Planning / Q1")
    runner = CodexTaskRunner(
        threading_mode="auto",
        thread_dir="/threads",
        llm_adapter=llm_adapter,
    )
    fake_backend = _FakeBackend()
    runner._codex_backend = fake_backend

    async def _no_required_toolkits(*, context):
        del context
        return []

    monkeypatch.setattr(runner, "get_required_toolkits", _no_required_toolkits)

    room = _FakeRoom()
    caller = _FakeParticipant(name="caller", participant_id="caller-id")
    context = TaskContext(
        session=AgentSessionContext(system_role=None),
        room=room,
        caller=caller,
        on_behalf_of=None,
        toolkits=[],
    )

    result = await runner.ask(
        context=context,
        arguments={"prompt": "Plan the Q1 release milestones"},
    )

    assert result == "assistant response"
    assert len(llm_adapter.calls) == 1

    thread_name_call = llm_adapter.calls[0]
    assert thread_name_call["output_schema"] is not None
    assert thread_name_call["output_schema"]["required"] == ["thread_name"]

    assert fake_backend.open_thread_key == "/threads/release-planning-q1.thread"
    assert fake_backend.closed_thread_key == "/threads/release-planning-q1.thread"
    assert fake_backend.used_event_handler is True

    assert len(_FakeThreadAdapter.instances) == 1
    adapter = _FakeThreadAdapter.instances[0]
    assert adapter.path == "/threads/release-planning-q1.thread"
    assert adapter.started
    assert adapter.appended
    assert adapter.stopped
    assert adapter.writes == [("Plan the Q1 release milestones", "caller")]
    assert [event["method"] for event in adapter.events] == [
        "item/started",
        "item/agentmessage/delta",
        "item/completed",
    ]


@pytest.mark.asyncio
async def test_task_runner_auto_threading_uses_custom_name_rules(monkeypatch) -> None:
    _FakeThreadAdapter.instances.clear()
    monkeypatch.setattr(task_runner_module, "CodexThreadAdapter", _FakeThreadAdapter)

    llm_adapter = _FakeLLMAdapter(generated_thread_name="custom thread")
    runner = CodexTaskRunner(
        threading_mode="auto",
        thread_dir="/threads",
        thread_name_rules=["pick a kebab-case name from this task prompt"],
        llm_adapter=llm_adapter,
    )
    fake_backend = _FakeBackend()
    runner._codex_backend = fake_backend

    async def _no_required_toolkits(*, context):
        del context
        return []

    monkeypatch.setattr(runner, "get_required_toolkits", _no_required_toolkits)

    room = _FakeRoom()
    caller = _FakeParticipant(name="caller", participant_id="caller-id")
    context = TaskContext(
        session=AgentSessionContext(system_role=None),
        room=room,
        caller=caller,
        on_behalf_of=None,
        toolkits=[],
    )

    result = await runner.ask(
        context=context,
        arguments={"prompt": "Do custom naming"},
    )

    assert result == "assistant response"
    assert len(llm_adapter.calls) == 1
    thread_name_call = llm_adapter.calls[0]
    assert thread_name_call["context"].instructions is not None
    assert (
        "pick a kebab-case name from this task prompt"
        in thread_name_call["context"].instructions
    )
