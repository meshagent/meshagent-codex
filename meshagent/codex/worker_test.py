from typing import Optional

import pytest

import meshagent.codex.worker as codex_worker_module
from meshagent.agents.context import AgentSessionContext
from meshagent.codex.worker import CodexWorker


class _FakeParticipant:
    def __init__(self, *, name: str, participant_id: str):
        self._name = name
        self.id = participant_id

    def get_attribute(self, key: str):
        if key == "name":
            return self._name
        return None


class _FakeElement:
    def __init__(self, *, tag_name: str, attributes: Optional[dict] = None):
        self.tag_name = tag_name
        self._attributes = dict(attributes or {})
        self._children: list["_FakeElement"] = []

    def get_attribute(self, key: str):
        return self._attributes.get(key)

    def append_child(self, tag_name: str, attributes: Optional[dict] = None):
        child = _FakeElement(tag_name=tag_name, attributes=attributes)
        self._children.append(child)
        return child

    def get_children(self):
        return self._children


class _FakeThreadRoot:
    def __init__(self):
        self._members = _FakeElement(tag_name="members")

    def get_children_by_tag_name(self, tag_name: str):
        if tag_name == "members":
            return [self._members]
        return []

    @property
    def members(self) -> _FakeElement:
        return self._members


class _FakeThreadDocument:
    def __init__(self):
        self.root = _FakeThreadRoot()


class _FakeThreadListDocument:
    def __init__(self):
        self.root = _FakeElement(tag_name="thread_list")

    def get_state(self) -> bytes:
        return b"thread-list-state"


class _FakeSync:
    def __init__(self):
        self.document = _FakeThreadListDocument()
        self.open_calls: list[dict] = []
        self.close_calls: list[str] = []
        self.sync_calls: list[dict] = []

    async def open(self, *, path: str, schema=None):
        self.open_calls.append({"path": path, "schema": schema})
        return self.document

    async def close(self, *, path: str):
        self.close_calls.append(path)

    async def sync(self, *, path: str, data: bytes):
        self.sync_calls.append({"path": path, "data": data})


class _FakeStorage:
    def __init__(self):
        self.exists_calls: list[str] = []

    async def exists(self, *, path: str) -> bool:
        self.exists_calls.append(path)
        return False


class _FakeRoom:
    def __init__(self):
        self.local_participant = _FakeParticipant(
            name="assistant",
            participant_id="assistant-id",
        )
        self.sync = _FakeSync()
        self.storage = _FakeStorage()


class _FakeThreadAdapter:
    instances: list["_FakeThreadAdapter"] = []

    def __init__(self, *, room, path: str):
        del room
        self.path = path
        self.thread = _FakeThreadDocument()
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
        self.writes.append((text, participant))

    def push(self, *, event: dict) -> None:
        self.events.append(event)


class _FakeCodexBackend:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.open_calls: list[dict] = []
        self.next_calls: list[dict] = []
        self.close_calls: list[str] = []

    async def ensure_ready(self, *, room):
        del room
        return None

    async def on_thread_open(
        self,
        *,
        thread_key: str,
        room,
        context,
        model: str,
        skill_dirs,
    ):
        del room, context, skill_dirs
        self.open_calls.append({"thread_key": thread_key, "model": model})

    async def next(
        self,
        *,
        thread_key: str,
        message: str,
        developer_instructions,
        room,
        toolkits,
        event_handler,
        model: str,
        on_behalf_of,
    ):
        del developer_instructions, room, toolkits, on_behalf_of
        self.next_calls.append(
            {"thread_key": thread_key, "message": message, "model": model}
        )
        if event_handler is not None:
            event_handler({"type": "codex.event", "summary": "done"})
        return "assistant response"

    async def on_thread_close(self, *, thread_key: str):
        self.close_calls.append(thread_key)

    async def close(self):
        return None


@pytest.mark.asyncio
async def test_codex_worker_manual_threading_writes_initial_message(
    monkeypatch,
) -> None:
    _FakeThreadAdapter.instances.clear()
    monkeypatch.setattr(
        codex_worker_module,
        "_CodexAppServerBackend",
        _FakeCodexBackend,
    )

    worker = CodexWorker(
        queue="tasks",
        model="gpt-5.2-codex",
        threading_mode="manual",
        thread_dir="/threads",
        initial_message_mode="code",
        initial_message_from="worker",
    )
    room = _FakeRoom()
    worker._room = room
    worker._threading_helper._thread_adapter_type = _FakeThreadAdapter

    result = await worker.process_message(
        chat_context=AgentSessionContext(system_role=None),
        message={"prompt": "Codex payload", "path": "/threads/manual.thread"},
        toolkits=[],
    )

    assert result == "assistant response"
    assert len(_FakeThreadAdapter.instances) == 1
    thread_adapter = _FakeThreadAdapter.instances[0]
    assert thread_adapter.path == "/threads/manual.thread"
    assert thread_adapter.started
    assert thread_adapter.stopped
    assert thread_adapter.appended
    assert thread_adapter.writes == [("```text\nCodex payload\n```", "worker")]
    assert thread_adapter.events == [{"type": "codex.event", "summary": "done"}]

    assert worker._codex_backend.open_calls == [
        {"thread_key": "/threads/manual.thread", "model": "gpt-5.2-codex"}
    ]
    assert worker._codex_backend.next_calls == [
        {
            "thread_key": "/threads/manual.thread",
            "message": "Codex payload",
            "model": "gpt-5.2-codex",
        }
    ]
    assert worker._codex_backend.close_calls == ["/threads/manual.thread"]
