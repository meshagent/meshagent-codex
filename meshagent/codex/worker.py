import asyncio
import contextlib
import logging
from typing import Optional

from meshagent.agents import AgentChatContext, LLMAdapter
from meshagent.agents.worker import Worker
from meshagent.api import Requirement, RoomClient
from meshagent.api.specs.service import ContainerMountSpec
from meshagent.tools import Toolkit

from .app_server import _CodexAppServerBackend

logger = logging.getLogger("codex.worker")


class _CodexWorkerAdapter(LLMAdapter):
    def __init__(self, *, model: str):
        self._model = model

    def default_model(self) -> str:
        return self._model

    async def next(self, **kwargs):
        del kwargs
        raise RuntimeError("CodexWorker routes turns through codex app-server directly")


class CodexWorker(Worker):
    def __init__(
        self,
        *,
        queue: str,
        name=None,
        title: Optional[str] = None,
        description: Optional[str] = None,
        requires: Optional[list[Requirement]] = None,
        toolkits: Optional[list[Toolkit]] = None,
        rules: Optional[list[str]] = None,
        toolkit_name: Optional[str] = None,
        skill_dirs: Optional[list[str]] = None,
        supports_context: bool = True,
        annotations: Optional[list[str]] = None,
        model: str = "gpt-5.2-codex",
        command: Optional[str] = None,
        ws_url: Optional[str] = None,
        image: Optional[str] = None,
        mounts: Optional[ContainerMountSpec] = None,
        cwd: Optional[str] = None,
        approval_policy: Optional[str] = None,
        sandbox_policy: Optional[str] = None,
        app_server_env: Optional[dict[str, str]] = None,
        verbose: bool = False,
    ):
        self._model = model
        self._codex_skill_dirs = skill_dirs

        adapter_kwargs = {"model": model}
        if command is not None:
            adapter_kwargs["command"] = command
        if ws_url is not None:
            adapter_kwargs["ws_url"] = ws_url
        if image is not None:
            adapter_kwargs["image"] = image
        if mounts is not None:
            adapter_kwargs["mounts"] = mounts
        if cwd is not None:
            adapter_kwargs["cwd"] = cwd
        if approval_policy is not None:
            adapter_kwargs["approval_policy"] = approval_policy
        if sandbox_policy is not None:
            adapter_kwargs["sandbox_policy"] = sandbox_policy
        if app_server_env is not None:
            adapter_kwargs["env"] = app_server_env
        if verbose:
            adapter_kwargs["verbose_rpc"] = True

        self._codex_backend = _CodexAppServerBackend(**adapter_kwargs)

        super().__init__(
            queue=queue,
            name=name,
            title=title,
            description=description,
            requires=requires,
            llm_adapter=_CodexWorkerAdapter(model=model),
            toolkits=toolkits,
            rules=rules,
            toolkit_name=toolkit_name,
            skill_dirs=skill_dirs,
            supports_context=supports_context,
            annotations=annotations,
        )

    async def init_chat_context(self):
        return AgentChatContext(system_role=None)

    def default_model(self) -> str:
        return self._model

    async def preflight_start(self, *, room: RoomClient) -> None:
        try:
            await self._codex_backend.ensure_ready(room=room)
        except Exception:
            with contextlib.suppress(Exception):
                await self._codex_backend.close()
            raise

    async def process_message(
        self,
        *,
        chat_context: AgentChatContext,
        message: dict,
        toolkits: list[Toolkit],
    ):
        await self.append_message_context(message=message, chat_context=chat_context)

        prompt = None
        if len(chat_context.messages) > 0:
            content = chat_context.messages[-1].get("content")
            if isinstance(content, str):
                prompt = content

        if prompt is None or prompt.strip() == "":
            prompt = self.get_prompt_for_message(message=message)

        model = message.get("model", self.default_model())
        if not isinstance(model, str) or model.strip() == "":
            model = self.default_model()

        thread_key = chat_context.id
        developer_instructions = chat_context.get_system_instructions()

        try:
            await self._codex_backend.on_thread_open(
                thread_key=thread_key,
                room=self.room,
                context=chat_context,
                model=model,
                skill_dirs=self._codex_skill_dirs,
            )
            return await self._codex_backend.next(
                thread_key=thread_key,
                message=prompt,
                developer_instructions=developer_instructions,
                room=self.room,
                toolkits=toolkits,
                event_handler=None,
                model=model,
                on_behalf_of=None,
            )
        finally:
            try:
                await self._codex_backend.on_thread_close(thread_key=thread_key)
            except Exception as ex:
                logger.warning(
                    "unable to close codex worker thread '%s'",
                    thread_key,
                    exc_info=ex,
                )

    async def stop(self):
        try:
            with contextlib.suppress(Exception):
                await asyncio.wait_for(self._codex_backend.close(), timeout=10)
        finally:
            await super().stop()
