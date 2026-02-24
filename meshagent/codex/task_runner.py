import base64
import asyncio
import contextlib
import io
import logging
import mimetypes
import tarfile
from typing import Optional

from meshagent.agents import (
    AgentSessionContext,
    LLMAdapter,
    TaskContext,
    ThreadedTaskRunner,
)
from meshagent.agents.threaded_task_runner import ThreadingMode
from meshagent.api import Requirement
from meshagent.api.specs.service import ContainerMountSpec
from meshagent.api.schema_util import prompt_schema
from meshagent.tools import Toolkit

from .app_server import _CodexAppServerBackend
from .thread_adapter import CodexThreadAdapter

logger = logging.getLogger("codex.taskrunner")


class CodexTaskRunner(ThreadedTaskRunner):
    """
    A TaskRunner that uses Codex app-server.

    Structured outputs are intentionally disabled for this runner because
    codex app-server does not support JSON schema constrained outputs.
    """

    def __init__(
        self,
        *,
        name=None,
        title: Optional[str] = None,
        description: Optional[str] = None,
        requires: Optional[list[Requirement]] = None,
        toolkits: Optional[list[Toolkit]] = None,
        supports_tools: bool = False,
        allow_thread_selection: bool = False,
        threading_mode: Optional[ThreadingMode] = None,
        thread_dir: str = ".threads",
        thread_name_rules: Optional[list[str]] = None,
        llm_adapter: Optional[LLMAdapter] = None,
        input_schema: Optional[dict] = None,
        rules: Optional[list[str]] = None,
        annotations: Optional[list[str]] = None,
        client_rules: Optional[dict[str, list[str]]] = None,
        skill_dirs: Optional[list[str]] = None,
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
        resolved_threading_mode = self.resolve_threading_mode(
            threading_mode=threading_mode,
            input_path=allow_thread_selection,
        )

        if input_schema is None:
            input_schema = prompt_schema(description="use a prompt to generate content")
            input_schema = self.with_manual_thread_path_schema(
                input_schema=input_schema,
                threading_mode=resolved_threading_mode,
            )

        static_toolkits = list(toolkits or [])
        super().__init__(
            name=name,
            title=title,
            description=description,
            input_schema=input_schema,
            output_schema=None,
            requires=requires,
            supports_tools=supports_tools,
            annotations=annotations,
            toolkits=static_toolkits,
            input_path=allow_thread_selection,
            threading_mode=threading_mode,
            thread_dir=thread_dir,
            thread_name_rules=thread_name_rules,
            thread_name_adapter=llm_adapter,
            thread_adapter_type=CodexThreadAdapter,
        )

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
        self._extra_rules = rules or []
        self._client_rules = client_rules
        self._skill_dirs = skill_dirs
        self._toolkits = static_toolkits

    @property
    def output_schema(self):
        # Codex app-server does not support structured JSON-schema output.
        return None

    def to_json(self) -> dict:
        data = super().to_json()
        data.pop("output_schema", None)
        return data

    async def init_session(self):
        return AgentSessionContext(system_role=None)

    def default_model(self) -> str:
        return self._model

    async def get_rules(self, *, context: TaskContext) -> list[str]:
        rules = [*self._extra_rules]

        participant = context.caller
        client = (
            participant.get_attribute("client") if participant is not None else None
        )

        if self._client_rules is not None and client is not None:
            client_rules = self._client_rules.get(client)
            if client_rules is not None:
                rules.extend(client_rules)

        return rules

    async def ask(
        self,
        *,
        context: TaskContext,
        arguments: dict,
        attachment: Optional[bytes] = None,
    ):
        prompt = arguments.get("prompt")
        if prompt is None:
            raise ValueError("`prompt` is required")

        model = arguments.get("model", self.default_model())
        selected_path = await self.resolve_thread_path(
            context=context,
            arguments=arguments,
            prompt=prompt,
            model=model,
        )
        adapter_arguments = arguments
        if selected_path is not None:
            adapter_arguments = {
                **arguments,
                "path": selected_path,
            }

        selected_path = self._selected_thread_path(arguments=adapter_arguments)
        thread_key = selected_path if selected_path is not None else context.chat.id

        thread_adapter = self.create_thread_adapter(
            context=context,
            arguments=adapter_arguments,
            attachment=attachment,
        )
        if thread_adapter is not None:
            await thread_adapter.start()
            thread_adapter.append_messages(context=context.chat)
            thread_adapter.write_text_message(
                text=prompt,
                participant=context.caller if context.caller is not None else "user",
            )

        def push(event: dict) -> None:
            if thread_adapter is not None:
                thread_adapter.push(event=event)

        rules = await self.get_rules(context=context)
        context.chat.append_rules(rules)
        context.chat.append_user_message(prompt)

        turn_input: list[dict] = [{"type": "text", "text": prompt}]
        non_image_file_notes: list[str] = []

        if attachment is not None:
            buf = io.BytesIO(attachment)
            with tarfile.open(fileobj=buf, mode="r:*") as tar:
                for member in tar.getmembers():
                    if not member.isfile():
                        continue

                    extracted = tar.extractfile(member)
                    if extracted is None:
                        continue

                    content = extracted.read()
                    mime_type, _encoding = mimetypes.guess_type(member.name)
                    normalized_mime = mime_type or "application/octet-stream"

                    if normalized_mime.startswith("image/"):
                        turn_input.append(
                            {
                                "type": "image",
                                "url": (
                                    f"data:{normalized_mime};base64,"
                                    f"{base64.b64encode(content).decode('ascii')}"
                                ),
                            }
                        )
                    else:
                        non_image_file_notes.append(
                            f"{member.name} ({normalized_mime})"
                        )

        if len(non_image_file_notes) > 0:
            notes = "\n".join(f"- {note}" for note in non_image_file_notes)
            turn_input.append(
                {
                    "type": "text",
                    "text": (
                        f"The caller also attached files that are not images:\n{notes}"
                    ),
                }
            )

        combined_toolkits: list[Toolkit] = [
            *self._toolkits,
            *context.toolkits,
            *await self.get_required_toolkits(context=context),
        ]

        try:
            await self._codex_backend.on_thread_open(
                thread_key=thread_key,
                room=context.room,
                context=context.chat,
                model=model,
                skill_dirs=self._skill_dirs,
            )
            response = await self._codex_backend.next(
                thread_key=thread_key,
                message=turn_input,
                developer_instructions=rules,
                room=context.room,
                toolkits=combined_toolkits,
                event_handler=push if thread_adapter is not None else None,
                model=model,
                on_behalf_of=context.on_behalf_of,
            )

            return response
        finally:
            try:
                await self._codex_backend.on_thread_close(thread_key=thread_key)
            except Exception as ex:
                logger.warning(
                    "unable to close codex task thread '%s'",
                    thread_key,
                    exc_info=ex,
                )
            if thread_adapter is not None:
                with contextlib.suppress(Exception):
                    await thread_adapter.stop()

    async def run(
        self,
        *,
        room,
        arguments: dict,
        attachment: Optional[bytes] = None,
        caller=None,
    ):
        try:
            return await super().run(
                room=room,
                arguments=arguments,
                attachment=attachment,
                caller=caller,
            )
        finally:
            with contextlib.suppress(Exception):
                await asyncio.wait_for(self._codex_backend.close(), timeout=10)

    async def stop(self):
        try:
            await asyncio.wait_for(self._codex_backend.close(), timeout=10)
        finally:
            await super().stop()
