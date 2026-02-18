import base64
import asyncio
import contextlib
import io
import logging
import mimetypes
import posixpath
import re
import tarfile
from datetime import datetime, timezone
from typing import Literal, Optional

from meshagent.agents import AgentChatContext, LLMAdapter, TaskContext, TaskRunner
from meshagent.api import Requirement
from meshagent.api.specs.service import ContainerMountSpec
from meshagent.api.schema_util import merge, prompt_schema
from meshagent.tools import Toolkit

from .app_server import _CodexAppServerBackend
from .thread_adapter import CodexThreadAdapter

logger = logging.getLogger("codex.taskrunner")

ThreadingMode = Literal["auto", "manual", "none"]

DEFAULT_THREAD_NAME_RULES = [
    "generate a concise topic name for storing this task in a thread",
    "return only a thread_name value suitable for a file name",
    "thread_name should be 2-6 words, lowercase, and topic-focused",
    "do not include slashes or a .thread extension",
]


class CodexTaskRunner(TaskRunner):
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
        if threading_mode is None:
            resolved_threading_mode: ThreadingMode = (
                "manual" if allow_thread_selection else "none"
            )
        else:
            resolved_threading_mode = threading_mode

        if resolved_threading_mode == "auto" and llm_adapter is None:
            raise ValueError(
                "`llm_adapter` is required when `threading_mode` is 'auto'"
            )

        self.threading_mode = resolved_threading_mode
        self.thread_dir = thread_dir
        if thread_name_rules is not None and len(thread_name_rules) > 0:
            self.thread_name_rules = [*thread_name_rules]
        else:
            self.thread_name_rules = [*DEFAULT_THREAD_NAME_RULES]
        self._thread_name_adapter = llm_adapter

        if input_schema is None:
            input_schema = prompt_schema(description="use a prompt to generate content")

            if self.threading_mode == "manual":
                input_schema = merge(
                    schema=input_schema,
                    additional_properties={"path": {"type": ["string", "null"]}},
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

    async def init_chat_context(self):
        return AgentChatContext(system_role=None)

    def default_model(self) -> str:
        return self._model

    def _sanitize_thread_name(self, *, value: str) -> str:
        normalized = value.strip().lower()
        if normalized.endswith(".thread"):
            normalized = normalized[: -len(".thread")]

        normalized = re.sub(r"[^a-z0-9]+", "-", normalized)
        normalized = re.sub(r"-{2,}", "-", normalized).strip("-")
        if normalized == "":
            normalized = "thread"
        return normalized[:64].strip("-") or "thread"

    def _fallback_thread_name(self, *, prompt: str) -> str:
        del prompt
        return f"thread-{datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S')}"

    def _thread_path_for_name(self, *, thread_name: str) -> str:
        return posixpath.join(self.thread_dir, f"{thread_name}.thread")

    async def _generate_thread_path(
        self,
        *,
        context: TaskContext,
        prompt: str,
        model: str,
    ) -> str:
        if self._thread_name_adapter is None:
            raise RuntimeError(
                "auto threading mode requires a configured llm adapter for thread naming"
            )

        cloned_context = context.chat.copy()
        cloned_context.replace_rules(rules=self.thread_name_rules)
        cloned_context.append_user_message(prompt)

        generated_name = self._fallback_thread_name(prompt=prompt)
        try:
            response = await self._thread_name_adapter.next(
                context=cloned_context,
                room=context.room,
                model=model,
                on_behalf_of=context.on_behalf_of,
                toolkits=[],
                output_schema={
                    "type": "object",
                    "required": ["thread_name"],
                    "additionalProperties": False,
                    "properties": {
                        "thread_name": {
                            "type": "string",
                            "description": "2-6 word topic name for the task thread",
                        },
                    },
                },
            )
            if isinstance(response, dict):
                thread_name = response.get("thread_name")
                if isinstance(thread_name, str):
                    generated_name = self._sanitize_thread_name(value=thread_name)
        except Exception as ex:
            logger.warning(
                "unable to auto-generate thread name, using fallback",
                exc_info=ex,
            )

        return self._thread_path_for_name(thread_name=generated_name)

    async def resolve_thread_path(
        self,
        *,
        context: TaskContext,
        arguments: dict,
        prompt: str,
        model: str,
    ) -> str | None:
        if self.threading_mode == "none":
            return None

        if self.threading_mode == "manual":
            path = arguments.get("path")
            if path is None:
                return None
            if not isinstance(path, str):
                raise ValueError("`path` must be a string or null")

            selected_path = path.strip()
            if selected_path == "":
                return None
            return selected_path

        return await self._generate_thread_path(
            context=context,
            prompt=prompt,
            model=model,
        )

    def _selected_thread_path(self, *, arguments: dict) -> str | None:
        if self.threading_mode == "none":
            return None

        path = arguments.get("path")
        if path is None:
            return None
        if not isinstance(path, str):
            raise ValueError("`path` must be a string or null")

        selected_path = path.strip()
        if selected_path == "":
            return None
        return selected_path

    def create_thread_adapter(
        self,
        *,
        context: TaskContext,
        arguments: dict,
        attachment: Optional[bytes] = None,
    ) -> CodexThreadAdapter | None:
        del attachment
        selected_path = self._selected_thread_path(arguments=arguments)
        if selected_path is None:
            return None

        return CodexThreadAdapter(
            room=context.room,
            path=selected_path,
        )

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
