import base64
import io
import logging
import mimetypes
import tarfile
from typing import Optional

from meshagent.agents import AgentChatContext, TaskContext, TaskRunner
from meshagent.api import Requirement
from meshagent.api.specs.service import ContainerMountSpec
from meshagent.api.schema_util import prompt_schema
from meshagent.tools import Toolkit

from .app_server import _CodexAppServerBackend

logger = logging.getLogger("codex.taskrunner")


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
    ):
        self._model = model

        if input_schema is None:
            input_schema = prompt_schema(description="use a prompt to generate content")
            input_schema["properties"]["model"] = {"type": ["string", "null"]}

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
        thread_key = context.chat.id

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
            return await self._codex_backend.next(
                thread_key=thread_key,
                message=turn_input,
                developer_instructions=rules,
                room=context.room,
                toolkits=combined_toolkits,
                event_handler=None,
                model=model,
                on_behalf_of=context.on_behalf_of,
            )
        finally:
            try:
                await self._codex_backend.on_thread_close(thread_key=thread_key)
            except Exception as ex:
                logger.warning(
                    "unable to close codex task thread '%s'",
                    thread_key,
                    exc_info=ex,
                )

    async def stop(self):
        try:
            await self._codex_backend.close()
        finally:
            await super().stop()
