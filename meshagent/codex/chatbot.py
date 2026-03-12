import asyncio
import json
import logging
import mimetypes
import uuid
from typing import Any, Callable, Optional
from datetime import datetime, timezone

from meshagent.agents import AgentSessionContext
from meshagent.agents.chat import (
    ChatBotBase,
    ChatThreadContext,
)
from meshagent.agents.thread_adapter import ThreadAdapter
from meshagent.api import MeshDocument, Requirement, RoomException
from meshagent.api import RemoteParticipant
from meshagent.api.specs.service import ContainerMountSpec
from meshagent.tools import Toolkit, make_toolkits

from .app_server import (
    DEFAULT_CODEX_CONTAINER_MOUNTS,
    CodexAppServerError,
    _CodexAppServerBackend,
)
from .thread_adapter import CodexThreadAdapter

logger = logging.getLogger("codex.chatbot")


class CodexChatBot(ChatBotBase):
    """
    ChatBot that uses Codex app-server as its LLM backend.

    It inherits ChatBot's message/control behavior, including:
    - `opened`
    - `chat`
    - `clear`
    - `cancel`
    - typing/listening signals and per-thread status attributes
    """

    def __init__(
        self,
        *,
        name=None,
        title=None,
        description=None,
        requires: Optional[list[Requirement]] = None,
        toolkits: Optional[list[Toolkit]] = None,
        rules: Optional[list[str]] = None,
        client_rules: Optional[dict[str, list[str]]] = None,
        auto_greet_message: Optional[str] = None,
        empty_state_title: Optional[str] = None,
        annotations: Optional[list[str]] = None,
        skill_dirs: Optional[list[str]] = None,
        thread_dir: Optional[str] = None,
        threading_mode: Optional[str] = None,
        thread_name_rules: Optional[list[str]] = None,
        model: str = "gpt-5.3-codex",
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
        self._mounts = mounts or DEFAULT_CODEX_CONTAINER_MOUNTS
        adapter_kwargs = {"model": model}
        if command is not None:
            adapter_kwargs["command"] = command
        if ws_url is not None:
            adapter_kwargs["ws_url"] = ws_url
        if image is not None:
            adapter_kwargs["image"] = image
        adapter_kwargs["mounts"] = self._mounts
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

        self._pending_approvals_lock = asyncio.Lock()
        self._pending_approvals: dict[str, asyncio.Future[str]] = {}
        self._pending_approval_keys_by_thread: dict[str, set[str]] = {}
        self._cancelling_threads: set[str] = set()
        adapter_kwargs["approval_request_handler"] = self._on_approval_requested

        self._codex_backend = _CodexAppServerBackend(**adapter_kwargs)
        super().__init__(
            name=name,
            title=title,
            description=description,
            requires=requires,
            toolkits=toolkits,
            rules=rules,
            client_rules=client_rules,
            auto_greet_message=auto_greet_message,
            empty_state_title=empty_state_title,
            annotations=annotations,
            skill_dirs=skill_dirs,
            thread_dir=thread_dir,
            threading_mode=threading_mode,
            thread_name_rules=thread_name_rules,
        )

    def default_model(self) -> str:
        return self._model

    def create_thread_adapter(self, *, path: str) -> ThreadAdapter:
        return CodexThreadAdapter(
            room=self.room,
            path=path,
            format_message=self.format_message,
        )

    async def create_thread_context(
        self,
        *,
        path: str,
        thread: MeshDocument,
        participants: list[RemoteParticipant],
        event_handler: Callable[[dict], None],
    ) -> ChatThreadContext:
        context = AgentSessionContext(system_role=None)
        context.append_rules(self._rules)
        return ChatThreadContext(
            path=path,
            thread=thread,
            participants=participants,
            event_handler=event_handler,
            session=context,
        )

    def _external_thread_id_from_thread(
        self, *, thread_context: ChatThreadContext
    ) -> Optional[str]:
        messages = thread_context.thread.root.get_children_by_tag_name("messages")
        if len(messages) == 0:
            return None

        value = messages[0].get_attribute("external_thread_id")
        if not isinstance(value, str):
            return None

        normalized = value.strip()
        if normalized == "":
            return None

        return normalized

    def _set_external_thread_id_on_thread(
        self,
        *,
        thread_context: ChatThreadContext,
        external_thread_id: str,
    ) -> None:
        normalized = external_thread_id.strip()
        if normalized == "":
            return

        messages = thread_context.thread.root.get_children_by_tag_name("messages")
        if len(messages) == 0:
            return

        messages[0].set_attribute("external_thread_id", normalized)

    def _clear_external_thread_id_on_thread(
        self, *, thread_context: ChatThreadContext
    ) -> None:
        messages = thread_context.thread.root.get_children_by_tag_name("messages")
        if len(messages) == 0:
            return

        messages[0].set_attribute("external_thread_id", "")

    async def _open_codex_thread(
        self,
        *,
        thread_context: ChatThreadContext,
        model: str,
    ) -> None:
        stored_thread_id = self._external_thread_id_from_thread(
            thread_context=thread_context
        )
        resolved_thread_id = await self._codex_backend.on_thread_open(
            thread_key=thread_context.path,
            room=self._room,
            context=thread_context.session,
            model=model,
            skill_dirs=self._skill_dirs,
            external_thread_id=stored_thread_id,
        )
        self._set_external_thread_id_on_thread(
            thread_context=thread_context,
            external_thread_id=resolved_thread_id,
        )

    def _status_event_details(
        self, *, event: dict
    ) -> tuple[Optional[str], Optional[str], Optional[str]]:
        event_type = event.get("type")
        if event_type not in ("agent.event", "codex.event"):
            return None, None, None

        kind = event.get("kind")
        if not isinstance(kind, str):
            kind = ""
        kind = kind.strip().lower()
        if kind not in (
            "exec",
            "tool",
            "collab",
            "web",
            "image",
            "diff",
            "approval",
        ):
            return None, None, None

        state = event.get("state")
        if not isinstance(state, str):
            state = ""
        state = state.strip().lower()

        key = None
        for candidate in (
            event.get("correlation_key"),
            event.get("event_key"),
            event.get("item_id"),
            event.get("name"),
            event.get("method"),
        ):
            if isinstance(candidate, str) and candidate.strip() != "":
                key = candidate.strip()
                break

        text = None
        for candidate in (
            event.get("headline"),
            event.get("summary"),
            event.get("name"),
            event.get("method"),
        ):
            if isinstance(candidate, str):
                normalized = candidate.strip()
                if normalized != "":
                    text = normalized
                    break

        return key, state, text

    def _update_thread_status_from_event(self, *, path: str, event: dict) -> None:
        if path in self._cancelling_threads:
            return

        key, state, text = self._status_event_details(event=event)
        if state is None:
            return

        is_active = state in ("queued", "in_progress", "running", "pending")
        if is_active:
            if text is None:
                return
            if key is not None:
                self._thread_status_keys[path] = key
            self._set_thread_status_nowait(path=path, status=text)
            return

        # Keep a cancellable in-progress status visible for the whole turn.
        # Per-item completion events are noisy and should not clear thread status.
        fallback_status = "Thinking"

        if key is not None:
            tracked = self._thread_status_keys.get(path)
            if tracked is not None and tracked == key:
                self._thread_status_keys.pop(path, None)
                self._set_thread_status_nowait(path=path, status=fallback_status)
            return

        if state in ("completed", "failed", "cancelled"):
            self._set_thread_status_nowait(path=path, status=fallback_status)

    def _approval_key(self, *, thread_key: str, approval_id: str) -> str:
        return f"{thread_key}:{approval_id}"

    def _approval_id(self, *, params: dict) -> str:
        for key in ("item_id", "itemId", "approval_id", "approvalId"):
            value = params.get(key)
            if isinstance(value, str) and value.strip() != "":
                return value.strip()

        item = params.get("item")
        if isinstance(item, dict):
            for key in ("id", "item_id", "itemId"):
                value = item.get(key)
                if isinstance(value, str) and value.strip() != "":
                    return value.strip()

        return str(uuid.uuid4())

    def _first_nested_text(self, *, value: Any, keys: tuple[str, ...]) -> str:
        key_set = {key.lower() for key in keys}

        if isinstance(value, dict):
            for key, nested in value.items():
                if key.lower() in key_set and isinstance(nested, str):
                    text = nested.strip()
                    if text != "":
                        return text

            for nested in value.values():
                text = self._first_nested_text(value=nested, keys=keys)
                if text != "":
                    return text

        elif isinstance(value, list):
            for nested in value:
                text = self._first_nested_text(value=nested, keys=keys)
                if text != "":
                    return text

        return ""

    def _approval_details(self, *, method: str, params: dict) -> list[str]:
        del method

        command = self._first_nested_text(
            value=params,
            keys=("command", "cmd", "shell_command"),
        )
        reason = self._first_nested_text(
            value=params,
            keys=("reason", "message", "prompt", "description", "explanation"),
        )
        tool = self._first_nested_text(
            value=params,
            keys=("tool", "tool_name"),
        )
        action = self._first_nested_text(
            value=params,
            keys=("action", "operation"),
        )
        target = self._first_nested_text(
            value=params,
            keys=("path", "file", "url"),
        )

        details: list[str] = []
        seen: set[str] = set()

        def append_detail(text: str) -> None:
            normalized = " ".join(text.strip().lower().split())
            if normalized == "" or normalized in seen:
                return
            seen.add(normalized)
            details.append(text.strip())

        if command != "":
            append_detail(command)
        if reason != "":
            append_detail(reason)
        if tool != "":
            append_detail(f"Tool: {tool}")
        if action != "" and action.lower() not in (
            "requestapproval",
            "request_approval",
            "approval",
        ):
            append_detail(f"Action: {action}")
        if target != "":
            append_detail(f"Target: {target}")

        if len(details) == 0:
            append_detail("Review the action and approve to continue.")

        return details

    def _approval_event_payload(
        self,
        *,
        method: str,
        params: dict,
        approval_id: str,
        state: str,
        summary: str,
        headline: str,
    ) -> dict:
        details = self._approval_details(
            method=method,
            params=params,
        )

        return {
            "type": "agent.event",
            "source": "codex",
            "name": "approval.requested" if state == "queued" else "approval.completed",
            "kind": "approval",
            "state": state,
            "method": method,
            "item_id": approval_id,
            "correlation_key": f"approval:{approval_id}",
            "summary": summary,
            "headline": headline,
            "details": details,
            "data": json.dumps(params, ensure_ascii=False, default=str),
        }

    async def _register_pending_approval(
        self, *, thread_key: str, approval_id: str
    ) -> asyncio.Future[str]:
        loop = asyncio.get_running_loop()
        future = loop.create_future()
        key = self._approval_key(thread_key=thread_key, approval_id=approval_id)

        async with self._pending_approvals_lock:
            self._pending_approvals[key] = future
            keys = self._pending_approval_keys_by_thread.get(thread_key)
            if keys is None:
                keys = set()
                self._pending_approval_keys_by_thread[thread_key] = keys
            keys.add(key)

        return future

    async def _remove_pending_approval(
        self, *, thread_key: str, approval_id: str
    ) -> None:
        key = self._approval_key(thread_key=thread_key, approval_id=approval_id)
        async with self._pending_approvals_lock:
            self._pending_approvals.pop(key, None)
            keys = self._pending_approval_keys_by_thread.get(thread_key)
            if keys is None:
                return
            keys.discard(key)
            if len(keys) == 0:
                self._pending_approval_keys_by_thread.pop(thread_key, None)

    async def _resolve_pending_approval(
        self,
        *,
        thread_key: str,
        approval_id: Optional[str],
        decision: str,
    ) -> bool:
        pending_key = None
        pending_future = None

        async with self._pending_approvals_lock:
            if approval_id is not None and approval_id.strip() != "":
                key = self._approval_key(
                    thread_key=thread_key,
                    approval_id=approval_id.strip(),
                )
                future = self._pending_approvals.get(key)
                if future is not None:
                    pending_key = key
                    pending_future = future
            else:
                keys = list(
                    self._pending_approval_keys_by_thread.get(thread_key, set())
                )
                if len(keys) == 1:
                    key = keys[0]
                    future = self._pending_approvals.get(key)
                    if future is not None:
                        pending_key = key
                        pending_future = future

        if pending_future is None or pending_key is None:
            return False

        if not pending_future.done():
            pending_future.set_result(decision)

        return True

    async def _cancel_all_pending_approvals(self, *, thread_key: str) -> None:
        async with self._pending_approvals_lock:
            keys = list(self._pending_approval_keys_by_thread.get(thread_key, set()))
            futures = [self._pending_approvals.get(key) for key in keys]

        for future in futures:
            if future is not None and not future.done():
                future.set_result("cancel")

    async def _on_approval_requested(
        self,
        *,
        thread_key: str,
        method: str,
        params: dict,
    ) -> str:
        thread_context = self._thread_contexts.get(thread_key)
        if thread_context is None:
            logger.warning(
                "received codex approval request for unopened thread '%s'",
                thread_key,
            )
            return "accept"

        approval_id = self._approval_id(params=params)
        pending = await self._register_pending_approval(
            thread_key=thread_key,
            approval_id=approval_id,
        )

        try:
            thread_context.emit(
                self._approval_event_payload(
                    method=method,
                    params=params,
                    approval_id=approval_id,
                    state="queued",
                    summary="Approval required",
                    headline="Approval Required",
                )
            )
        except Exception as ex:
            logger.warning(
                "unable to emit pending approval event for thread '%s'",
                thread_key,
                exc_info=ex,
            )

        decision = "cancel"
        try:
            decision = await pending
            return decision
        finally:
            await self._remove_pending_approval(
                thread_key=thread_key,
                approval_id=approval_id,
            )

            if decision in ("accept", "acceptForSession"):
                state = "completed"
                summary = "Approved"
                headline = "Approval Granted"
            elif decision == "cancel":
                state = "cancelled"
                summary = "Cancelled"
                headline = "Approval Cancelled"
            else:
                state = "failed"
                summary = "Rejected"
                headline = "Approval Rejected"

            try:
                thread_context.emit(
                    self._approval_event_payload(
                        method=method,
                        params=params,
                        approval_id=approval_id,
                        state=state,
                        summary=summary,
                        headline=headline,
                    )
                )
            except Exception as ex:
                logger.warning(
                    "unable to emit completion approval event for thread '%s'",
                    thread_key,
                    exc_info=ex,
                )

    async def on_thread_open(self, *, thread_context: ChatThreadContext):
        self._cancelling_threads.discard(thread_context.path)
        await self.clear_thread_status(path=thread_context.path)
        await self._open_codex_thread(
            thread_context=thread_context,
            model=self._model,
        )

    async def on_thread_clear(self, *, thread_context: ChatThreadContext):
        await self._cancel_all_pending_approvals(thread_key=thread_context.path)
        await self._codex_backend.on_thread_clear(
            thread_key=thread_context.path,
            context=thread_context.session,
        )
        self._cancelling_threads.discard(thread_context.path)
        self._clear_external_thread_id_on_thread(thread_context=thread_context)
        await self.clear_thread_status(path=thread_context.path)

    async def on_thread_cancel(self, *, thread_context: ChatThreadContext):
        should_show_cancelling = self._codex_backend.has_active_turn(
            thread_key=thread_context.path
        )
        already_showing_cancelling = should_show_cancelling
        if should_show_cancelling:
            self._cancelling_threads.add(thread_context.path)
            self._thread_status_keys.pop(thread_context.path, None)
            await self.set_thread_status(
                path=thread_context.path,
                status="Cancelling",
                mode="busy",
            )
        await self._cancel_all_pending_approvals(thread_key=thread_context.path)
        await self._codex_backend.on_thread_cancel(thread_key=thread_context.path)

        should_show_cancelling = self._codex_backend.has_active_turn(
            thread_key=thread_context.path
        )
        if should_show_cancelling:
            self._cancelling_threads.add(thread_context.path)
            if not already_showing_cancelling:
                self._thread_status_keys.pop(thread_context.path, None)
                await self.set_thread_status(
                    path=thread_context.path,
                    status="Cancelling",
                    mode="busy",
                )
            return

        self._cancelling_threads.discard(thread_context.path)
        await self.clear_thread_status(path=thread_context.path)

    async def on_thread_close(self, *, thread_context: ChatThreadContext):
        await self._cancel_all_pending_approvals(thread_key=thread_context.path)
        await self._codex_backend.on_thread_close(thread_key=thread_context.path)
        self._cancelling_threads.discard(thread_context.path)
        await self.clear_thread_status(path=thread_context.path)

    def processing_thread_status_mode(
        self, *, path: str, thread_context: Optional[ChatThreadContext]
    ) -> str:
        del path
        del thread_context
        return "steerable"

    async def cancel_thread_task(
        self, *, path: str, thread_context: Optional[ChatThreadContext]
    ) -> None:
        del path
        del thread_context

    async def on_approved(
        self,
        *,
        thread_context: ChatThreadContext,
        from_participant: RemoteParticipant,
        message: dict,
    ):
        del from_participant
        decision = message.get("decision")
        if not isinstance(decision, str) or decision.strip() == "":
            decision = "accept"
        approval_id = message.get("approval_id")
        if not isinstance(approval_id, str):
            approval_id = None

        resolved = await self._resolve_pending_approval(
            thread_key=thread_context.path,
            approval_id=approval_id,
            decision=decision,
        )
        if not resolved:
            logger.warning(
                "received approval response for unknown request on thread '%s'",
                thread_context.path,
            )

    async def on_rejected(
        self,
        *,
        thread_context: ChatThreadContext,
        from_participant: RemoteParticipant,
        message: dict,
    ):
        del from_participant
        decision = message.get("decision")
        if not isinstance(decision, str) or decision.strip() == "":
            decision = "decline"
        approval_id = message.get("approval_id")
        if not isinstance(approval_id, str):
            approval_id = None

        resolved = await self._resolve_pending_approval(
            thread_key=thread_context.path,
            approval_id=approval_id,
            decision=decision,
        )
        if not resolved:
            logger.warning(
                "received rejection response for unknown request on thread '%s'",
                thread_context.path,
            )

    def _append_user_message_to_context(
        self,
        *,
        thread_context: ChatThreadContext,
        from_participant: RemoteParticipant,
        text: str,
    ) -> None:
        iso_timestamp = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
        formatted_message = self.format_message(
            user_name=from_participant.get_attribute("name"),
            message=text,
            iso_timestamp=iso_timestamp,
        )
        thread_context.session.append_user_message(message=formatted_message)

    def _normalize_attachment_path(self, *, path: str) -> Optional[str]:
        cleaned = path.strip().replace("\\", "/")
        if cleaned in ("", ".", "/"):
            return None
        normalized = cleaned.strip("/")
        if normalized == "":
            return None
        parts = [part for part in normalized.split("/") if part != ""]
        if len(parts) == 0:
            return None
        if any(part in (".", "..") for part in parts):
            return None
        return "/".join(parts)

    def _normalize_room_subpath(self, *, subpath: Optional[str]) -> Optional[str]:
        if subpath is None:
            return ""
        cleaned = subpath.strip().replace("\\", "/")
        if cleaned in ("", ".", "/"):
            return ""
        normalized = cleaned.strip("/")
        if normalized == "":
            return ""
        parts = [part for part in normalized.split("/") if part != ""]
        if len(parts) == 0:
            return ""
        if any(part in (".", "..") for part in parts):
            return None
        return "/".join(parts)

    def _normalize_container_mount_path(self, *, path: str) -> Optional[str]:
        cleaned = path.strip().replace("\\", "/")
        if cleaned == "":
            return None
        normalized = cleaned if cleaned.startswith("/") else f"/{cleaned}"
        parts = [part for part in normalized.split("/") if part != ""]
        if len(parts) == 0:
            return "/"
        if any(part in (".", "..") for part in parts):
            return None
        return f"/{'/'.join(parts)}"

    def _resolve_attachment_container_path(self, *, path: str) -> Optional[str]:
        room_path = self._normalize_attachment_path(path=path)
        if room_path is None:
            return None

        room_mounts = self._mounts.room
        if room_mounts is None:
            return None

        best_match_path = None
        best_match_subpath_length = -1

        for room_mount in room_mounts:
            mount_path = self._normalize_container_mount_path(path=room_mount.path)
            if mount_path is None:
                continue

            mount_subpath = self._normalize_room_subpath(subpath=room_mount.subpath)
            if mount_subpath is None:
                continue

            relative_path = None
            if mount_subpath == "":
                relative_path = room_path
            elif room_path == mount_subpath:
                relative_path = ""
            elif room_path.startswith(f"{mount_subpath}/"):
                relative_path = room_path[len(mount_subpath) + 1 :]

            if relative_path is None:
                continue

            if relative_path == "":
                resolved_path = mount_path
            elif mount_path == "/":
                resolved_path = f"/{relative_path}"
            else:
                resolved_path = f"{mount_path}/{relative_path}"

            subpath_length = len(mount_subpath)
            if subpath_length > best_match_subpath_length:
                best_match_subpath_length = subpath_length
                best_match_path = resolved_path

        return best_match_path

    async def _message_to_turn_input(self, *, message: dict) -> list[dict]:
        text = message.get("text")
        if not isinstance(text, str):
            text = ""
        turn_input = []
        if text.strip() != "":
            turn_input.append({"type": "text", "text": text})

        attachments = message.get("attachments", [])
        if not isinstance(attachments, list):
            attachments = []

        for attachment in attachments:
            if not isinstance(attachment, dict):
                continue

            path = attachment.get("path")
            if not isinstance(path, str) or path.strip() == "":
                continue

            mounted_path = self._resolve_attachment_container_path(path=path)
            path_for_text = mounted_path if mounted_path is not None else path.strip()

            hinted_mime = None
            for candidate in (
                attachment.get("mime_type"),
                attachment.get("content_type"),
                mimetypes.guess_type(path)[0],
            ):
                if isinstance(candidate, str):
                    normalized = candidate.split(";")[0].strip().lower()
                    if normalized == "image/jpg":
                        normalized = "image/jpeg"
                    hinted_mime = normalized
                    break

            if (
                hinted_mime is not None
                and hinted_mime.startswith("image/")
                and mounted_path is not None
            ):
                turn_input.append({"type": "localImage", "path": mounted_path})
            else:
                turn_input.append(
                    {"type": "text", "text": f"file attached {path_for_text}"}
                )

        if len(turn_input) == 0:
            raise RoomException("steering message cannot be empty")

        return turn_input

    async def on_thread_steer(
        self,
        *,
        thread_context: ChatThreadContext,
        from_participant: RemoteParticipant,
        message: dict,
    ) -> None:
        turn_input = await self._message_to_turn_input(
            message=message,
        )
        try:
            await self._codex_backend.steer(
                thread_key=thread_context.path,
                message=turn_input,
            )
        except CodexAppServerError as ex:
            if "no active turn" not in str(ex).lower():
                raise

            logger.info(
                "codex thread '%s' has no active turn to steer; handling as chat",
                thread_context.path,
            )
            await self.on_chat_received(
                thread_context=thread_context,
                from_participant=from_participant,
                message=message,
            )
            return

        text = message.get("text")
        if not isinstance(text, str):
            text = ""
        self._append_user_message_to_context(
            thread_context=thread_context,
            from_participant=from_participant,
            text=text,
        )

    async def on_chat_received(
        self,
        *,
        thread_context: ChatThreadContext,
        from_participant: RemoteParticipant,
        message: dict,
    ) -> Optional[str]:
        rules = await self.get_rules(
            thread_context=thread_context,
            participant=from_participant,
        )
        thread_context.session.replace_rules(rules)

        text = message["text"]
        turn_input = await self._message_to_turn_input(
            message=message,
        )
        self._append_user_message_to_context(
            thread_context=thread_context,
            from_participant=from_participant,
            text=text,
        )

        model = message.get("model")
        if not isinstance(model, str) or model.strip() == "":
            model = self.default_model()

        thread_toolkits = await self.get_thread_toolkits(
            thread_context=thread_context,
            participant=from_participant,
        )
        thread_tool_providers = self.get_toolkit_builders()

        message_toolkits = [*thread_toolkits]
        message_tools = message.get("tools")
        if message_tools is not None and len(message_tools) > 0:
            message_toolkits.extend(
                await make_toolkits(
                    room=self.room,
                    model=model,
                    providers=thread_tool_providers,
                    tools=message_tools,
                )
            )

        await self._open_codex_thread(
            thread_context=thread_context,
            model=model,
        )
        self._cancelling_threads.discard(thread_context.path)

        try:
            return await self._codex_backend.next(
                thread_key=thread_context.path,
                message=turn_input,
                developer_instructions=rules,
                room=self._room,
                toolkits=message_toolkits,
                event_handler=thread_context.emit,
                model=model,
                on_behalf_of=from_participant,
            )
        finally:
            self._cancelling_threads.discard(thread_context.path)

    async def stop(self):
        await super().stop()
        await self._codex_backend.close()
