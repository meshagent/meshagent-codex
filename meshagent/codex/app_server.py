import asyncio
import contextlib
import json
import logging
import os
import shlex
import sys
from collections import deque
from pathlib import Path
from typing import Any, Awaitable, Callable, Optional

import aiohttp
from meshagent.agents import ToolResponseAdapter
from meshagent.agents.agent import AgentChatContext
from meshagent.api import RoomClient, RoomException, RemoteParticipant
from meshagent.api.specs.service import ContainerMountSpec, RoomStorageMountSpec
from meshagent.tools import Toolkit

from .version import __version__

logger = logging.getLogger("codex.app_server")

DEFAULT_CODEX_CONTAINER_MOUNTS = ContainerMountSpec(
    room=[RoomStorageMountSpec(path="/data")]
)


class CodexAppServerError(RoomException):
    pass


def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default

    value = value.strip().lower()
    if value in ("1", "true", "yes", "on"):
        return True
    if value in ("0", "false", "no", "off"):
        return False
    return default


def _to_text(value: Any) -> str:
    if value is None:
        return ""

    if isinstance(value, str):
        return value

    if isinstance(value, list):
        parts: list[str] = []
        for item in value:
            if isinstance(item, str):
                parts.append(item)
                continue

            if not isinstance(item, dict):
                continue

            if isinstance(item.get("text"), str):
                parts.append(item["text"])
            elif isinstance(item.get("value"), str):
                parts.append(item["value"])

        return "".join(parts)

    if isinstance(value, dict):
        if isinstance(value.get("text"), str):
            return value["text"]
        if isinstance(value.get("value"), str):
            return value["value"]
        if isinstance(value.get("content"), (str, list, dict)):
            return _to_text(value.get("content"))

    return str(value)


def _get_nested_text(item: dict) -> str:
    if not isinstance(item, dict):
        return ""

    for key in ("text", "content", "message"):
        if key in item:
            text = _to_text(item.get(key))
            if text:
                return text

    return ""


def _item_type(item: dict) -> str:
    value = item.get("type")
    if not isinstance(value, str):
        return ""
    return value.lower()


def _is_agent_message(item: dict) -> bool:
    type_name = _item_type(item)
    return type_name in ("agentmessage", "agent_message")


def _get_nested_id(params: dict, *, singular: str, nested_key: str) -> Optional[str]:
    if not isinstance(params, dict):
        return None

    camel = f"{singular}Id"
    snake = f"{singular}_id"

    for key in (camel, snake, singular):
        value = params.get(key)
        if isinstance(value, str):
            return value
        if isinstance(value, dict):
            nested = value.get("id")
            if isinstance(nested, str):
                return nested

    item = params.get("item")
    if isinstance(item, dict):
        for key in (camel, snake):
            value = item.get(key)
            if isinstance(value, str):
                return value

    nested = params.get(nested_key)
    if isinstance(nested, dict):
        value = nested.get("id")
        if isinstance(value, str):
            return value

    return None


class _CodexJsonRpcSession:
    def __init__(
        self,
        *,
        command: Optional[str] = None,
        ws_url: Optional[str] = None,
        image: Optional[str] = None,
        cwd: Optional[str] = None,
        env: Optional[dict[str, str]] = None,
        mounts: Optional[ContainerMountSpec] = DEFAULT_CODEX_CONTAINER_MOUNTS,
        forward_stdout: bool = False,
        forward_stderr: bool = True,
        request_timeout_s: float = 300.0,
        server_request_handler: Optional[
            Callable[[str, dict], Awaitable[Optional[dict]]]
        ] = None,
    ):
        if command is None and ws_url is None and image is None:
            raise CodexAppServerError(
                "codex transport is not configured (missing command, ws_url, and image)"
            )

        self._command = command
        self._ws_url = ws_url
        self._image = image
        self._cwd = cwd
        self._env = env
        self._mounts = mounts
        self._forward_stdout = forward_stdout
        self._forward_stderr = forward_stderr
        self._request_timeout_s = request_timeout_s
        self._server_request_handler = server_request_handler

        self._process: Optional[asyncio.subprocess.Process] = None
        self._client_session: Optional[aiohttp.ClientSession] = None
        self._websocket: Optional[aiohttp.ClientWebSocketResponse] = None
        self._container_id: Optional[str] = None
        self._container_exec = None
        self._room: Optional[RoomClient] = None
        self._reader_task: Optional[asyncio.Task] = None
        self._stderr_task: Optional[asyncio.Task] = None

        self._notifications: asyncio.Queue[dict] = asyncio.Queue()
        self._pending: dict[int, asyncio.Future] = {}
        self._next_id = 1

        self._started = False
        self._start_lock = asyncio.Lock()
        self._write_lock = asyncio.Lock()

    def _mark_transport_failure(self, *, message: str) -> None:
        logger.error(message)
        self._started = False

        for future in self._pending.values():
            if not future.done():
                future.set_exception(CodexAppServerError(message))

        self._notifications.put_nowait(
            {
                "method": "__session_error__",
                "params": {"error": message},
            }
        )

    def _cleanup_dead_transport_handles(self) -> None:
        if self._process is not None and self._process.returncode is not None:
            self._process = None

        if self._websocket is not None and self._websocket.closed:
            self._websocket = None

        if (
            self._container_exec is not None
            and hasattr(self._container_exec, "result")
            and self._container_exec.result.done()
        ):
            self._container_exec = None

        if self._client_session is not None and self._client_session.closed:
            self._client_session = None

    def _container_transport_enabled(self) -> bool:
        return self._image is not None and self._ws_url is None

    def set_room(self, *, room: Optional[RoomClient]) -> None:
        self._room = room

    async def start(self, *, room: Optional[RoomClient] = None) -> None:
        async with self._start_lock:
            if self._started:
                return

            if room is not None:
                self._room = room

            self._cleanup_dead_transport_handles()
            login_api_key: Optional[str] = None
            key_candidate = (self._env or {}).get("OPENAI_API_KEY")
            if isinstance(key_candidate, str) and key_candidate.strip() != "":
                login_api_key = key_candidate.strip()

            if self._ws_url is not None:
                self._client_session = aiohttp.ClientSession()
                try:
                    self._websocket = await self._client_session.ws_connect(
                        self._ws_url
                    )
                except Exception as exc:
                    await self._client_session.close()
                    self._client_session = None
                    raise CodexAppServerError(
                        f"unable to connect to codex app-server websocket: {self._ws_url}"
                    ) from exc

                self._reader_task = asyncio.create_task(self._read_websocket())
            elif self._container_transport_enabled():
                if self._room is None:
                    raise CodexAppServerError(
                        "room is required for codex container transport"
                    )

                if self._command is None:
                    raise CodexAppServerError("codex command was empty")
                if self._command.strip() == "":
                    raise CodexAppServerError("codex command was empty")

                running = False
                if self._container_id is not None:
                    with contextlib.suppress(Exception):
                        for container in await self._room.containers.list():
                            if container.id == self._container_id:
                                running = True
                                break

                    if not running:
                        self._container_id = None

                if self._container_id is None:
                    container_env = dict(self._env or {})

                    if (
                        not isinstance(container_env.get("OPENAI_BASE_URL"), str)
                        or container_env.get("OPENAI_BASE_URL", "").strip() == ""
                    ):
                        protocol_url = getattr(self._room.protocol, "url", None)
                        if isinstance(protocol_url, str) and protocol_url.strip() != "":
                            room_url = protocol_url.strip().rstrip("/")
                            if room_url.startswith("wss:"):
                                room_url = "https:" + room_url.removeprefix("wss:")
                            elif room_url.startswith("ws:"):
                                room_url = "http:" + room_url.removeprefix("ws:")
                            container_env["OPENAI_BASE_URL"] = f"{room_url}/openai/v1"

                    try:
                        self._container_id = await self._room.containers.run(
                            command="sleep infinity",
                            image=self._image,
                            mounts=self._mounts,
                            writable_root_fs=True,
                            env=container_env,
                        )
                    except Exception as exc:
                        raise CodexAppServerError(
                            f"unable to launch codex app-server container image: {self._image}"
                        ) from exc

                command_to_run = self._command
                if self._cwd is not None and self._cwd.strip() != "":
                    command_to_run = (
                        f"cd {shlex.quote(self._cwd.strip())} && {command_to_run}"
                    )

                try:
                    logger.info(f"starting codex: {command_to_run}")
                    self._container_exec = await self._room.containers.exec(
                        container_id=self._container_id,
                        command=["bash", "-lc", command_to_run],
                        tty=False,
                    )
                except Exception as exc:
                    raise CodexAppServerError(
                        "unable to start codex app-server in container"
                    ) from exc

                self._reader_task = asyncio.create_task(self._read_container_stdout())
                self._stderr_task = asyncio.create_task(self._read_container_stderr())
            else:
                if self._command is None:
                    raise CodexAppServerError("codex command was empty")

                argv = shlex.split(self._command)
                if len(argv) == 0:
                    raise CodexAppServerError("codex command was empty")

                try:
                    self._process = await asyncio.create_subprocess_exec(
                        *argv,
                        cwd=self._cwd,
                        env=self._env,
                        stdin=asyncio.subprocess.PIPE,
                        stdout=asyncio.subprocess.PIPE,
                        stderr=asyncio.subprocess.PIPE,
                    )
                except FileNotFoundError as exc:
                    raise CodexAppServerError(
                        f"unable to launch codex app-server with command: {self._command}"
                    ) from exc

                self._reader_task = asyncio.create_task(self._read_stdout())
                self._stderr_task = asyncio.create_task(self._read_stderr())

            await self._request(
                method="initialize",
                params={
                    "protocolVersion": "0.2.0",
                    "clientInfo": {
                        "name": "meshagent-codex",
                        "version": __version__,
                    },
                    "capabilities": {
                        "experimentalApi": True,
                    },
                },
                ensure_started=False,
            )
            await self._notify(
                method="initialized",
                params={},
                ensure_started=False,
            )

            if login_api_key is None and self._room is not None:
                token = getattr(self._room.protocol, "token", None)
                if isinstance(token, str) and token.strip() != "":
                    login_api_key = token.strip()

            if login_api_key is not None:
                logger.info("authenticating codex app-server via account/login/start")
                await self._request(
                    method="account/login/start",
                    params={
                        "type": "apiKey",
                        "apiKey": login_api_key,
                    },
                    ensure_started=False,
                )

            self._started = True

    async def close(self) -> None:
        if (
            self._process is None
            and self._websocket is None
            and self._client_session is None
            and self._container_exec is None
            and self._container_id is None
        ):
            return

        for future in self._pending.values():
            if not future.done():
                future.set_exception(CodexAppServerError("codex session closed"))
        self._pending.clear()

        with contextlib.suppress(Exception):
            if self._process is not None and self._process.stdin is not None:
                self._process.stdin.close()

        if self._reader_task is not None:
            self._reader_task.cancel()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await self._reader_task
            self._reader_task = None

        if self._stderr_task is not None:
            self._stderr_task.cancel()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await self._stderr_task
            self._stderr_task = None

        if self._websocket is not None:
            with contextlib.suppress(Exception):
                await self._websocket.close()
            self._websocket = None

        if self._client_session is not None:
            with contextlib.suppress(Exception):
                await self._client_session.close()
            self._client_session = None

        if self._process is not None:
            if self._process.returncode is None:
                self._process.terminate()
                with contextlib.suppress(Exception):
                    await asyncio.wait_for(self._process.wait(), timeout=3)

            if self._process.returncode is None:
                self._process.kill()
                with contextlib.suppress(Exception):
                    await self._process.wait()

            self._process = None

        if self._container_exec is not None:
            with contextlib.suppress(Exception):
                await self._container_exec.kill()
            with contextlib.suppress(Exception):
                await self._container_exec.result
            self._container_exec = None

        if self._container_id is not None:
            if self._room is not None:
                with contextlib.suppress(Exception):
                    await self._room.containers.stop(
                        container_id=self._container_id,
                        force=True,
                    )
            self._container_id = None

        self._started = False
        self._room = None

    async def request(self, *, method: str, params: Optional[dict] = None) -> Any:
        return await self._request(method=method, params=params, ensure_started=True)

    async def notify(self, *, method: str, params: Optional[dict] = None) -> None:
        await self._notify(method=method, params=params, ensure_started=True)

    async def next_notification(self) -> dict:
        await self.start()
        return await self._notifications.get()

    async def _request(
        self,
        *,
        method: str,
        params: Optional[dict] = None,
        ensure_started: bool,
    ) -> Any:
        if ensure_started:
            await self.start()

        request_id = self._next_id
        self._next_id += 1

        loop = asyncio.get_running_loop()
        future = loop.create_future()
        self._pending[request_id] = future

        await self._send_json(
            {
                "jsonrpc": "2.0",
                "id": request_id,
                "method": method,
                "params": params or {},
            }
        )

        try:
            return await asyncio.wait_for(future, timeout=self._request_timeout_s)
        finally:
            self._pending.pop(request_id, None)

    async def _notify(
        self,
        *,
        method: str,
        params: Optional[dict] = None,
        ensure_started: bool,
    ) -> None:
        if ensure_started:
            await self.start()

        await self._send_json(
            {
                "jsonrpc": "2.0",
                "method": method,
                "params": params or {},
            }
        )

    async def _send_json(self, payload: dict) -> None:
        if self._websocket is not None:
            encoded = json.dumps(payload)
            async with self._write_lock:
                await self._websocket.send_str(encoded)
            return

        if self._container_exec is not None:
            encoded = (json.dumps(payload) + "\n").encode("utf-8")
            async with self._write_lock:
                await self._container_exec.write(encoded)
            return

        if self._process is None or self._process.stdin is None:
            raise CodexAppServerError("codex app-server transport is not running")

        encoded = (json.dumps(payload) + "\n").encode("utf-8")
        async with self._write_lock:
            self._process.stdin.write(encoded)
            await self._process.stdin.drain()

    async def _handle_message_text(self, *, text: str) -> None:
        try:
            message = json.loads(text)
        except Exception:
            logger.warning("unable to parse codex app-server line as json: %s", text)
            return

        await self._dispatch_message(message)

    async def _read_stdout(self) -> None:
        if self._process is None or self._process.stdout is None:
            return

        try:
            while True:
                line = await self._process.stdout.readline()
                if not line:
                    break

                text = line.decode("utf-8", errors="replace").strip()
                if text == "":
                    continue

                if self._forward_stdout:
                    print(text, file=sys.stdout, flush=True)

                await self._handle_message_text(text=text)
        except asyncio.CancelledError:
            raise

        return_code = self._process.returncode
        if return_code is None:
            with contextlib.suppress(Exception):
                await self._process.wait()
            return_code = self._process.returncode

        if return_code is None:
            message = "codex app-server subprocess stdout closed unexpectedly"
        else:
            message = f"codex app-server subprocess exited unexpectedly with return code {return_code}"
        self._mark_transport_failure(message=message)
        self._process = None

    async def _read_container_stdout(self) -> None:
        if self._container_exec is None:
            return

        buffer = bytearray()
        try:
            async for chunk in self._container_exec.stdout():
                if not chunk:
                    continue

                buffer.extend(chunk)
                while True:
                    newline_index = buffer.find(b"\n")
                    if newline_index < 0:
                        break

                    raw_line = bytes(buffer[:newline_index])
                    del buffer[: newline_index + 1]

                    text = raw_line.decode("utf-8", errors="replace").strip()
                    if text == "":
                        continue

                    if self._forward_stdout:
                        print(text, file=sys.stdout, flush=True)

                    await self._handle_message_text(text=text)

            if len(buffer) > 0:
                text = buffer.decode("utf-8", errors="replace").strip()
                if text != "":
                    if self._forward_stdout:
                        print(text, file=sys.stdout, flush=True)
                    await self._handle_message_text(text=text)
        except asyncio.CancelledError:
            raise

        status_text: Optional[str] = None
        if self._container_exec is not None:
            with contextlib.suppress(Exception):
                result = await self._container_exec.result
                if result is not None:
                    status_text = str(result)

        if status_text is None or status_text == "":
            message = "codex app-server container stdout closed unexpectedly"
        else:
            message = (
                "codex app-server container exec exited unexpectedly with status "
                f"{status_text}"
            )
        self._mark_transport_failure(message=message)
        self._container_exec = None

    async def _read_container_stderr(self) -> None:
        if self._container_exec is None:
            return

        buffer = bytearray()
        async for chunk in self._container_exec.stderr():
            if not chunk:
                continue

            buffer.extend(chunk)
            while True:
                newline_index = buffer.find(b"\n")
                if newline_index < 0:
                    break

                raw_line = bytes(buffer[:newline_index])
                del buffer[: newline_index + 1]
                text = raw_line.decode("utf-8", errors="replace").rstrip()
                if text:
                    if self._forward_stderr:
                        print(text, file=sys.stderr, flush=True)
                    else:
                        logger.debug("codex stderr: %s", text)

        if len(buffer) > 0:
            text = buffer.decode("utf-8", errors="replace").rstrip()
            if text:
                if self._forward_stderr:
                    print(text, file=sys.stderr, flush=True)
                else:
                    logger.debug("codex stderr: %s", text)

    async def _read_websocket(self) -> None:
        if self._websocket is None:
            return

        message = "codex app-server websocket closed unexpectedly"
        try:
            while True:
                ws_message = await self._websocket.receive()
                if ws_message.type == aiohttp.WSMsgType.TEXT:
                    text = ws_message.data.strip()
                    if text != "":
                        await self._handle_message_text(text=text)
                    continue

                if ws_message.type == aiohttp.WSMsgType.BINARY:
                    text = ws_message.data.decode("utf-8", errors="replace").strip()
                    if text != "":
                        await self._handle_message_text(text=text)
                    continue

                if ws_message.type == aiohttp.WSMsgType.ERROR:
                    message = f"codex websocket error: {self._websocket.exception()}"
                    break

                if ws_message.type in (
                    aiohttp.WSMsgType.CLOSED,
                    aiohttp.WSMsgType.CLOSE,
                    aiohttp.WSMsgType.CLOSING,
                ):
                    break
        except asyncio.CancelledError:
            raise

        self._mark_transport_failure(message=message)
        self._websocket = None

    async def _read_stderr(self) -> None:
        if self._process is None or self._process.stderr is None:
            return

        while True:
            line = await self._process.stderr.readline()
            if not line:
                break
            text = line.decode("utf-8", errors="replace").rstrip()
            if text:
                if self._forward_stderr:
                    print(text, file=sys.stderr, flush=True)
                else:
                    logger.debug("codex stderr: %s", text)

    async def _dispatch_message(self, message: dict) -> None:
        request_id = message.get("id")
        method = message.get("method")

        # Standard JSON-RPC response
        if request_id is not None and method is None:
            request_key = None
            if isinstance(request_id, int):
                request_key = request_id
            elif isinstance(request_id, str) and request_id.isdigit():
                request_key = int(request_id)

            if request_key is None:
                return

            future = self._pending.get(request_key)
            if future is None or future.done():
                return

            if "error" in message:
                error = message.get("error") or {}
                future.set_exception(
                    CodexAppServerError(
                        f"codex app-server request failed: {error.get('message', error)}"
                    )
                )
                return

            future.set_result(message.get("result"))
            return

        # Server-initiated request.
        if request_id is not None and isinstance(method, str):
            asyncio.create_task(self._handle_server_request(message=message))
            return

        if isinstance(method, str):
            self._notifications.put_nowait(message)

    async def _handle_server_request(self, *, message: dict) -> None:
        request_id = message.get("id")
        method = message.get("method")
        params = message.get("params") or {}
        if not isinstance(params, dict):
            params = {}

        if not isinstance(method, str):
            return

        if not isinstance(request_id, (int, str)):
            return

        if self._server_request_handler is not None:
            try:
                handled = await self._server_request_handler(method, params)
            except Exception as exc:
                logger.error(
                    "codex app-server server request handler failed for %s",
                    method,
                    exc_info=exc,
                )
                await self._send_json(
                    {
                        "jsonrpc": "2.0",
                        "id": request_id,
                        "error": {
                            "code": -32000,
                            "message": str(exc),
                        },
                    }
                )
                return

            if handled is not None:
                await self._send_json(
                    {
                        "jsonrpc": "2.0",
                        "id": request_id,
                        "result": handled,
                    }
                )
                return

        if method.endswith("/requestApproval"):
            await self._send_json(
                {
                    "jsonrpc": "2.0",
                    "id": request_id,
                    "result": {"decision": "accept"},
                }
            )
            return

        logger.debug(
            "received unsupported server request from codex app-server: %s %s",
            method,
            params,
        )
        await self._send_json(
            {
                "jsonrpc": "2.0",
                "id": request_id,
                "error": {
                    "code": -32601,
                    "message": f"unsupported server request: {method}",
                },
            }
        )


class _CodexThreadState:
    def __init__(self, *, thread_id: str, context: AgentChatContext):
        self.thread_id = thread_id
        self.context = context


class _CodexAppServerBackend:
    def __init__(
        self,
        *,
        model: str = os.getenv("CODEX_MODEL", "codex-mini-latest"),
        command: Optional[str] = os.getenv("MESHAGENT_CODEX_COMMAND"),
        ws_url: Optional[str] = os.getenv("MESHAGENT_CODEX_WS_URL"),
        image: Optional[str] = os.getenv("MESHAGENT_CODEX_IMAGE"),
        mounts: Optional[ContainerMountSpec] = DEFAULT_CODEX_CONTAINER_MOUNTS,
        cwd: Optional[str] = os.getenv("MESHAGENT_CODEX_CWD"),
        approval_policy: str = os.getenv("MESHAGENT_CODEX_APPROVAL_POLICY", "never"),
        sandbox_policy: str = os.getenv(
            "MESHAGENT_CODEX_SANDBOX_POLICY", "workspace-write"
        ),
        forward_stdout: bool = _env_bool("MESHAGENT_CODEX_FORWARD_STDOUT", False),
        forward_stderr: bool = _env_bool("MESHAGENT_CODEX_FORWARD_STDERR", True),
        env: Optional[dict[str, str]] = None,
        request_timeout_s: float = 300.0,
        approval_request_handler: Optional[Callable[..., Awaitable[str]]] = None,
    ):
        if ws_url is None and command is None:
            if image is not None:
                command = (
                    "codex app-server "
                    "-c model_providers.openai.name='OpenAI' "
                    '-c model_providers.openai.base_url="$OPENAI_BASE_URL"'
                )
            else:
                command = "codex app-server"
        elif ws_url is not None:
            # Explicit websocket transport takes precedence over local process launch.
            command = None
            image = None

        self._model = model
        self._command = command
        self._ws_url = ws_url
        self._image = image
        self._mounts = mounts
        self._cwd = cwd
        self._approval_policy = approval_policy
        self._sandbox_policy = sandbox_policy
        self._approval_request_handler = approval_request_handler

        launch_env = os.environ.copy()
        if env is not None:
            launch_env.update(env)
        container_env = env or {}
        session_env = container_env if image is not None else launch_env

        self._session = _CodexJsonRpcSession(
            command=command,
            ws_url=ws_url,
            image=image,
            mounts=mounts,
            cwd=cwd,
            env=session_env,
            forward_stdout=forward_stdout,
            forward_stderr=forward_stderr,
            request_timeout_s=request_timeout_s,
            server_request_handler=self._handle_server_request,
        )
        self._router_start_lock = asyncio.Lock()
        self._router_route_lock = asyncio.Lock()
        self._router_task: Optional[asyncio.Task] = None
        self._router_error: Optional[Exception] = None
        self._turn_queues: dict[tuple[str, str], asyncio.Queue[dict]] = {}
        self._pending_turn_notifications: dict[tuple[str, str], deque[dict]] = {}
        self._thread_map_lock = asyncio.Lock()
        self._thread_states: dict[str, _CodexThreadState] = {}
        self._thread_keys_by_thread_id: dict[str, str] = {}
        self._active_turns_lock = asyncio.Lock()
        self._active_turns: dict[str, set[tuple[str, str]]] = {}

    async def close(self) -> None:
        if self._router_task is not None:
            self._router_task.cancel()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await self._router_task
            self._router_task = None

        async with self._router_route_lock:
            self._turn_queues.clear()
            self._pending_turn_notifications.clear()

        async with self._thread_map_lock:
            self._thread_states.clear()
            self._thread_keys_by_thread_id.clear()

        async with self._active_turns_lock:
            self._active_turns.clear()

        self._router_error = None
        await self._session.close()

    def _extract_thread_id(self, result: Any) -> str:
        thread_id = None
        if isinstance(result, dict):
            thread = result.get("thread")
            if isinstance(thread, dict):
                candidate = thread.get("id")
                if isinstance(candidate, str):
                    thread_id = candidate

            if thread_id is None:
                candidate = result.get("threadId")
                if isinstance(candidate, str):
                    thread_id = candidate

        if thread_id is None:
            raise CodexAppServerError("thread/start did not return a thread id")

        return thread_id

    async def _start_thread(
        self,
        *,
        model: str,
    ) -> str:
        default_cwd = os.getcwd()
        if self._image is not None and self._ws_url is None:
            default_cwd = "/data"

        result = await self._session.request(
            method="thread/start",
            params={
                "model": model,
                "cwd": self._cwd or default_cwd,
                "approvalPolicy": self._approval_policy,
                "sandboxPolicy": self._sandbox_policy,
            },
        )
        return self._extract_thread_id(result)

    async def _resume_thread(self, *, thread_id: str) -> bool:
        try:
            await self._session.request(
                method="thread/resume",
                params={"threadId": thread_id},
            )
            return True
        except Exception:
            return False

    async def _get_thread_state(
        self, *, thread_key: str
    ) -> Optional[_CodexThreadState]:
        async with self._thread_map_lock:
            return self._thread_states.get(thread_key)

    async def _set_thread_state(
        self,
        *,
        thread_key: str,
        thread_id: str,
        context: AgentChatContext,
    ) -> None:
        async with self._thread_map_lock:
            previous = self._thread_states.get(thread_key)
            if previous is not None:
                self._thread_keys_by_thread_id.pop(previous.thread_id, None)
            self._thread_states[thread_key] = _CodexThreadState(
                thread_id=thread_id,
                context=context,
            )
            self._thread_keys_by_thread_id[thread_id] = thread_key

    async def _clear_thread_state(self, *, thread_key: str) -> None:
        async with self._thread_map_lock:
            previous = self._thread_states.pop(thread_key, None)
            if previous is not None:
                self._thread_keys_by_thread_id.pop(previous.thread_id, None)

    async def _thread_key_for_thread_id(self, *, thread_id: str) -> Optional[str]:
        async with self._thread_map_lock:
            thread_key = self._thread_keys_by_thread_id.get(thread_id)
            if thread_key is not None:
                return thread_key

            for key, state in self._thread_states.items():
                if state.thread_id == thread_id:
                    self._thread_keys_by_thread_id[thread_id] = key
                    return key

        return None

    def _normalize_approval_decision(self, *, decision: Any) -> str:
        if isinstance(decision, str):
            normalized = decision.strip()
            if normalized != "":
                lower = normalized.lower().replace("-", "_").replace(" ", "_")
                if lower in ("accept",):
                    return "accept"
                if lower in ("accept_for_session", "acceptforsession"):
                    return "acceptForSession"
                if lower in ("decline", "reject", "rejected"):
                    return "decline"
                if lower in ("cancel", "cancelled", "canceled"):
                    return "cancel"

        return "accept"

    async def _handle_server_request(self, method: str, params: dict) -> Optional[dict]:
        if not isinstance(method, str):
            return None

        if not method.endswith("/requestApproval"):
            return None

        thread_id = _get_nested_id(
            params,
            singular="thread",
            nested_key="thread",
        )

        decision = "accept"
        if thread_id is not None and self._approval_request_handler is not None:
            thread_key = await self._thread_key_for_thread_id(thread_id=thread_id)
            if thread_key is not None:
                decision = await self._approval_request_handler(
                    thread_key=thread_key,
                    method=method,
                    params=params,
                )
            else:
                logger.warning(
                    "received codex approval request for unknown thread id '%s'",
                    thread_id,
                )

        return {"decision": self._normalize_approval_decision(decision=decision)}

    def _normalize_skill_path(self, *, path: str) -> str:
        normalized = path.strip()
        if normalized == "":
            return ""

        skill_path = Path(normalized).expanduser()

        if skill_path.is_file():
            return str(skill_path.resolve())

        if skill_path.is_dir():
            upper = skill_path / "SKILL.md"
            if upper.is_file():
                return str(upper.resolve())

            lower = skill_path / "skill.md"
            if lower.is_file():
                return str(lower.resolve())

            return str((skill_path / "SKILL.md").resolve())

        lower_name = skill_path.name.lower()
        if lower_name == "skill.md":
            return normalized

        return str(skill_path / "SKILL.md")

    def _normalize_skill_paths(self, *, paths: list[str]) -> list[str]:
        normalized: list[str] = []
        seen = set[str]()

        for path in paths:
            normalized_path = self._normalize_skill_path(path=path)
            if normalized_path == "" or normalized_path in seen:
                continue
            seen.add(normalized_path)
            normalized.append(normalized_path)

        return normalized

    async def set_skill_enabled(self, *, path: str, enabled: bool) -> None:
        await self._session.request(
            method="skills/config/write",
            params={
                "path": path,
                "enabled": enabled,
            },
        )

    async def enable_skills(self, *, paths: list[str]) -> None:
        for path in self._normalize_skill_paths(paths=paths):
            await self.set_skill_enabled(path=path, enabled=True)

    async def disable_skills(self, *, paths: list[str]) -> None:
        for path in self._normalize_skill_paths(paths=paths):
            await self.set_skill_enabled(path=path, enabled=False)

    async def on_thread_open(
        self,
        *,
        thread_key: str,
        room: RoomClient,
        context: AgentChatContext,
        model: Optional[str] = None,
        skill_dirs: Optional[list[str]] = None,
    ) -> None:
        if model is None:
            model = self._model

        await self._ensure_router_started(room=room)

        existing_state = await self._get_thread_state(thread_key=thread_key)
        if existing_state is not None:
            resumed = await self._resume_thread(thread_id=existing_state.thread_id)
            if resumed:
                await self._set_thread_state(
                    thread_key=thread_key,
                    thread_id=existing_state.thread_id,
                    context=context,
                )
                return

        thread_id = await self._start_thread(model=model)
        await self._set_thread_state(
            thread_key=thread_key,
            thread_id=thread_id,
            context=context,
        )

        if skill_dirs is not None and len(skill_dirs) > 0:
            await self.enable_skills(paths=skill_dirs)

    async def on_thread_clear(
        self,
        *,
        thread_key: str,
        context: AgentChatContext,
    ) -> None:
        await self.on_thread_cancel(thread_key=thread_key)
        await self._clear_thread_state(thread_key=thread_key)

    async def on_thread_cancel(self, *, thread_key: str) -> None:
        async with self._active_turns_lock:
            active_turns = list(self._active_turns.pop(thread_key, set()))

        for thread_id, turn_id in active_turns:
            with contextlib.suppress(Exception):
                await self._session.request(
                    method="turn/interrupt",
                    params={"threadId": thread_id, "turnId": turn_id},
                )

    async def on_thread_close(self, *, thread_key: str) -> None:
        await self.on_thread_cancel(thread_key=thread_key)
        await self._clear_thread_state(thread_key=thread_key)

    async def _track_active_turn(
        self,
        *,
        thread_key: str,
        thread_id: str,
        turn_id: str,
    ) -> None:
        async with self._active_turns_lock:
            active_turns = self._active_turns.get(thread_key)
            if active_turns is None:
                active_turns = set()
                self._active_turns[thread_key] = active_turns
            active_turns.add((thread_id, turn_id))

    async def _untrack_active_turn(
        self,
        *,
        thread_key: str,
        thread_id: str,
        turn_id: str,
    ) -> None:
        async with self._active_turns_lock:
            active_turns = self._active_turns.get(thread_key)
            if active_turns is None:
                return

            active_turns.discard((thread_id, turn_id))
            if len(active_turns) == 0:
                self._active_turns.pop(thread_key, None)

    def _extract_turn_id(self, result: Any) -> str:
        if isinstance(result, dict):
            turn = result.get("turn")
            if isinstance(turn, dict):
                value = turn.get("id")
                if isinstance(value, str):
                    return value

            for key in ("turnId", "turn_id", "id"):
                value = result.get(key)
                if isinstance(value, str):
                    return value

        raise CodexAppServerError("turn/start did not return a turn id")

    def _notification_turn_key(
        self,
        *,
        notification: dict,
    ) -> tuple[Optional[str], Optional[str]]:
        params = notification.get("params") or {}
        if not isinstance(params, dict):
            return None, None

        thread_id = _get_nested_id(
            params,
            singular="thread",
            nested_key="thread",
        )
        turn_id = _get_nested_id(
            params,
            singular="turn",
            nested_key="turn",
        )

        msg = params.get("msg")
        if isinstance(msg, dict):
            if thread_id is None:
                for key in (
                    "thread_id",
                    "threadId",
                    "conversation_id",
                    "conversationId",
                ):
                    value = msg.get(key)
                    if isinstance(value, str):
                        thread_id = value
                        break

            if turn_id is None:
                for key in ("turn_id", "turnId"):
                    value = msg.get(key)
                    if isinstance(value, str):
                        turn_id = value
                        break

        if thread_id is None:
            for key in ("threadId", "thread_id", "conversationId", "conversation_id"):
                value = params.get(key)
                if isinstance(value, str):
                    thread_id = value
                    break

        if turn_id is None:
            for key in ("turnId", "turn_id"):
                value = params.get(key)
                if isinstance(value, str):
                    turn_id = value
                    break

        return thread_id, turn_id

    async def _route_notifications(self) -> None:
        try:
            while True:
                notification = await self._session.next_notification()

                if notification.get("method") == "__session_error__":
                    error = (notification.get("params") or {}).get(
                        "error", "codex session transport failed"
                    )
                    async with self._router_route_lock:
                        for queue in self._turn_queues.values():
                            queue.put_nowait(
                                {
                                    "method": "__router_error__",
                                    "params": {"error": str(error)},
                                }
                            )
                    continue

                thread_id, turn_id = self._notification_turn_key(
                    notification=notification
                )
                if thread_id is None or turn_id is None:
                    continue

                key = (thread_id, turn_id)
                async with self._router_route_lock:
                    queue = self._turn_queues.get(key)
                    if queue is not None:
                        queue.put_nowait(notification)
                        continue

                    pending = self._pending_turn_notifications.get(key)
                    if pending is None:
                        pending = deque()
                        self._pending_turn_notifications[key] = pending

                    pending.append(notification)

                    # Keep a small bounded backlog to prevent growth from stale turns.
                    while len(pending) > 100:
                        pending.popleft()

        except asyncio.CancelledError:
            raise
        except Exception as exc:
            self._router_error = exc
            async with self._router_route_lock:
                for queue in self._turn_queues.values():
                    queue.put_nowait(
                        {
                            "method": "__router_error__",
                            "params": {"error": str(exc)},
                        }
                    )

    async def _ensure_router_started(self, *, room: Optional[RoomClient]) -> None:
        self._session.set_room(room=room)
        await self._session.start(room=room)

        async with self._router_start_lock:
            if self._router_task is not None and not self._router_task.done():
                return

            self._router_error = None
            self._router_task = asyncio.create_task(self._route_notifications())

    async def _register_turn_queue(
        self,
        *,
        thread_id: str,
        turn_id: str,
    ) -> asyncio.Queue[dict]:
        key = (thread_id, turn_id)
        async with self._router_route_lock:
            existing = self._turn_queues.get(key)
            if existing is not None:
                return existing

            queue: asyncio.Queue[dict] = asyncio.Queue()
            self._turn_queues[key] = queue

            pending = self._pending_turn_notifications.pop(key, None)
            if pending is not None:
                while pending:
                    queue.put_nowait(pending.popleft())

            return queue

    async def _unregister_turn_queue(self, *, thread_id: str, turn_id: str) -> None:
        key = (thread_id, turn_id)
        async with self._router_route_lock:
            self._turn_queues.pop(key, None)

    async def _next_turn_notification(
        self,
        *,
        turn_queue: asyncio.Queue[dict],
    ) -> dict:
        if self._router_error is not None:
            raise CodexAppServerError(
                f"codex app-server notification router failed: {self._router_error}"
            )

        notification = await turn_queue.get()
        if notification.get("method") == "__router_error__":
            error = (notification.get("params") or {}).get("error")
            raise CodexAppServerError(
                f"codex app-server notification router failed: {error}"
            )

        return notification

    def _extract_delta(self, *, params: dict) -> str:
        for key in ("delta", "textDelta", "text_delta"):
            value = params.get(key)
            if isinstance(value, str):
                return value

        item = params.get("item")
        if isinstance(item, dict):
            for key in ("delta", "textDelta", "text_delta"):
                value = item.get(key)
                if isinstance(value, str):
                    return value

        msg = params.get("msg")
        if isinstance(msg, dict):
            for key in ("delta", "textDelta", "text_delta"):
                value = msg.get(key)
                if isinstance(value, str):
                    return value

        return ""

    def _extract_item(self, *, params: dict) -> dict:
        item = params.get("item")
        if isinstance(item, dict):
            return item
        msg = params.get("msg")
        if isinstance(msg, dict):
            item = msg.get("item")
            if isinstance(item, dict):
                return item
        return {}

    def _normalize_developer_instructions(
        self,
        *,
        developer_instructions: Optional[str | list[str]],
    ) -> Optional[str]:
        if developer_instructions is None:
            return None

        if isinstance(developer_instructions, list):
            if len(developer_instructions) == 0:
                return None
            instructions = "\n".join(developer_instructions).strip()
            return instructions if instructions != "" else None

        instructions = developer_instructions.strip()
        return instructions if instructions != "" else None

    def _resolve_model(self, *, model: Optional[str]) -> str:
        resolved = model if model is not None else self._model
        if not isinstance(resolved, str):
            resolved = ""

        resolved = resolved.strip()
        if resolved == "":
            resolved = "codex-mini-latest"

        return resolved

    def _normalize_turn_input(self, *, message: str | list[dict]) -> list[dict]:
        if isinstance(message, str):
            if message.strip() == "":
                raise CodexAppServerError("message cannot be empty")
            return [{"type": "text", "text": message}]

        if not isinstance(message, list) or len(message) == 0:
            raise CodexAppServerError(
                "message must be a non-empty string or input list"
            )

        turn_input: list[dict] = []
        for item in message:
            if not isinstance(item, dict):
                raise CodexAppServerError("input list items must be objects")
            turn_input.append(item)

        if len(turn_input) == 0:
            raise CodexAppServerError("input list cannot be empty")

        return turn_input

    def _should_emit_status_event(self, *, method: Optional[str]) -> bool:
        if not isinstance(method, str) or method == "":
            return False

        lower = method.lower()

        # Assistant message text deltas are already streamed as normal chat output.
        if lower in (
            "item/agentmessage/delta",
            "codex/event/agent_message_delta",
        ):
            return False

        # These notifications do not carry stable command/item payloads.
        # Use item/started + item/completed for exec rendering instead.
        if lower in (
            "codex/event/exec_command_begin",
            "codex/event/exec_command_end",
        ):
            return False

        # Avoid flooding thread history with token- or stream-level deltas.
        if (
            lower.endswith("/delta")
            or lower.endswith("/outputdelta")
            or lower.endswith("/summarytextdelta")
            or lower.endswith("/summarypartadded")
            or lower.endswith("/textdelta")
            or lower.endswith("/terminalinteraction")
        ):
            return False

        return True

    def _truncate_text(self, *, text: str, limit: int) -> str:
        if len(text) <= limit:
            return text
        return text[:limit] + "..."

    def _normalize_name(self, *, value: str) -> str:
        return "".join(ch for ch in value.lower() if ch.isalnum())

    def _first_text(self, *, source: dict, keys: tuple[str, ...]) -> str:
        for key in keys:
            value = source.get(key)
            if isinstance(value, str):
                text = value.strip()
                if text != "":
                    return text
        return ""

    def _first_nested_text(self, *, value: Any, keys: tuple[str, ...]) -> str:
        key_set = {key.lower() for key in keys}

        if isinstance(value, dict):
            for key, nested in value.items():
                if key.lower() in key_set:
                    text = _to_text(nested).strip()
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

    def _command_value_text(self, *, value: Any) -> str:
        if isinstance(value, list):
            parts: list[str] = []
            for item in value:
                text = self._command_value_text(value=item)
                if text != "":
                    parts.append(text)
            return " ".join(parts)

        return _to_text(value).strip()

    def _mapping_command_text(self, *, mapping: dict, keys: tuple[str, ...]) -> str:
        for key in keys:
            if key not in mapping:
                continue
            text = self._command_value_text(value=mapping.get(key))
            if text != "":
                return text
        return ""

    def _extract_exec_command(self, *, item: dict) -> str:
        keys = (
            "command",
            "cmd",
            "shell_command",
            "shellcommand",
            "raw_command",
            "rawcommand",
        )

        command = self._mapping_command_text(mapping=item, keys=keys)
        if command != "":
            return command

        action = item.get("action")
        if isinstance(action, dict):
            command = self._mapping_command_text(mapping=action, keys=keys)
            if command != "":
                return command

        raw_actions = item.get("commandActions")
        if not isinstance(raw_actions, list):
            raw_actions = item.get("command_actions")
        if isinstance(raw_actions, list):
            for action in raw_actions:
                if not isinstance(action, dict):
                    continue
                command = self._mapping_command_text(mapping=action, keys=keys)
                if command != "":
                    return command

        return self._first_nested_text(
            value=item,
            keys=keys,
        )

    def _is_active_state(self, *, state: str) -> bool:
        return state in ("in_progress", "queued")

    def _item_kind_from_type(self, *, item_type: str) -> Optional[str]:
        normalized = self._normalize_name(value=item_type)
        if normalized == "":
            return None

        if normalized == "agentmessage":
            return "message"
        if normalized == "reasoning":
            return "reasoning"
        if normalized == "plan":
            return "plan"
        if normalized == "commandexecution":
            return "exec"
        if normalized == "filechange":
            return "diff"
        if normalized == "mcptoolcall":
            return "tool"
        if normalized == "collabagenttoolcall":
            return "collab"
        if normalized == "websearch":
            return "web"
        if normalized == "imageview":
            return "image"
        if normalized == "contextcompaction":
            return "context"
        return None

    def _command_action_detail(self, *, action: dict) -> Optional[str]:
        type_name = action.get("type")
        if not isinstance(type_name, str):
            return None

        normalized = self._normalize_name(value=type_name)
        if normalized == "read":
            text = self._first_text(
                source=action,
                keys=("name", "path", "command"),
            )
            return f"Read {text}" if text != "" else None

        if normalized == "listfiles":
            text = self._first_text(
                source=action,
                keys=("path", "command"),
            )
            return f"List {text if text != '' else 'files'}"

        if normalized == "search":
            query = self._first_text(
                source=action,
                keys=("query",),
            )
            command = self._first_text(
                source=action,
                keys=("command",),
            )
            path = self._first_text(
                source=action,
                keys=("path",),
            )
            text = query if query != "" else command
            if path != "":
                if text != "":
                    text = f"{text} in {path}"
                else:
                    text = path
            return f"Search {text}" if text != "" else None

        if normalized in ("run", "unknown", "command"):
            text = self._mapping_command_text(
                mapping=action,
                keys=("command", "cmd"),
            )
            return f"Run {text}" if text != "" else None

        return None

    def _extract_exec_action_details(self, *, item: dict) -> list[str]:
        raw_actions = item.get("commandActions")
        if not isinstance(raw_actions, list):
            raw_actions = item.get("command_actions")
        if not isinstance(raw_actions, list):
            return []

        details: list[str] = []
        seen: set[str] = set()
        for action in raw_actions:
            if not isinstance(action, dict):
                continue

            detail = self._command_action_detail(action=action)
            if detail is None:
                continue

            key = detail.strip().lower()
            if key in seen:
                continue
            seen.add(key)
            details.append(detail)

        return details

    def _is_exploration_actions(self, *, action_details: list[str]) -> bool:
        if len(action_details) == 0:
            return False
        for detail in action_details:
            if not (
                detail.startswith("Read ")
                or detail.startswith("List ")
                or detail.startswith("Search ")
            ):
                return False
        return True

    def _exec_display(
        self, *, status: str, item: dict
    ) -> tuple[Optional[str], list[str]]:
        command = self._extract_exec_command(item=item)
        action_details = self._extract_exec_action_details(item=item)
        if command == "" and len(action_details) == 0:
            return None, []

        details: list[str] = []
        if command != "":
            details.append(command)
        for detail in action_details:
            if command != "" and detail.lower().startswith("run "):
                continue
            if detail in details:
                continue
            details.append(detail)

        is_exploration = self._is_exploration_actions(action_details=action_details)
        if is_exploration:
            if command != "":
                command_line = f"Run {command}"
                if command_line not in details:
                    details.insert(0, command_line)

            if status == "failed":
                return "Exploration Failed", details
            if status == "cancelled":
                return "Exploration Cancelled", details
            if self._is_active_state(state=status):
                return "Exploring", details
            return "Explored", details

        if status == "failed":
            return "Command Failed", details
        if status == "cancelled":
            return "Command Cancelled", details
        if self._is_active_state(state=status):
            return "Running Command", details
        if status == "completed":
            return "Ran Command", details
        return "Command", details

    def _line_count_summary(self, *, added: int, removed: int) -> str:
        return f"(+{added} -{removed})"

    def _diff_line_counts(self, *, diff: str, kind: str) -> tuple[int, int]:
        added = 0
        removed = 0
        for line in diff.splitlines():
            if line.startswith("+++ ") or line.startswith("--- "):
                continue
            if line.startswith("+"):
                added += 1
                continue
            if line.startswith("-"):
                removed += 1

        if added == 0 and removed == 0:
            line_count = len([line for line in diff.splitlines() if line.strip() != ""])
            normalized_kind = self._normalize_name(value=kind)
            if normalized_kind == "add":
                added = line_count
            elif normalized_kind == "delete":
                removed = line_count

        return added, removed

    def _file_change_preview_lines(
        self,
        *,
        diff: str,
        kind: str,
        limit: int = 6,
    ) -> list[str]:
        lines: list[str] = []
        for line in diff.splitlines():
            if (
                line.startswith("@@")
                or line.startswith("+++ ")
                or line.startswith("--- ")
            ):
                continue
            if line.startswith("+") or line.startswith("-"):
                lines.append(line)
            if len(lines) >= limit:
                break

        if len(lines) > 0:
            return lines

        normalized_kind = self._normalize_name(value=kind)
        for line in diff.splitlines():
            text = line.strip()
            if text == "":
                continue
            if normalized_kind == "add":
                lines.append(f"+{text}")
            elif normalized_kind == "delete":
                lines.append(f"-{text}")
            else:
                lines.append(text)
            if len(lines) >= limit:
                break
        return lines

    def _plan_lines(self, *, params: dict) -> list[str]:
        raw_plan = params.get("plan")
        if not isinstance(raw_plan, list):
            return []

        lines: list[str] = []
        for step in raw_plan:
            if not isinstance(step, dict):
                continue
            text = self._first_text(source=step, keys=("step",))
            if text == "":
                continue
            raw_status = self._first_text(source=step, keys=("status",))
            status = self._normalize_name(value=raw_status)
            if status == "completed":
                lines.append(f"\u2713 {text}")
            elif status in ("inprogress", "in_progress"):
                lines.append(f"\u2192 {text}")
            else:
                lines.append(f"\u2022 {text}")
        return lines

    def _event_display(
        self,
        *,
        method: str,
        params: dict,
        status: str,
        kind: str,
        item: dict,
    ) -> tuple[Optional[str], list[str]]:
        if kind == "plan" and method.lower().startswith("turn/plan/"):
            lines = self._plan_lines(params=params)
            headline = "Updated Plan"
            if self._is_active_state(state=status):
                headline = "Updating Plan"
            return headline, lines

        if kind == "exec":
            return self._exec_display(status=status, item=item)

        if kind == "diff":
            raw_changes = item.get("changes")
            if not isinstance(raw_changes, list) or len(raw_changes) == 0:
                return None, []

            changes: list[tuple[str, str, str, int, int, str]] = []
            total_added = 0
            total_removed = 0
            for change in raw_changes:
                if not isinstance(change, dict):
                    continue
                path = self._first_text(source=change, keys=("path",))
                diff = self._first_text(source=change, keys=("diff",))

                raw_kind = change.get("kind")
                kind_type = ""
                move_path = ""
                if isinstance(raw_kind, str):
                    kind_type = raw_kind
                elif isinstance(raw_kind, dict):
                    kind_type = self._first_text(source=raw_kind, keys=("type",))
                    move_path = self._first_text(
                        source=raw_kind, keys=("movePath", "move_path")
                    )

                if path == "":
                    continue

                added, removed = self._diff_line_counts(diff=diff, kind=kind_type)
                total_added += added
                total_removed += removed
                changes.append((path, move_path, kind_type, added, removed, diff))

            if len(changes) == 0:
                return None, []

            if len(changes) == 1:
                path, move_path, kind_type, added, removed, diff = changes[0]
                normalized_kind = self._normalize_name(value=kind_type)
                verb = "Edited"
                if normalized_kind == "add":
                    verb = "Added"
                elif normalized_kind == "delete":
                    verb = "Deleted"

                path_display = path if move_path == "" else f"{path} \u2192 {move_path}"
                headline = (
                    f"{verb} {path_display} "
                    f"{self._line_count_summary(added=added, removed=removed)}"
                )
                detail_lines = self._file_change_preview_lines(
                    diff=diff,
                    kind=kind_type,
                )
                return headline, detail_lines

            noun = "file" if len(changes) == 1 else "files"
            headline = (
                f"Edited {len(changes)} {noun} "
                f"{self._line_count_summary(added=total_added, removed=total_removed)}"
            )
            detail_lines: list[str] = []
            for path, move_path, _, added, removed, _ in changes:
                path_display = path if move_path == "" else f"{path} \u2192 {move_path}"
                detail_lines.append(
                    f"{path_display} "
                    f"{self._line_count_summary(added=added, removed=removed)}"
                )
            return headline, detail_lines

        if kind == "web":
            query = self._first_text(source=item, keys=("query",))
            action = item.get("action")
            detail = query
            if isinstance(action, dict):
                action_type = self._normalize_name(
                    value=self._first_text(source=action, keys=("type",))
                )
                if action_type == "openpage":
                    url = self._first_text(source=action, keys=("url",))
                    if url != "":
                        detail = f"Open {url}"
                elif action_type == "findinpage":
                    url = self._first_text(source=action, keys=("url",))
                    pattern = self._first_text(source=action, keys=("pattern",))
                    if pattern != "" and url != "":
                        detail = f"Find '{pattern}' in {url}"
                    elif pattern != "":
                        detail = f"Find '{pattern}'"
                elif action_type == "search":
                    search_query = self._first_text(source=action, keys=("query",))
                    if search_query != "":
                        detail = search_query

            headline = "Searched"
            if self._is_active_state(state=status):
                headline = "Searching the web"
            return headline, [detail] if detail != "" else []

        if kind == "tool":
            server = self._first_text(source=item, keys=("server",))
            tool = self._first_text(source=item, keys=("tool",))
            name = ".".join(part for part in (server, tool) if part != "")
            if name == "":
                return None, []

            if self._is_active_state(state=status):
                return f"Calling {name}", []
            if status == "failed":
                return f"Tool failed: {name}", []
            return f"Called {name}", []

        return None, []

    def _stringify_status_payload(self, *, value: Any) -> str:
        try:
            text = json.dumps(value, ensure_ascii=False, default=str, sort_keys=True)
        except Exception:
            text = str(value)
        return self._truncate_text(text=text, limit=8000)

    def _status_summary(self, *, method: str, params: dict, status: str) -> str:
        item = self._extract_item(params=params)
        item_type = _item_type(item)

        turn = params.get("turn")
        turn_status = None
        if isinstance(turn, dict):
            value = turn.get("status")
            if isinstance(value, str) and value.strip() != "":
                turn_status = value

        summary = method
        if item_type != "":
            summary += f" [{item_type}]"
        if turn_status is not None:
            summary += f" ({turn_status})"
        elif status in ("completed", "failed"):
            summary += f" ({status})"

        for key in ("message", "status", "phase", "state", "progress"):
            value = params.get(key)
            if isinstance(value, (str, int, float)) and str(value).strip() != "":
                summary += f": {value}"
                break

        return self._truncate_text(text=summary, limit=280)

    def _normalize_status_value(self, *, value: Any) -> Optional[str]:
        if not isinstance(value, str):
            return None

        normalized = value.strip().lower()
        if normalized == "":
            return None

        normalized = normalized.replace("-", "_").replace(" ", "_")
        if normalized == "inprogress":
            normalized = "in_progress"

        if normalized in ("failed", "error", "errored", "rejected"):
            return "failed"
        if normalized in ("cancelled", "canceled", "interrupted", "aborted", "stopped"):
            return "cancelled"
        if normalized in ("queued", "pending", "waiting"):
            return "queued"
        if normalized in ("running", "started", "starting", "in_progress"):
            return "in_progress"
        if normalized in (
            "completed",
            "complete",
            "done",
            "succeeded",
            "success",
            "finished",
        ):
            return "completed"

        # Handle partial composite values, for example "task_complete" or "command_failed".
        if "fail" in normalized or "error" in normalized:
            return "failed"
        if "cancel" in normalized or "interrupt" in normalized or "abort" in normalized:
            return "cancelled"
        if "queue" in normalized or "pending" in normalized or "wait" in normalized:
            return "queued"
        if "complete" in normalized or "success" in normalized or "done" in normalized:
            return "completed"
        if (
            "progress" in normalized
            or "running" in normalized
            or "start" in normalized
            or "stream" in normalized
            or "update" in normalized
        ):
            return "in_progress"

        return None

    def _status_from_payload(self, *, params: dict) -> Optional[str]:
        if params.get("error") is not None:
            return "failed"

        candidates: list[str] = []
        for key in ("status", "state", "phase"):
            value = params.get(key)
            if isinstance(value, str):
                candidates.append(value)

        for nested_key in ("turn", "item", "msg"):
            nested = params.get(nested_key)
            if not isinstance(nested, dict):
                continue

            for key in ("status", "state", "phase"):
                value = nested.get(key)
                if isinstance(value, str):
                    candidates.append(value)

            nested_item = nested.get("item")
            if isinstance(nested_item, dict):
                for key in ("status", "state", "phase"):
                    value = nested_item.get(key)
                    if isinstance(value, str):
                        candidates.append(value)

        for candidate in candidates:
            status = self._normalize_status_value(value=candidate)
            if status is not None:
                return status

        return None

    def _status_from_notification(self, *, method: str, params: dict) -> str:
        lower = method.lower()

        if (
            lower.endswith("/failed")
            or lower.endswith("_failed")
            or "/error" in lower
            or lower.endswith("_error")
        ):
            return "failed"
        if (
            lower.endswith("/cancelled")
            or lower.endswith("/canceled")
            or lower.endswith("_cancelled")
            or lower.endswith("_canceled")
            or lower.endswith("/interrupted")
            or lower.endswith("_interrupted")
            or lower.endswith("/aborted")
            or lower.endswith("_aborted")
        ):
            return "cancelled"
        if (
            lower.endswith("/queued")
            or lower.endswith("_queued")
            or lower.endswith("/pending")
            or lower.endswith("_pending")
        ):
            return "queued"
        if (
            lower.endswith("/completed")
            or lower.endswith("_completed")
            or lower == "turn/completed"
            or lower == "turn_completed"
        ):
            return "completed"
        if lower.endswith("/started") or lower.endswith("_started"):
            return "in_progress"

        payload_status = self._status_from_payload(params=params)
        if payload_status is not None:
            return payload_status

        if (
            lower.endswith("/delta")
            or lower.endswith("_delta")
            or lower.endswith("/updated")
            or lower.endswith("_updated")
            or lower.endswith("/progress")
            or lower.endswith("_progress")
            or lower.endswith("/outputdelta")
            or lower.endswith("_outputdelta")
            or lower.endswith("/terminalinteraction")
            or lower.endswith("_terminalinteraction")
            or lower.endswith("/summarytextdelta")
            or lower.endswith("_summarytextdelta")
            or lower.endswith("/summarypartadded")
            or lower.endswith("_summarypartadded")
            or lower.endswith("/textdelta")
            or lower.endswith("_textdelta")
        ):
            return "in_progress"

        if "task_complete" in lower:
            return "completed"

        return "info"

    def _event_ids(self, *, params: dict) -> tuple[Optional[str], Optional[str]]:
        turn_id = _get_nested_id(
            params,
            singular="turn",
            nested_key="turn",
        )
        item_id = _get_nested_id(
            params,
            singular="item",
            nested_key="item",
        )
        return turn_id, item_id

    def _event_key(
        self,
        *,
        method: str,
        turn_id: Optional[str],
        item_id: Optional[str],
    ) -> str:
        lower = method.lower()
        if lower.startswith("codex/event/"):
            suffix = lower[len("codex/event/") :]
            if suffix == "item_started":
                lower = "item/started"
            elif suffix == "item_completed":
                lower = "item/completed"
            elif suffix in ("agent_message_delta", "agent_message_content_delta"):
                lower = "item/agentmessage/delta"
            elif suffix == "task_complete":
                lower = "turn/completed"

        if item_id is not None:
            return f"item:{item_id}"

        if turn_id is not None:
            if lower in ("turn/started", "turn/completed"):
                return f"turn:{turn_id}"
            if lower.startswith("turn/plan/"):
                return f"turn.plan:{turn_id}"
            if lower.startswith("turn/diff/"):
                return f"turn.diff:{turn_id}"
            return f"turn:{turn_id}:{lower}"

        return f"method:{lower}"

    def _event_kind(self, *, method: str, item_type: str) -> str:
        item_kind = self._item_kind_from_type(item_type=item_type)
        if item_kind is not None:
            return item_kind

        lower = method.lower()

        if lower.startswith("turn/plan/"):
            return "plan"
        if lower.startswith("turn/diff/"):
            return "diff"
        if lower.startswith("turn/"):
            return "turn"
        if lower.startswith("item/"):
            return "item"

        if lower.startswith("codex/event/"):
            suffix = lower[len("codex/event/") :]
            if "plan" in suffix:
                return "plan"
            if "diff" in suffix:
                return "diff"
            if "agent_message" in suffix or "message" in suffix:
                return "message"
            if "terminal" in suffix or "command" in suffix:
                return "exec"
            return "codex"

        first = lower.split("/", 1)[0]
        return first if first != "" else "system"

    def _build_status_event(self, *, method: str, params: dict) -> dict:
        status = self._status_from_notification(
            method=method,
            params=params,
        )
        event_type = ".".join(method.split("/"))
        turn_id, item_id = self._event_ids(params=params)
        item = self._extract_item(params=params)
        item_type = _item_type(item)
        kind = self._event_kind(method=method, item_type=item_type)
        correlation_key = self._event_key(
            method=method,
            turn_id=turn_id,
            item_id=item_id,
        )
        headline, detail_lines = self._event_display(
            method=method,
            params=params,
            status=status,
            kind=kind,
            item=item,
        )

        return {
            "type": "agent.event",
            "source": "codex",
            "name": event_type,
            "kind": kind,
            "state": status,
            "method": method,
            "correlation_key": correlation_key,
            "item_id": item_id,
            "item_type": item_type if item_type != "" else None,
            "headline": headline,
            "details": detail_lines,
            "summary": headline
            if isinstance(headline, str) and headline.strip() != ""
            else self._status_summary(
                method=method,
                params=params,
                status=status,
            ),
            "data": self._stringify_status_payload(value=params),
        }

    async def next(
        self,
        *,
        thread_key: str,
        message: str | list[dict],
        developer_instructions: Optional[str | list[str]] = None,
        room: RoomClient,
        toolkits: list[Toolkit],
        tool_adapter: Optional[ToolResponseAdapter] = None,
        event_handler: Optional[Callable[[dict], None]] = None,
        model: Optional[str] = None,
        on_behalf_of: Optional[RemoteParticipant] = None,
    ) -> Any:
        del toolkits
        del tool_adapter
        del on_behalf_of

        resolved_model = self._resolve_model(model=model)
        turn_input = self._normalize_turn_input(message=message)

        await self._ensure_router_started(room=room)

        thread_state = await self._get_thread_state(thread_key=thread_key)
        if thread_state is None:
            raise CodexAppServerError(
                f"codex thread was not opened for thread key '{thread_key}'"
            )
        thread_id = thread_state.thread_id
        context = thread_state.context
        turn_id = None
        turn_queue: Optional[asyncio.Queue[dict]] = None
        final_text = ""
        output_started = False

        try:
            normalized_instructions = self._normalize_developer_instructions(
                developer_instructions=developer_instructions
            )

            turn_params = {
                "threadId": thread_id,
                "input": turn_input,
                "model": resolved_model,
            }
            if normalized_instructions is not None:
                turn_params["collaborationMode"] = {
                    "mode": "default",
                    "settings": {
                        "model": resolved_model,
                        "developer_instructions": normalized_instructions,
                    },
                }

            turn_result = await self._session.request(
                method="turn/start",
                params=turn_params,
            )

            turn_id = self._extract_turn_id(turn_result)
            await self._track_active_turn(
                thread_key=thread_key,
                thread_id=thread_id,
                turn_id=turn_id,
            )
            turn_queue = await self._register_turn_queue(
                thread_id=thread_id,
                turn_id=turn_id,
            )

            while True:
                notification = await self._next_turn_notification(turn_queue=turn_queue)

                method = notification.get("method")
                params = notification.get("params") or {}
                if not isinstance(params, dict):
                    continue

                if event_handler is not None and self._should_emit_status_event(
                    method=method
                ):
                    event_handler(
                        self._build_status_event(
                            method=method,
                            params=params,
                        )
                    )

                if method in ("item/started", "codex/event/item_started"):
                    item = self._extract_item(params=params)
                    if _is_agent_message(item) and not output_started:
                        output_started = True
                        if event_handler is not None:
                            event_handler({"type": "response.content_part.added"})

                elif method in (
                    "item/agentMessage/delta",
                    "codex/event/agent_message_delta",
                ):
                    delta = self._extract_delta(params=params)
                    if delta != "":
                        if not output_started:
                            output_started = True
                            if event_handler is not None:
                                event_handler({"type": "response.content_part.added"})

                        final_text += delta
                        if event_handler is not None:
                            event_handler(
                                {"type": "response.output_text.delta", "delta": delta}
                            )

                elif method in ("item/completed", "codex/event/item_completed"):
                    item = self._extract_item(params=params)
                    if _is_agent_message(item):
                        completed_text = _get_nested_text(item)
                        if completed_text != "":
                            final_text = completed_text

                        if event_handler is not None:
                            event_handler(
                                {
                                    "type": "response.output_text.done",
                                    "text": final_text,
                                }
                            )

                elif method == "codex/event/task_complete":
                    msg = params.get("msg")
                    if isinstance(msg, dict):
                        last_message = msg.get("last_agent_message")
                        if isinstance(last_message, dict):
                            completed_text = _get_nested_text(last_message)
                            if completed_text != "":
                                final_text = completed_text

                elif method == "turn/completed":
                    turn = params.get("turn")
                    if isinstance(turn, dict):
                        status = turn.get("status")
                        if status == "failed":
                            error = turn.get("error")
                            raise CodexAppServerError(
                                f"codex turn failed: {error or 'unknown error'}"
                            )
                    break

            if final_text == "":
                raise CodexAppServerError("codex app-server returned no text output")

            context.append_assistant_message(final_text)

            return final_text

        finally:
            if turn_id is not None:
                await self._untrack_active_turn(
                    thread_key=thread_key,
                    thread_id=thread_id,
                    turn_id=turn_id,
                )
                await self._unregister_turn_queue(
                    thread_id=thread_id,
                    turn_id=turn_id,
                )
