from typing import TYPE_CHECKING, Any

from .process import CodexAgentProcess
from .supervisor import DEFAULT_CODEX_MODEL, CodexAgentSupervisor
from .thread_storage import CodexThreadStorage, CodexThreadStorageRepository
from .vendor.openai_codex.client import AppServerConfig
from .vendor.openai_codex.errors import AppServerError as CodexAppServerError
from .version import __version__

if TYPE_CHECKING:
    from .vendor.openai_codex.async_client import AsyncAppServerClient


def __getattr__(name: str) -> Any:
    if name == "AsyncAppServerClient":
        from .vendor.openai_codex.async_client import AsyncAppServerClient

        return AsyncAppServerClient
    raise AttributeError(f"module 'meshagent.codex' has no attribute {name!r}")


__all__ = [
    "__version__",
    "AppServerConfig",
    "AsyncAppServerClient",
    "CodexAgentProcess",
    "CodexAgentSupervisor",
    "CodexAppServerError",
    "CodexThreadStorage",
    "CodexThreadStorageRepository",
    "DEFAULT_CODEX_MODEL",
]
