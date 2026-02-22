from typing import TYPE_CHECKING, Any

from .app_server import CodexAppServerError
from .version import __version__

if TYPE_CHECKING:
    from .chatbot import CodexChatBot
    from .task_runner import CodexTaskRunner
    from .worker import CodexWorker


def __getattr__(name: str) -> Any:
    if name == "CodexChatBot":
        from .chatbot import CodexChatBot

        return CodexChatBot
    if name == "CodexTaskRunner":
        from .task_runner import CodexTaskRunner

        return CodexTaskRunner
    if name == "CodexWorker":
        from .worker import CodexWorker

        return CodexWorker
    raise AttributeError(f"module 'meshagent.codex' has no attribute {name!r}")


__all__ = [
    "__version__",
    "CodexAppServerError",
    "CodexChatBot",
    "CodexTaskRunner",
    "CodexWorker",
]
