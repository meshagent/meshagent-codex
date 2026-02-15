# meshagent-codex

`meshagent-codex` adds a Codex app-server backed `CodexChatBot`
that reuses MeshAgent chat thread handling and control messages.

## Included

- `CodexChatBot`: chat agent backed by `codex app-server`

## Example

```python
from meshagent.api.services import ServiceHost
from meshagent.codex import CodexChatBot

service = ServiceHost()


@service.path("/agent")
class MyCodexAgent(CodexChatBot):
    def __init__(self):
        super().__init__(
            name="meshagent.codex-chatbot",
            title="codex chatbot",
            description="chatbot powered by codex app-server",
            rules=["You are a concise assistant."],
            model="codex-mini-latest",
        )
```

By default, the backend launches Codex via `codex app-server`.

You can override transport with environment variables:

- `MESHAGENT_CODEX_COMMAND` to change the launch command.
- `MESHAGENT_CODEX_WS_URL` to connect to an existing Codex app-server websocket instead of launching a local process.
