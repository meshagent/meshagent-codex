# meshagent-codex

`meshagent-codex` provides a Meshagent `AgentSupervisor` implementation backed
by the vendored Codex Python app-server SDK.

## Included

- `CodexAgentSupervisor`: creates one `CodexAgentProcess` per Meshagent thread.
- `CodexAgentProcess`: sends `TurnStart`, `TurnSteer`, and `TurnInterrupt`
  messages to Codex and emits standard Meshagent `AgentMessage` events.
- Vendored `openai_codex`: the upstream Codex Python SDK package, including
  generated app-server models and the async client.

## Example

```python
from meshagent.codex import AppServerConfig, CodexAgentSupervisor

supervisor = CodexAgentSupervisor(
    participant=room.local_participant,
    config=AppServerConfig(cwd="/workspace"),
    default_model="gpt-5.5",
)
```

The default `AppServerConfig` uses the pinned `openai-codex-cli-bin` runtime
dependency. Set `AppServerConfig.codex_bin` to launch a specific Codex binary.
