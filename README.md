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

By default, `AppServerConfig` uses `codex` on `PATH`. Set
`AppServerConfig.codex_bin` to launch a specific Codex binary. If you install
`openai-codex-cli-bin` separately on a supported platform, it can still be used
as a fallback runtime.

## MeshAgent Codex distribution

The independently buildable Rust distribution lives in [`distribution/`](distribution/README.md).
It pins an OpenAI Codex release and maintains an ordered patch series without
checking the upstream Rust source into this repository. The Python integration
in this package remains separate from that source overlay.
