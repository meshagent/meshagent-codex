# MeshAgent Codex distribution

MeshAgent Codex is an enterprise distribution based on the open-source OpenAI
Codex Rust codebase. It is maintained by MeshAgent and is not an official
OpenAI distribution.

This directory is a source overlay, not a forked source tree. The repository
contains only:

- `upstream.lock`, which pins an immutable upstream tag and commit;
- `patches/series`, which defines patch application order;
- the patches themselves; and
- documentation and the `codex-distribution` helper.

The upstream checkout, build outputs, and Cargo cache stay outside this
repository. Do not copy a materialized OpenAI Codex checkout into this tree.

## Current distribution

- Upstream: `openai/codex` tag `rust-v0.152.1`
- Upstream commit: `5adb68a49933ae446bf11935662c83dba55a0804`
- Distribution version: `0.152.1-meshagent.1`
- Default state directory: `~/.meshagent/codex`
- Installed command name: `codex`

`MESHAGENT_CODEX_HOME` is the only state-root override. The distribution does
not inherit `CODEX_HOME`, so an official Codex process and a MeshAgent Codex
process cannot select the same root through ambient configuration.

Daemon resources are namespaced inside that root:

- control socket: `codex-app-server/control.sock`;
- lifecycle state: `codex-app-server-daemon/`; and
- managed executable: `packages/standalone/current/codex`.

Automatic use of OpenAI's standalone updater is disabled. MeshAgent packaging
owns updates to the managed executable.

## Authentication and projects

MeshAgent is the default model provider and the default login flow. MeshAgent
credentials, saved accounts, and the active project use the same
`~/.meshagent/settings.json` file as the MeshAgent Rust CLI. OpenAI credentials
remain separate, so both providers can be signed in at the same time.

```sh
codex login                              # MeshAgent OAuth login
codex login --api-url https://example   # MeshAgent login for another API
codex login --provider=openai            # OpenAI/ChatGPT login
codex login status                       # Status for all providers
codex login switch [account]             # Switch saved MeshAgent account
codex project list                       # List MeshAgent projects
codex project activate [project]         # Activate by ID, key, or name
codex logout                             # Log out of every provider
codex logout --provider=openai            # Log out of OpenAI only
```

Within the TUI, `/project` opens the MeshAgent project picker. Project and
account changes are resolved for subsequent requests without restarting Codex.

## Prerequisites

Materialization needs Git and network access to GitHub. Rust commands use the
toolchain pinned by upstream. Tests require `just` and `cargo-nextest`; the full
upstream formatter may also require the non-Rust tools documented by upstream.

## Quick start

Run commands from this directory:

```sh
./codex-distribution materialize
./codex-distribution check
./codex-distribution build
./codex-distribution install-managed
./codex-distribution test-patched
```

`materialize` checks out the exact locked commit in a temporary external work
area and applies each patch with `git am --3way`. It refuses to reuse a checkout
whose upstream commit or patch digest differs. Set `MESHAGENT_CODEX_WORKTREE` to
choose an explicit external path.

`build` writes a release binary named `codex` to `dist/` inside the
external materialized checkout. It does not put generated source or binaries in
this repository.

`install-managed` builds and copies that binary to the daemon-managed path. It
uses `MESHAGENT_CODEX_HOME` when set, otherwise `~/.meshagent/codex`; pass an
absolute home path as its optional argument to stage an isolated package root.

Useful commands:

```text
materialize [path]  Fetch the locked source and apply the patch series
path                Print the default or configured worktree path
status              Show lock identity and materialized checkout status
format              Run the upstream formatter
check-format        Check Rust formatting
check               Run locked checks for patched crates
build               Build and stage the release binary
install-managed [home]
                    Build and install the daemon-managed binary
test-patched        Test every crate touched by the patch series
test [arguments]    Run the upstream test suite, forwarding nextest arguments
run [arguments]     Run the debug CLI from the materialized source
verify              Format-check, check, test patched crates, build, and smoke-test
```

For patch updates, upstream releases, CI guidance, and release/legal
requirements, read [`docs/maintaining.md`](docs/maintaining.md).
