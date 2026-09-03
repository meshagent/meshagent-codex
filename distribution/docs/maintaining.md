# Maintaining MeshAgent Codex

## Non-vendoring rule

Never commit OpenAI Codex source files, a Git submodule, a Cargo target tree, or
a materialized checkout to the MeshAgent repository. The durable inputs are the
immutable upstream identity in `upstream.lock` and the ordered files listed in
`patches/series`. CI and developers reconstruct the source outside the repo.

Before changing a patch, read the upstream `AGENTS.md` in the materialized
checkout. Upstream instructions are part of the contribution contract for that
source version. At minimum, run `just fmt` when its complete toolchain is
available, use `just test -p <crate>` rather than `cargo test`, and inspect and
accept only intentional TUI snapshot changes. The repository-level requirement
to run `cargo fmt` and `cargo check` also applies.

## Patch policy

Keep patches small, ordered, and independently explainable. Prefer adding a new
patch for a new concern. Amend an existing patch only while that concern is
still under review. Every patch must:

- apply to the exact commit in `upstream.lock`;
- preserve upstream behavior unless the distribution intentionally changes it;
- include tests and snapshots affected by the change;
- use Rust-native, typed abstractions; and
- avoid secrets, customer policy, generated build output, and machine paths.

The initial patch changes distribution identity and the default state root:

- Cargo version `0.152.1-meshagent.1`;
- CLI branding, help, and legal attribution;
- TUI product name and snapshots; and
- the default configuration root from `~/.codex` to `~/.meshagent/codex`.

The runtime-isolation patch makes the boundary explicit:

- only `MESHAGENT_CODEX_HOME` overrides the MeshAgent state root; ambient
  `CODEX_HOME` is ignored;
- Codex app-server socket and lifecycle state live beneath the isolated
  MeshAgent state root while retaining their `codex-app-server` component
  names;
- the daemon runs the managed `codex` executable without renaming any upstream
  commands; and
- OpenAI's standalone updater is disabled so distribution updates remain under
  MeshAgent packaging control.

The MeshAgent authentication patch adds the distribution's provider behavior
as one reviewable change:

- MeshAgent OAuth is the default `codex login` path, with an optional
  `--api-url` and shared MeshAgent CLI account storage;
- explicit `--provider=openai` login and logout preserve independent OpenAI
  credentials, while an unqualified logout clears both providers;
- saved MeshAgent accounts can be switched with `codex login switch`;
- projects can be listed and activated with `codex project`, or selected in the
  TUI with `/project`; and
- MeshAgent API routing, authentication, and project headers are resolved for
  each request so switching does not require an app-server restart.

## Editing a patch

1. Materialize the current distribution into a disposable external checkout.
2. Confirm the checkout is clean and read its `AGENTS.md`.
3. Edit the materialized Rust source and run the required formatter/checks/tests.
4. Amend or create commits in the disposable checkout.
5. Export each distribution commit with:

   ```sh
   git format-patch --no-signature --zero-commit -1 --stdout
   ```

6. Replace the corresponding file under `patches/`, keep `patches/series` in
   application order, and increment `patchset` in `upstream.lock` whenever patch
   content changes.
7. Point `MESHAGENT_CODEX_WORKTREE` at a new empty path and run
   `./codex-distribution verify`. This final run must reconstruct from only the
   files committed here; do not validate solely in the editing checkout.

Patch files are mail-formatted commits because `git am --3way` preserves clear
commit boundaries and gives useful conflict context during an upstream update.

## Updating upstream

1. Select an upstream release tag, resolve its peeled commit, and review release
   notes plus upstream `AGENTS.md` changes.
2. Update `tag` and `commit` together in `upstream.lock`. Never pin only a moving
   branch or tag name.
3. Set `distribution_version` to `<upstream-version>-meshagent.<revision>` and
   update the workspace version in the branding patch.
4. Apply the existing series to the new commit in a disposable checkout. Resolve
   conflicts there, preserving the intent of each patch rather than merely
   making hunks apply.
5. Regenerate the patches, increment `patchset`, and run full reconstruction.
6. Run focused tests for all touched crates. Request approval before running the
   complete upstream test suite if upstream `AGENTS.md` still requires it.
7. Build every supported release target and smoke-test `--version`, `--help`, and
   a clean-home startup before publishing artifacts.

If a patch becomes broadly useful and is not MeshAgent-specific, prefer sending
it upstream and deleting it from this series after the pinned release contains
the change.

## CI and releases

CI should use a fresh external worktree and execute:

```sh
./codex-distribution verify
```

Release packaging should take only the staged `dist/codex` binary (or
the platform equivalent) plus required licenses and notices. Installers that
enable the daemon must also place that same binary at
`MESHAGENT_CODEX_HOME/packages/standalone/current/codex` (or the
platform equivalent); `./codex-distribution install-managed` provides the
reference layout. Preserve the
upstream Apache-2.0 `LICENSE` and `NOTICE`, include notices for dependencies, and
retain the visible statement that MeshAgent Codex is based on OpenAI Codex but
is not an official OpenAI distribution. Have counsel review trademark and
distribution requirements before public release.

Managed enterprise policy belongs in deployment/configuration layers, not in a
customer-specific source patch. Keep credentials and organization identifiers
out of this overlay.
