# Maintaining MeshAgent Codex

## Non-vendoring rule

Never commit OpenAI Codex source files, a Git submodule, a Cargo target tree, or
a materialized checkout to the MeshAgent repository. The durable inputs are the
immutable upstream identity in `upstream.lock` and the ordered files listed in
`patches/series`. CI and developers reconstruct the source outside the repo.
The distribution's `.gitattributes` preserves LF line endings on every platform
so Windows checkouts do not add carriage returns to patch names or contents.

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

- Cargo version `0.160.0-meshagent.1`;
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
- OpenAI's startup update checks, standalone updater, `codex update`, and daemon
  package updates are disabled so updates remain under MeshAgent packaging
  control. Check for new installation and update entry points when rebasing.

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

The helper's default `build` and `install-managed` commands include both the
CLI and `codex-code-mode-host`. The publishing workflow uses this full build.
Before compiling, it downloads the exact locked V8 version's sandbox-enabled
archive and Rust bindings from OpenAI Codex's `rusty-v8-v<version>` release and
verifies both checksums. It exports `RUSTY_V8_ARCHIVE` and
`RUSTY_V8_SRC_BINDING_PATH`, following the pinned upstream
`.github/actions/setup-rusty-v8/action.yml`; V8 is linked from those prebuilt
artifacts rather than compiled from source. Review that action and artifact
profile when updating the upstream pin.

Release packaging takes the staged `dist/codex` and `dist/codex-code-mode-host`
binaries (with `.exe` suffixes on Windows), plus required licenses and notices.
Linux archives include `codex-resources/bwrap` with its license and source; the
workflow builds and hashes that helper before building the CLI so upstream's
runtime integrity check has the correct digest. Windows archives also include
`codex-command-runner.exe` and `codex-windows-sandbox-setup.exe`, built from
`codex-windows-sandbox`, because the CLI needs them for sandboxed commands.
The Code Mode host includes V8 and must remain beside the CLI. Installers that
enable the daemon must place the CLI under
`MESHAGENT_CODEX_HOME/packages/standalone/current/`;
`./codex-distribution install-managed` provides the reference layout and also
installs the Code Mode host beside the CLI. Preserve the upstream Apache-2.0
`LICENSE` and `NOTICE`, include notices for dependencies, and
retain the visible statement that MeshAgent Codex is based on OpenAI Codex but
is not an official OpenAI distribution. Have counsel review trademark and
distribution requirements before public release.

On Windows, check MeshAgent authentication against the built CLI with:

```powershell
$env:RUN_MESHAGENT_CLOUD_SMOKE = "1"
python smoke-windows-auth.py C:\path\to\codex.exe
```

The check removes `HOME` and uses an empty temporary `USERPROFILE`.
Authentication must resolve the native Windows home directory and report that
the user is signed out. Version and help checks alone do not exercise this path.

The reusable `.github/workflows/publish-enterprise-codex.yml` workflow fetches
the locked source, applies the ordered patches, builds the CLI, Code Mode host,
and required platform sandbox helpers, and checks CLI version/help. It archives
the executables with `LICENSE` and `NOTICE`, extracts each archive, and checks
CLI startup plus the host's protocol handshake, JavaScript execution, ICU number
formatting, and session shutdown on native builds. Cross-compiled archives get
architecture checks here and runtime checks in the native installation job.
It uploads archives to GCS
and retains copies as GitHub Actions artifacts for package generation and testing.
Package managers use the public GCS URLs because this source repository is private.

The workflow reads `meshagent-version` using the existing
`.github/actions/meshagent-get-cli-version` action. GCS folders use that version,
and archive names plus Homebrew/WinGet package versions use
`<meshagent-version>-meshagent.<upstream-version>`. The upstream version comes
from the `rust-v<version>` tag in `upstream.lock`. For example:

```text
gs://meshagent-enterprise-codex-builds/0.52.2/enterprise-codex-0.52.2-meshagent.0.153.4-linux-arm64.tar.gz
```

Windows uses the same layout with `windows-x86_64.zip` or `windows-arm64.zip`.
The executable still reports the patched `distribution_version` from
`upstream.lock`; CI checks that version separately from the package version.
When generating Homebrew metadata, pass it as `--cli-version` so the formula's
test checks the executable's actual version.

Both macOS builds use the Apple Silicon larger runner, `macos-26-xlarge`.
The Intel build uses the native ARM Rust compiler with
`./codex-distribution build --target x86_64-apple-darwin`, the pinned toolchain's
Intel standard library, and the macOS SDK. Its prebuilt V8 archive and bindings
also target `x86_64-apple-darwin`. The helper stages the CLI and host from Cargo's
target-specific output directory. CI verifies their Mach-O architecture without
executing them on ARM; the Intel installation job stays on `macos-26-large`
and runs all installed runtime checks for both Homebrew formulas. Release
optimization settings are unchanged so build duration can be compared with the
previous native Intel builds.

Both Windows builds use `meshagent-runner-windows-arm`. The x64 build uses the
native ARM Rust compiler with `--target x86_64-pc-windows-msvc`, the pinned
toolchain's x64 standard library, and the ARM-hosted MSVC tools targeting x64.
V8's prebuilt archive and bindings also target x64. The CLI, Code Mode host, and
both sandbox helpers are staged from the target-specific Cargo output directory;
CI verifies all four PE architectures before uploading the ZIP. Runtime and
WinGet installation tests stay on `meshagent-runner-windows-x64` for x64 and
`meshagent-runner-windows-arm` for ARM64. Each architecture gets a separate ZIP.
Release optimization settings are unchanged for the timing comparison.

Homebrew publication requires all four macOS and Linux architectures. The formula installs
the CLI and host together under `libexec`, with a `codex` command symlink and
checksums computed from the built artifacts. Each release publishes two formulas:
`enterprise-codex.rb` tracks the latest release, and
`enterprise-codex@<meshagent-version>.rb` provides that specific MeshAgent release.
Both use the same full package version, archive URLs, and checksums. Generate the
versioned form with `--homebrew-version <meshagent-version>` and the matching
output filename. It uses Homebrew's versioned class name and
`keg_only :versioned_formula`, so it does not replace the latest formula's
global `codex` command. Its notices live under `share/<formula-name>`.

Publication copies both tested formulas into the tap and preserves formulas for
other releases. It does not backfill earlier releases. Re-running the same
MeshAgent version updates that version's formula and GCS objects; use a new
MeshAgent release number for each release that must remain separately installable,
and retain its GCS archives. See the distribution README for install and PATH
selection commands.

Windows builds generate WinGet ZIP/portable manifests with both x64 and
ARM64 installers, each pointing to its architecture's GCS ZIP with a checksum
verified against the public download.
The manifests expose `codex.exe` as the `codex` command. WinGet retains
the entire ZIP contents, including the host and sandbox helpers beside the CLI;
only the CLI needs a portable command alias. The workflow validates and uploads
the YAML files whenever `build-windows: true`, regardless of `submit-winget`.
Candidate YAML files are retained as the `winget-manifests` Actions artifact
and at `gs://<build-bucket>/<meshagent-version>/winget-manifests/`, even if an
installation test fails. The GCS folder uses the same MeshAgent release version
as the package archives.
Check the installation job results before manually installing a candidate.
WinGet selects the installer matching the Windows machine's architecture.
Both Windows archives are required to generate the manifests. Public submission
requires `build-windows: true` and `submit-winget: true`.

### Native installation tests and publication gate

The release workflow runs in this order:

1. Run the fast distribution-tooling tests in `prepare`.
2. Build each selected target, verify its archive (native runtime checks or
   cross-build architecture checks), and upload it to GCS and GitHub Actions.
3. Generate the candidate latest and versioned Homebrew formulas and WinGet manifests once, using
   checksums from those Actions archives and download URLs pointing at GCS.
   Save the metadata as Actions artifacts and also copy the WinGet YAML files to GCS.
4. Run `test-installation` on a separate native runner for every selected target.
5. Only after the entire installation matrix succeeds, copy both tested formulas
   into the Homebrew tap when both `build-macos` and `build-linux` are enabled,
   and submit the tested WinGet manifests when both `build-windows` and
   `submit-winget` are enabled. Publication downloads the same metadata
   artifacts; it does not regenerate them.

Installation jobs use disposable native runners with no existing Enterprise
Codex installation. They do not materialize upstream source, restore Cargo
caches, or compile Rust. The current runner images still contain developer
tools; these jobs test package installation, not compatibility with every
minimal OS image or older OS version.

On macOS and Linux, `test-install-homebrew.sh` creates a temporary local tap and
installs both formulas from GCS using a fresh Homebrew download cache. It runs
`brew test` and the installed runtime checks for each formula, verifies that
installing the versioned formula preserves the latest formula's global `codex`
command, then uninstalls both packages and the tap. The versioned smoke test
selects its command by prepending that formula's `bin` directory to `PATH`.
The `homebrew-formula` Actions artifact contains both candidate Ruby files.
Mac-only or Linux-only test runs generate formulas limited to the selected
operating system using `--homebrew-os`. Both architectures are included for each
selected operating system. The generator also supports `--homebrew-platform`
for manual generation. Publishing the Homebrew tap requires all four Unix targets.

On Windows, `test-install-winget.ps1` validates the local manifests, enables
local manifest installation, and installs from GCS into a temporary directory
with spaces using user scope. It lets WinGet choose the architecture, then
uninstalls the package and checks that its executable and command alias were
removed. These jobs do not submit to the WinGet repository or use `WINGET_TOKEN`.
Failed Windows jobs retain WinGet diagnostic logs as Actions artifacts.

Both paths run `smoke-installed.py` with `RUN_MESHAGENT_CLOUD_SMOKE=1`. It checks
the installed executables and notices against the original archive, verifies
native executable architecture, checks that `codex` resolves to this
installation, verifies version and branding, and runs the Code Mode protocol,
JavaScript, and ICU smoke test. With isolated Codex state and no inherited build
paths or credentials, it runs a local command through a read-only sandbox.
Windows also runs `smoke-windows-auth.py` with `HOME` unset. The Windows sandbox
check uses restricted-token mode; elevated sandbox provisioning is not covered.
These checks require no model-provider login or remote application service.

The smoke workspace and isolated Codex state live in a temporary directory under
the runner's home, outside system temp paths. This allows Codex to create its
PATH helpers and avoids sandbox exceptions for writable temp directories. On
Windows, the fixture grants the runner's own user SID inheritable read/execute
access: Python 3.13's private directory ACL can otherwise depend on administrator
ownership that the restricted token cannot use. The check requires a sandboxed
read to succeed and an attempted overwrite to fail without changing the file.
Windows sandbox logs are printed before cleanup if command execution fails.

Linux installation jobs follow upstream Codex's CI namespace setup, enabling
unprivileged user namespaces and disabling the AppArmor restriction on those
namespaces when present. This configuration is scoped to disposable CI runners;
it is not applied by the packages or recommended as a production host policy.

Use GitHub Actions' **Re-run failed jobs** to retry a failed installation and
its dependent publication jobs without rebuilding successful targets. The
original build and manifest artifacts must still be retained, and the GCS
objects must still match their checksums. Workflow or test-code changes require
a new run to use the changed code.

The `meshagent-publish-life.yml` and `meshagent-publish-com.yml` release workflows
call this workflow with their respective `meshagent-life` and `meshagent-com`
environments. Both enable macOS, Linux, and Windows builds for x64 and ARM64,
but `.life` sets `submit-winget: false` and `.com` sets `submit-winget: true`.
Both upload the WinGet manifests to GCS and run the native installation tests.
Enterprise Codex runs after `publish-brew-items` so
the two publishers do not push competing updates to the same Homebrew tap.
Both environments need the publishing configuration listed below;
only `.com` needs `WINGET_TOKEN` for public WinGet submission.

The three build flags each select both architectures and their native
installation tests. All three default to true in the reusable workflow; set a
platform's flag to false to disable its builds. At least one platform must be
enabled. WinGet submission also defaults to true; callers can set
`submit-winget: false` to build, upload, and installation-test the Windows
packages and retain their manifests as Actions artifacts without submitting
them to the public WinGet repository. The manifests are also uploaded to GCS
whether or not submission is enabled.

Download and extract the `winget-manifests` Actions artifact to test locally.
Alternatively, download all three YAML files from
`gs://<build-bucket>/<meshagent-version>/winget-manifests/` into a local
`winget-manifests` directory.
Enable local manifests once
from an administrator PowerShell with `winget settings --enable LocalManifestFiles`,
then run `winget install --manifest .\winget-manifests` from the extracted
artifact's parent directory. Installation downloads the matching ZIP from GCS.

Public submission runs after installation tests pass: `submit-winget: true`,
with `build-windows: true`, submits the manifests using Microsoft's `wingetcreate`.
This supports both the first package submission and later versions; installation
through the community WinGet source becomes available after Microsoft accepts
the PR.

Configure these values in the selected GitHub environment:

- `MESHAGENT_ENTERPRISE_CODEX_BUILD_BUCKET`: a bucket whose release objects are
  publicly readable, with write access for the configured GCP workload identity;
- `GCP_WORKLOAD_PROVIDER` and `GCP_WORKLOAD_SERVICE_ACCOUNT`;
- `ENTERPRISE_CODEX_HOMEBREW_REPOSITORY` (or `THE_GITHUB_ORG` for the
  `<org>/homebrew-meshagent` default), and `REPO_TOKEN` with write access to the tap;
- `CODEX_WINGET_PACKAGE_ID` (default `MeshAgent.EnterpriseCodex`);
- `WINGET_TOKEN`, a classic GitHub token with `public_repo` scope for WinGet PRs,
  required only when `submit-winget: true`.

Validate distribution tooling locally with:

```sh
bash -n codex-distribution
shellcheck codex-distribution
python3 -m unittest discover -s tests
```

Managed enterprise policy belongs in deployment/configuration layers, not in a
customer-specific source patch. Keep credentials and organization identifiers
out of this overlay.
