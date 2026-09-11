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

- Upstream: `openai/codex` tag `rust-v0.153.4`
- Upstream commit: `3d2ee51ca2d5db578f328aa75e20aa22c0197c9a`
- Distribution version: `0.153.4-meshagent.1`
- Default state directory: `~/.meshagent/codex`
- Installed command name: `codex`

`MESHAGENT_CODEX_HOME` is the only state-root override. The distribution does
not inherit `CODEX_HOME`, so an official Codex process and a MeshAgent Codex
process cannot select the same root through ambient configuration.

Daemon resources are namespaced inside that root:

- control socket: `codex-app-server/control.sock`;
- lifecycle state: `codex-app-server-daemon/`; and
- managed executables: `packages/standalone/current/codex` and
  `packages/standalone/current/codex-code-mode-host`.

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

## Install on Windows using local WinGet manifests

You can install a published Windows build directly from its GCS manifests,
including `.life` builds that are not submitted to the public WinGet repository.
You need PowerShell and WinGet (`winget --version`), but no GitHub token, Google
Cloud CLI, or access to this private source repository. Use a release whose
Windows installation tests have passed.

1. Open PowerShell **as administrator** and enable local manifests once:

   ```powershell
   winget settings --enable LocalManifestFiles
   ```

   Close that window and use a regular PowerShell window for the remaining steps.
   See Microsoft's [local manifest installation documentation](https://learn.microsoft.com/en-us/windows/package-manager/winget/install#local-install).

2. Set the bucket and MeshAgent release version supplied with the release,
   replacing the placeholders below. The version is the GCS folder name, such
   as `0.52.3`, rather than the upstream Codex version or full archive filename.
   Use the bucket for the intended environment (`.life` or `.com`).

   ```powershell
   $Bucket = "<build-bucket>"
   $Version = "<meshagent-version>"
   $PackageId = "MeshAgent.EnterpriseCodex"
   $BaseUrl = "https://storage.googleapis.com/$Bucket/$Version/winget-manifests"
   $ManifestDirectory = Join-Path $PWD "winget-manifests-$Version"
   New-Item -ItemType Directory -Path $ManifestDirectory -Force | Out-Null

   $Files = @(
       "$PackageId.yaml"
       "$PackageId.installer.yaml"
       "$PackageId.locale.en-US.yaml"
   )
   foreach ($File in $Files) {
       Invoke-WebRequest -UseBasicParsing -Uri "$BaseUrl/$File" `
           -OutFile (Join-Path $ManifestDirectory $File) -ErrorAction Stop
   }
   ```

   All three YAML files must be together in this directory. If the release uses
   a custom `CODEX_WINGET_PACKAGE_ID`, use that value for `$PackageId`.
   Alternatively, download and extract the release workflow's `winget-manifests`
   Actions artifact, then set `$ManifestDirectory` to the directory containing
   its three YAML files. Downloading that artifact requires GitHub repository
   access; downloading the public GCS files does not.

3. Validate the manifests and install for your Windows user:

   ```powershell
   winget validate --manifest $ManifestDirectory
   if ($LASTEXITCODE -ne 0) { throw "WinGet manifest validation failed." }

   $InstallDirectory = Join-Path $env:LOCALAPPDATA "Programs\EnterpriseCodex"
   winget install --manifest $ManifestDirectory --scope user --location $InstallDirectory
   if ($LASTEXITCODE -ne 0) { throw "WinGet installation failed." }
   ```

   WinGet selects x64 or ARM64 for your machine, downloads the matching ZIP from
   GCS, and verifies its checksum. It installs the CLI, Code Mode host, and
   Windows sandbox helpers together. The manifest already specifies the package
   version; you do not need to download or extract the ZIP manually.

4. Check and launch the installed executable from the same PowerShell window:

   ```powershell
   & (Join-Path $InstallDirectory "codex.exe") --version
   & (Join-Path $InstallDirectory "codex.exe")
   ```

   The full path ensures you run MeshAgent's executable if another `codex` is
   already on `PATH`. In a new PowerShell window, the path is
   `& "$env:LOCALAPPDATA\Programs\EnterpriseCodex\codex.exe"`.
   The executable reports the patched Codex version; this can differ from the
   MeshAgent release folder and WinGet package version.

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

`build` builds and stages both `dist/codex` and `dist/codex-code-mode-host`
(with `.exe` suffixes on Windows) inside the external materialized checkout.
The host includes the V8 runtime. It does not put generated source or binaries
in this repository.

`build --cli-only` runs `cargo build --locked --release -p codex-cli --bin codex`
and stages only the CLI, for local builds that do not need JavaScript Code Mode.

`build --target x86_64-apple-darwin` cross-compiles with the native compiler and
stages both executables from Cargo's `target/x86_64-apple-darwin/release`
directory. Install that target for the pinned Rust toolchain first. When using
prebuilt V8, both its archive and Rust bindings must match the requested target.

The release workflow builds and packages both the CLI and `codex-code-mode-host`,
plus the `bwrap` sandbox helper on Linux and two sandbox helpers on Windows.
It downloads and verifies OpenAI Codex's prebuilt V8 archive and matching Rust
bindings for the locked V8 version, following upstream's `setup-rusty-v8` action.
Homebrew and WinGet install the host beside the CLI so JavaScript Code Mode is
available without a separate installation. macOS Intel builds cross-compile on
Apple Silicon, and Windows x64 builds cross-compile on the Windows ARM runner.
CI verifies the architecture of every cross-compiled executable, including the
Windows sandbox helpers, and runs their runtime checks on native x64 runners.
Native builds also smoke-test JavaScript execution and ICU number formatting
using the host extracted from each release archive.
Separate native jobs then install the GCS packages through a temporary Homebrew
tap or local WinGet manifests, exercise the installed executables, and uninstall
them. Every selected target must pass before package metadata reaches either
repository. WinGet submission is enabled by default; set `submit-winget: false`
to build and test candidate manifests without submitting them. Windows builds
always save the candidate manifests to the `winget-manifests` Actions artifact
and `gs://<build-bucket>/<meshagent-version>/winget-manifests/`, regardless of
submission. The `.life` release workflow disables WinGet submission; `.com`
enables it.

The Homebrew tap publishes both `enterprise-codex` for the latest release and
`enterprise-codex@<meshagent-version>` for a specific release, starting with
releases published by this workflow. For example, after adding the tap:

```sh
brew install enterprise-codex@0.52.3
"$(brew --prefix enterprise-codex@0.52.3)/bin/codex" --version
```

Versioned formulas are keg-only, so they can coexist with the latest formula.
To use one as `codex` in the current shell, prepend its `bin` directory to `PATH`:

```sh
export PATH="$(brew --prefix enterprise-codex@0.52.3)/bin:$PATH"
```

`install-managed` builds and copies both binaries to the daemon-managed path. It
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
build [--cli-only] [--target <triple>]
                    Build and stage the CLI and Code Mode host (or only the CLI)
install-managed [home]
                    Build and install both daemon-managed binaries
test-patched        Test every crate touched by the patch series
test [arguments]    Run the upstream test suite, forwarding nextest arguments
run [arguments]     Run the debug CLI from the materialized source
verify              Format-check, check, test patched crates, build, and smoke-test
```

For patch updates, upstream releases, CI guidance, and release/legal
requirements, read [`docs/maintaining.md`](docs/maintaining.md).
