import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

OVERLAY = Path(__file__).resolve().parents[1]


class DistributionBuildTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.source = self.root / "source checkout"
        self.source.mkdir()
        subprocess.run(["git", "init", "--quiet", str(self.source)], check=True)
        (self.source / "codex-rs").mkdir()
        lock = (OVERLAY / "upstream.lock").read_text()
        commit = re.search(r'^commit = "(.+)"$', lock, re.MULTILINE)[1]
        patchset = re.search(r"^patchset = (\d+)$", lock, re.MULTILINE)[1]
        digest_input = "series\n"
        for patch in (OVERLAY / "patches/series").read_text().splitlines():
            if not patch or patch.startswith("#"):
                continue
            digest = subprocess.check_output(
                ["git", "hash-object", str(OVERLAY / "patches" / patch)], text=True
            ).strip()
            digest_input += f"{patch} {digest}\n"
        digest = subprocess.check_output(
            ["git", "hash-object", "--stdin"], input=digest_input, text=True
        ).strip()
        (self.source / ".git/meshagent-codex-distribution-state").write_text(
            f"upstream={commit}\npatchset={patchset}\npatch_digest={digest}\n"
        )
        commands = self.root / "bin"
        commands.mkdir()
        cargo = commands / "cargo"
        cargo.write_text(
            """#!/usr/bin/env bash
set -euo pipefail
printf '%s\\n' "$@" > "$CARGO_LOG"
if [[ "${FAIL_BUILD:-0}" == 1 ]]; then exit 42; fi
target_root="${CARGO_TARGET_DIR:-$PWD/target}"
args=("$@")
while [[ $# -gt 0 ]]; do
  if [[ "$1" == --target ]]; then
    shift
    target_root="$target_root/$1"
  fi
  shift
done
set -- "${args[@]}"
mkdir -p "$target_root/release"
suffix="${BINARY_NAME#codex}"
while [[ $# -gt 0 ]]; do
  if [[ "$1" == --bin ]]; then
    shift
    if [[ "$1" != codex-code-mode-host || "${OMIT_HOST:-0}" != 1 ]]; then
      printf '#!/usr/bin/env bash\\n# fixture %s\\nexit 0\\n' "$1" > "$target_root/release/$1$suffix"
    fi
  fi
  shift
done
"""
        )
        cargo.chmod(0o755)
        self.env = {
            **os.environ,
            "PATH": f"{commands}{os.pathsep}{os.environ['PATH']}",
            "MESHAGENT_CODEX_WORKTREE": str(self.source),
            "CARGO_TARGET_DIR": str(self.root / "cargo target"),
            "CARGO_BUILD_TARGET": "",
            "CARGO_LOG": str(self.root / "cargo.log"),
            "BINARY_NAME": "codex",
        }

    def run_helper(self, *args):
        return subprocess.run(
            ["bash", str(OVERLAY / "codex-distribution"), *args],
            env=self.env,
            capture_output=True,
            text=True,
            check=False,
        )

    def test_default_build_stages_cli_and_host_from_external_target_directory(self):
        result = self.run_helper("build")
        self.assertEqual(result.returncode, 0, result.stderr)
        staged = Path(result.stdout.strip())
        self.assertEqual(staged, self.source / "dist/codex")
        self.assertEqual(
            staged.read_bytes(),
            (Path(self.env["CARGO_TARGET_DIR"]) / "release/codex").read_bytes(),
        )
        self.assertTrue(os.access(staged, os.X_OK))
        host = self.source / "dist/codex-code-mode-host"
        self.assertEqual(
            host.read_bytes(),
            (
                Path(self.env["CARGO_TARGET_DIR"]) / "release/codex-code-mode-host"
            ).read_bytes(),
        )
        self.assertTrue(os.access(host, os.X_OK))
        self.assertEqual(
            (self.root / "cargo.log").read_text().splitlines(),
            [
                "build",
                "--locked",
                "--release",
                "-p",
                "codex-cli",
                "--bin",
                "codex",
                "-p",
                "codex-code-mode-host",
                "--bin",
                "codex-code-mode-host",
            ],
        )

    def test_cli_only_build_skips_host(self):
        result = self.run_helper("build", "--cli-only")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(Path(result.stdout.strip()), self.source / "dist/codex")
        self.assertTrue((self.source / "dist/codex").is_file())
        self.assertFalse((self.source / "dist/codex-code-mode-host").exists())
        self.assertEqual(
            (self.root / "cargo.log").read_text().splitlines(),
            ["build", "--locked", "--release", "-p", "codex-cli", "--bin", "codex"],
        )

    def test_cross_build_stages_target_binaries_instead_of_stale_native_binaries(self):
        for target, native_target, suffix in (
            ("x86_64-apple-darwin", "aarch64-apple-darwin", ""),
            ("x86_64-pc-windows-msvc", "aarch64-pc-windows-msvc", ".exe"),
        ):
            for cargo_target_dir, environment_target, args in (
                (str(self.root / "cargo target"), "", ["--target", target]),
                ("relative target", target, []),
                ("relative target", native_target, ["--target", target]),
            ):
                with self.subTest(
                    target=target,
                    target_dir=cargo_target_dir,
                    environment=environment_target,
                ):
                    self.env.update(
                        BINARY_NAME=f"codex{suffix}",
                        CARGO_TARGET_DIR=cargo_target_dir,
                        CARGO_BUILD_TARGET=environment_target,
                    )
                    target_root = self.source / "codex-rs" / cargo_target_dir
                    native = target_root / "release"
                    native.mkdir(parents=True, exist_ok=True)
                    names = [
                        f"{name}{suffix}" for name in ("codex", "codex-code-mode-host")
                    ]
                    for name in names:
                        (native / name).write_bytes(b"stale native executable")
                    result = self.run_helper("build", *args)
                    self.assertEqual(result.returncode, 0, result.stderr)
                    self.assertEqual(
                        Path(result.stdout.strip()), self.source / f"dist/codex{suffix}"
                    )
                    for name in names:
                        staged = self.source / "dist" / name
                        self.assertEqual(
                            staged.read_bytes(),
                            (target_root / target / "release" / name).read_bytes(),
                        )
                        self.assertNotEqual(
                            staged.read_bytes(), (native / name).read_bytes()
                        )
                        self.assertTrue(os.access(staged, os.X_OK))
                    self.assertEqual(
                        (self.root / "cargo.log").read_text().splitlines()[-2:],
                        ["--target", target],
                    )

    def test_missing_cross_target_does_not_start_cargo(self):
        result = self.run_helper("build", "--target")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("--target requires a Rust target triple", result.stderr)
        self.assertFalse((self.root / "cargo.log").exists())

    def test_windows_autocrlf_checkout_preserves_build_inputs(self):
        repository = self.root / "overlay repository"
        shutil.copytree(
            OVERLAY, repository, ignore=shutil.ignore_patterns("__pycache__")
        )
        subprocess.run(["git", "init", "--quiet", str(repository)], check=True)
        subprocess.run(
            ["git", "-C", str(repository), "-c", "core.autocrlf=false", "add", "."],
            check=True,
        )
        subprocess.run(
            [
                "git",
                "-C",
                str(repository),
                "-c",
                "user.name=Distribution Test",
                "-c",
                "user.email=distribution-test@example.invalid",
                "commit",
                "--quiet",
                "-m",
                "Fixture overlay",
            ],
            check=True,
        )
        checkout = self.root / "windows checkout"
        subprocess.run(
            [
                "git",
                "clone",
                "--quiet",
                "--no-local",
                "-c",
                "core.autocrlf=true",
                str(repository),
                str(checkout),
            ],
            check=True,
        )
        inputs = [
            checkout / "codex-distribution",
            checkout / "upstream.lock",
            checkout / "patches/series",
            *sorted((checkout / "patches").glob("*.patch")),
        ]
        for path in inputs:
            self.assertNotIn(b"\r\n", path.read_bytes(), str(path))
        result = subprocess.run(
            ["bash", str(checkout / "codex-distribution"), "build"],
            env=self.env,
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(Path(result.stdout.strip()), self.source / "dist/codex")
        self.assertTrue((self.source / "dist/codex-code-mode-host").is_file())

    def test_windows_cli_and_host_install_from_relative_target_directory(self):
        self.env.update(BINARY_NAME="codex.exe", CARGO_TARGET_DIR="relative target")
        managed_home = self.root / "managed home"
        result = self.run_helper("install-managed", str(managed_home))
        self.assertEqual(result.returncode, 0, result.stderr)
        installed, host = map(Path, result.stdout.splitlines())
        self.assertEqual(
            installed, managed_home / "packages/standalone/current/codex.exe"
        )
        self.assertEqual(
            installed.read_bytes(),
            (self.source / "codex-rs/relative target/release/codex.exe").read_bytes(),
        )
        self.assertEqual(
            host,
            managed_home / "packages/standalone/current/codex-code-mode-host.exe",
        )
        self.assertEqual(
            host.read_bytes(),
            (
                self.source
                / "codex-rs/relative target/release/codex-code-mode-host.exe"
            ).read_bytes(),
        )
        self.assertTrue(os.access(host, os.X_OK))

    def test_failed_build_does_not_report_successful_install(self):
        self.env["FAIL_BUILD"] = "1"
        result = self.run_helper("install-managed", str(self.root / "managed"))
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("failed to build Codex distribution executables", result.stderr)
        self.assertEqual(result.stdout, "")
        self.assertFalse((self.root / "managed").exists())

    def test_missing_host_prevents_managed_install(self):
        self.env["OMIT_HOST"] = "1"
        result = self.run_helper("install-managed", str(self.root / "managed"))
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("failed to stage Codex Code Mode host", result.stderr)
        self.assertEqual(result.stdout, "")
        self.assertFalse((self.root / "managed").exists())

    def test_invalid_build_option_does_not_start_cargo(self):
        result = self.run_helper("build", "--cli-onyl")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("usage: build [--cli-only]", result.stderr)
        self.assertFalse((self.root / "cargo.log").exists())

    def test_tests_cannot_change_shared_login_and_cleanup_after_failure(self):
        real_home = self.root / "developer home"
        settings = real_home / ".meshagent/settings.json"
        settings.parent.mkdir(parents=True)
        settings.write_text("saved login")
        commands = self.root / "bin"
        just = commands / "just"
        just.write_text(
            """#!/usr/bin/env bash
set -euo pipefail
printf '%s\\n' "$HOME" "$USERPROFILE" "$MESHAGENT_CODEX_HOME" "$CARGO_HOME" "$RUSTUP_HOME" \\
  "${OPENAI_API_KEY:+present}${CODEX_API_KEY:+present}${MESHAGENT_API_URL:+present}" > "$TEST_ENV_LOG"
mkdir -p "$HOME/.meshagent"
printf 'logged out' > "$HOME/.meshagent/settings.json"
exit "${FAKE_TEST_EXIT:-0}"
"""
        )
        just.chmod(0o755)
        nextest = commands / "cargo-nextest"
        nextest.write_text("#!/usr/bin/env bash\nexit 0\n")
        nextest.chmod(0o755)
        self.env.update(
            HOME=str(real_home),
            USERPROFILE=str(real_home),
            TMPDIR=str(self.root),
            CARGO_HOME=str(self.root / "cargo cache"),
            RUSTUP_HOME=str(self.root / "rustup cache"),
            OPENAI_API_KEY="fixture-key",
            CODEX_API_KEY="fixture-key",
            MESHAGENT_API_URL="https://api.example.invalid",
            TEST_ENV_LOG=str(self.root / "test-env.log"),
        )
        for exit_code in (0, 42):
            with self.subTest(exit_code=exit_code):
                self.env["FAKE_TEST_EXIT"] = str(exit_code)
                result = self.run_helper("test", "-p", "codex-login")
                self.assertEqual(result.returncode, exit_code, result.stderr)
                self.assertEqual(settings.read_text(), "saved login")
                home, profile, codex_home, cargo_home, rustup_home, credentials = (
                    (self.root / "test-env.log").read_text().splitlines()
                )
                self.assertEqual(home, profile)
                self.assertNotEqual(Path(home), real_home)
                self.assertEqual(Path(cargo_home), Path(self.env["CARGO_HOME"]))
                self.assertEqual(Path(rustup_home), Path(self.env["RUSTUP_HOME"]))
                self.assertEqual(credentials, "")
                self.assertFalse(Path(home).exists())
                self.assertFalse(Path(codex_home).exists())


class PackageManifestTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.version = "0.52.2-meshagent.0.153.4"
        self.cli_version = "0.153.4-meshagent.1"
        self.base_url = (
            "https://storage.googleapis.com/meshagent-enterprise-codex-builds/0.52.2"
        )

    def generate(self, format_name, output, *extra_args):
        return subprocess.run(
            [
                sys.executable,
                str(OVERLAY / "package-manifests.py"),
                format_name,
                "--version",
                self.version,
                "--base-url",
                self.base_url,
                "--artifacts",
                str(self.root),
                "--output",
                str(output),
                *extra_args,
                *(
                    ["--cli-version", self.cli_version]
                    if format_name == "homebrew"
                    else []
                ),
            ],
            capture_output=True,
            text=True,
            check=False,
        )

    def test_homebrew_maps_each_platform_to_its_artifact_checksum(self):
        expected = {}
        for platform in (
            "macos-arm64",
            "macos-x86_64",
            "linux-arm64",
            "linux-x86_64",
        ):
            filename = f"enterprise-codex-{self.version}-{platform}.tar.gz"
            payload = os.urandom(1024)
            (self.root / filename).write_bytes(payload)
            expected[f"{self.base_url}/{filename}"] = hashlib.sha256(
                payload
            ).hexdigest()
        for name, class_name, extra_args in (
            ("enterprise-codex", "EnterpriseCodex", []),
            (
                "enterprise-codex@0.52.2",
                "EnterpriseCodexAT0522",
                ["--homebrew-version", "0.52.2"],
            ),
        ):
            with self.subTest(formula=name):
                output = self.root / f"Formula/{name}.rb"
                result = self.generate("homebrew", output, *extra_args)
                self.assertEqual(result.returncode, 0, result.stderr)
                formula = output.read_text()
                pairs = re.findall(r'url "([^"]+)"\s+sha256 "([^"]+)"', formula)
                self.assertEqual(dict(pairs), expected)
                self.assertTrue(formula.startswith(f"class {class_name} < Formula\n"))
                self.assertEqual(
                    "keg_only :versioned_formula" in formula, bool(extra_args)
                )
                self.assertIn(f'version "{self.version}"', formula)
                self.assertIn(
                    f'assert_match "codex-cli {self.cli_version}", shell_output("#{{bin}}/codex --version")',
                    formula,
                )

    def test_versioned_homebrew_rejects_mislabeled_release(self):
        output = self.root / "Formula/enterprise-codex@0.52.1.rb"
        result = self.generate("homebrew", output, "--homebrew-version", "0.52.1")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("must match the MeshAgent release", result.stderr)
        self.assertFalse(output.exists())

    def test_winget_maps_architectures_to_public_archives_and_portable_codex(self):
        expected = {}
        for platform, architecture in (
            ("windows-x86_64", "x64"),
            ("windows-arm64", "arm64"),
        ):
            filename = f"enterprise-codex-{self.version}-{platform}.zip"
            payload = os.urandom(1024)
            (self.root / filename).write_bytes(payload)
            expected[architecture] = (
                f"{self.base_url}/{filename}",
                hashlib.sha256(payload).hexdigest().upper(),
            )
        output = self.root / "manifests"
        result = self.generate("winget", output)
        self.assertEqual(result.returncode, 0, result.stderr)
        manifest = (output / "MeshAgent.EnterpriseCodex.installer.yaml").read_text()
        installers = re.findall(
            r"- Architecture: (\w+)\s+InstallerUrl: (.+)\s+InstallerSha256: (\w+)",
            manifest,
        )
        self.assertEqual(len(installers), 2)
        self.assertEqual(
            {
                architecture: (json.loads(url), sha256)
                for architecture, url, sha256 in installers
            },
            expected,
        )
        self.assertIn("NestedInstallerType: portable\n", manifest)
        self.assertIn("RelativeFilePath: codex.exe\n", manifest)
        self.assertIn("PortableCommandAlias: codex\n", manifest)
        self.assertEqual(len(list(output.glob("*.yaml"))), 3)
        for suffix, manifest_type in (
            (".yaml", "version"),
            (".installer.yaml", "installer"),
            (".locale.en-US.yaml", "defaultLocale"),
        ):
            with self.subTest(manifest_type=manifest_type):
                content = (output / f"MeshAgent.EnterpriseCodex{suffix}").read_text()
                self.assertIn(f'PackageVersion: "{self.version}"\n', content)
                self.assertEqual(
                    content.splitlines()[0],
                    "# yaml-language-server: $schema=https://aka.ms/"
                    f"winget-manifest.{manifest_type}.1.10.0.schema.json",
                )
                self.assertIn(f"ManifestType: {manifest_type}\n", content)
                self.assertIn("ManifestVersion: 1.10.0\n", content)

    def test_homebrew_can_test_one_os_without_other_os_archives(self):
        for operating_system, other_os in (("macos", "linux"), ("linux", "macos")):
            with self.subTest(operating_system=operating_system):
                artifacts = self.root / operating_system
                artifacts.mkdir()
                for architecture in ("arm64", "x86_64"):
                    (
                        artifacts
                        / f"enterprise-codex-{self.version}-{operating_system}-{architecture}.tar.gz"
                    ).write_bytes(b"archive fixture")
                output = artifacts / "enterprise-codex.rb"
                result = self.generate(
                    "homebrew",
                    output,
                    "--artifacts",
                    str(artifacts),
                    "--homebrew-os",
                    operating_system,
                )
                self.assertEqual(result.returncode, 0, result.stderr)
                formula = output.read_text()
                self.assertIn(f"on_{operating_system} do", formula)
                self.assertNotIn(f"on_{other_os} do", formula)
                self.assertIn(hashlib.sha256(b"archive fixture").hexdigest(), formula)

    def test_homebrew_can_test_intel_only_latest_and_versioned_formulas(self):
        filename = f"enterprise-codex-{self.version}-macos-x86_64.tar.gz"
        payload = b"Intel archive fixture"
        (self.root / filename).write_bytes(payload)
        for formula_version in (None, "0.52.2"):
            with self.subTest(formula_version=formula_version):
                name = "enterprise-codex"
                args = ["--homebrew-platform", "macos-x86_64"]
                if formula_version:
                    name += f"@{formula_version}"
                    args += ["--homebrew-version", formula_version]
                output = self.root / f"{name}.rb"
                result = self.generate("homebrew", output, *args)
                self.assertEqual(result.returncode, 0, result.stderr)
                formula = output.read_text()
                self.assertIn("on_macos do\n    on_intel do", formula)
                self.assertNotIn("on_arm do", formula)
                self.assertNotIn("on_linux", formula)
                self.assertEqual(
                    re.findall(r'url "([^"]+)"\s+sha256 "([^"]+)"', formula),
                    [
                        (
                            f"{self.base_url}/{filename}",
                            hashlib.sha256(payload).hexdigest(),
                        )
                    ],
                )
                self.assertEqual(
                    "keg_only :versioned_formula" in formula, bool(formula_version)
                )

    def test_missing_arm64_archive_prevents_winget_manifest_creation(self):
        (self.root / f"enterprise-codex-{self.version}-windows-x86_64.zip").write_bytes(
            b"x64 archive"
        )
        output = self.root / "manifests"
        result = self.generate("winget", output)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("windows-arm64.zip", result.stderr)
        self.assertFalse(output.exists())

    def test_missing_artifact_prevents_formula_creation(self):
        output = self.root / "Formula/enterprise-codex.rb"
        result = self.generate("homebrew", output)
        self.assertNotEqual(result.returncode, 0)
        self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()
