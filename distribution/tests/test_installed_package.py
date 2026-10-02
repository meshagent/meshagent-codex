import importlib.util
import io
import os
import struct
import subprocess
import sys
import tarfile
import tempfile
import unittest
import zipfile
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

SCRIPT = Path(__file__).resolve().parents[1] / "smoke-installed.py"
spec = importlib.util.spec_from_file_location("smoke_installed", SCRIPT)
smoke_installed = importlib.util.module_from_spec(spec)
spec.loader.exec_module(smoke_installed)


def executable_header(target):
    header = bytearray(128)
    arm = target.endswith("arm64")
    if target.startswith("linux"):
        header[:6] = b"\x7fELF\x02\x01"
        struct.pack_into("<H", header, 18, 183 if arm else 62)
    elif target.startswith("macos"):
        header[:4] = b"\xcf\xfa\xed\xfe"
        struct.pack_into("<I", header, 4, 0x0100000C if arm else 0x01000007)
    else:
        header[:2] = b"MZ"
        struct.pack_into("<I", header, 60, 64)
        header[64:68] = b"PE\0\0"
        struct.pack_into("<H", header, 68, 0xAA64 if arm else 0x8664)
    return bytes(header)


class InstalledPackageTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)

    def package(self, target, executable_target=None, formula_name="enterprise-codex"):
        prefix = self.root / target
        windows = target.startswith("windows")
        binary_dir = prefix if windows else prefix / "libexec"
        names = ["codex", "codex-code-mode-host"]
        if windows:
            names = [
                name + ".exe"
                for name in names
                + ["codex-command-runner", "codex-windows-sandbox-setup"]
            ]
        elif target.startswith("linux"):
            names.append("codex-resources/bwrap")
        files = {name: executable_header(executable_target or target) for name in names}
        paths = {name: binary_dir / name for name in names}
        for name in ("LICENSE", "NOTICE"):
            files[name] = b"package legal notice"
            paths[name] = (
                prefix if windows else prefix / "share" / formula_name
            ) / name
        if target.startswith("linux"):
            for name in ("bwrap-LICENSE", "bwrap-source.tar.gz"):
                files[f"codex-resources/{name}"] = b"sandbox source/license fixture"
                paths[f"codex-resources/{name}"] = binary_dir / "codex-resources" / name
        for name, content in files.items():
            paths[name].parent.mkdir(parents=True, exist_ok=True)
            paths[name].write_bytes(content)
            paths[name].chmod(0o755)
        archive = prefix.with_suffix(".zip" if windows else ".tar.gz")
        if windows:
            with zipfile.ZipFile(archive, "w") as package:
                for name, content in files.items():
                    package.writestr(name, content)
        else:
            with tarfile.open(archive, "w:gz") as package:
                for name, content in files.items():
                    info = tarfile.TarInfo(name)
                    info.size = len(content)
                    package.addfile(info, io.BytesIO(content))
        return archive, prefix, binary_dir

    def test_installed_files_match_archive_for_all_six_targets(self):
        for operating_system in ("macos", "linux", "windows"):
            for architecture in ("arm64", "x86_64"):
                target = f"{operating_system}-{architecture}"
                with self.subTest(target=target):
                    archive, prefix, binary_dir = self.package(target)
                    self.assertEqual(
                        smoke_installed.verify_files(archive, prefix, target),
                        binary_dir,
                    )

    def test_changed_installed_host_is_rejected(self):
        archive, prefix, binary_dir = self.package("macos-arm64")
        with (binary_dir / "codex-code-mode-host").open("ab") as host:
            host.write(b"different build")
        with self.assertRaisesRegex(RuntimeError, "differs from release archive"):
            smoke_installed.verify_files(archive, prefix, "macos-arm64")

    def test_versioned_homebrew_checks_notices_under_its_own_formula_name(self):
        name = "enterprise-codex@0.52.2"
        for target in ("macos-arm64", "linux-x86_64"):
            with self.subTest(target=target):
                archive, prefix, binary_dir = self.package(target, formula_name=name)
                self.assertEqual(
                    smoke_installed.verify_files(archive, prefix, target, name),
                    binary_dir,
                )
                (prefix / "share" / name / "NOTICE").unlink()
                with self.assertRaises(FileNotFoundError):
                    smoke_installed.verify_files(archive, prefix, target, name)

    def test_wrong_architecture_is_rejected_even_when_checksum_matches(self):
        archive, prefix, _ = self.package("windows-arm64", "windows-x86_64")
        with self.assertRaisesRegex(RuntimeError, "Wrong executable architecture"):
            smoke_installed.verify_files(archive, prefix, "windows-arm64")

    def test_missing_windows_helper_is_rejected(self):
        archive, prefix, binary_dir = self.package("windows-x86_64")
        (binary_dir / "codex-windows-sandbox-setup.exe").unlink()
        with self.assertRaises(FileNotFoundError):
            smoke_installed.verify_files(archive, prefix, "windows-x86_64")

    def test_unknown_executable_format_is_rejected(self):
        path = self.root / "codex"
        path.write_bytes(b"not an executable")
        with self.assertRaisesRegex(RuntimeError, "Unrecognized executable"):
            smoke_installed.executable_platform(path)

    def run_smoke(self, target, write_exit_code, overwrite=False):
        archive, prefix, binary_dir = self.package(target)
        windows = target.startswith("windows-")
        binary = binary_dir / ("codex.exe" if windows else "codex")
        runner_home = (self.root / "runner home").resolve()
        runner_home.mkdir()
        version = "0.153.4-meshagent.1"
        workspaces = set()

        def execute(args, **kwargs):
            output = ""
            code = 0
            if "/user" in args:
                output = '"runner, admin","S-1-5-21-123-456-789-1001"\n'
            if "cwd" in kwargs:
                workspace = kwargs["cwd"]
                workspaces.add(workspace)
                self.assertEqual(workspace.parent, runner_home)
                self.assertNotIn("CODEX_HOME", kwargs["env"])
                self.assertNotIn("GITHUB_TOKEN", kwargs["env"])
            if "--version" in args:
                output = f"codex-cli {version}\n"
            elif "--help" in args:
                output = "MeshAgent Codex is maintained by MeshAgent"
            elif "sandbox" in args:
                config = Path(kwargs["env"]["MESHAGENT_CODEX_HOME"]) / "config.toml"
                self.assertIn('sandbox_mode = "read-only"', config.read_text())
                if "unexpected" in args[-1]:
                    code = write_exit_code
                    if overwrite:
                        (workspace / "input.txt").write_text("unexpected")
                else:
                    output = (workspace / "input.txt").read_text()
            return subprocess.CompletedProcess(args, code, output, "")

        with (
            mock.patch.object(smoke_installed.Path, "home", return_value=runner_home),
            mock.patch.object(
                smoke_installed.shutil, "which", return_value=str(binary)
            ),
            mock.patch.object(
                smoke_installed.platform, "machine", return_value="arm64"
            ),
            mock.patch.object(
                smoke_installed,
                "sys",
                SimpleNamespace(
                    platform="win32" if windows else "darwin",
                    executable=sys.executable,
                    stderr=sys.stderr,
                ),
            ),
            mock.patch.object(
                smoke_installed,
                "os",
                SimpleNamespace(
                    name="nt" if windows else "posix",
                    path=os.path,
                    access=os.access,
                    X_OK=os.X_OK,
                    environ={
                        "SystemRoot": "C:/Windows",
                        "CODEX_HOME": "existing state",
                        "GITHUB_TOKEN": "must not reach sandbox",
                    },
                ),
            ),
            mock.patch.object(smoke_installed.subprocess, "run", side_effect=execute),
        ):
            try:
                smoke_installed.smoke(prefix, archive, target, version)
            finally:
                self.assertTrue(workspaces)
                self.assertTrue(all(not path.exists() for path in workspaces))

    def test_readable_sandbox_fixture_stays_isolated_and_is_cleaned_up(self):
        self.run_smoke("windows-arm64", write_exit_code=1)

    def test_successful_sandbox_write_is_rejected(self):
        with self.assertRaisesRegex(RuntimeError, "allowed a workspace write"):
            self.run_smoke("windows-arm64", write_exit_code=0)

    def test_failed_sandbox_write_that_modifies_file_is_rejected(self):
        with self.assertRaisesRegex(RuntimeError, "allowed a workspace write"):
            self.run_smoke("macos-arm64", write_exit_code=1, overwrite=True)


if __name__ == "__main__":
    unittest.main()
