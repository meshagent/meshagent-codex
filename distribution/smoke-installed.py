"""Exercise a package-manager installation against its original release archive."""

import argparse
import csv
import hashlib
import os
import platform
import shutil
import struct
import subprocess
import sys
import tarfile
import tempfile
import zipfile
from pathlib import Path


def executable_platform(path: Path) -> str:
    with path.open("rb") as binary:
        header = binary.read(64)
        if header[:4] == b"\x7fELF" and header[4:6] == b"\x02\x01":
            return (
                "linux-"
                + {62: "x86_64", 183: "arm64"}[struct.unpack_from("<H", header, 18)[0]]
            )
        if header[:4] == b"\xcf\xfa\xed\xfe":
            return (
                "macos-"
                + {0x01000007: "x86_64", 0x0100000C: "arm64"}[
                    struct.unpack_from("<I", header, 4)[0]
                ]
            )
        if header[:2] == b"MZ":
            binary.seek(struct.unpack_from("<I", header, 60)[0])
            pe_header = binary.read(6)
            if pe_header[:4] == b"PE\0\0":
                return (
                    "windows-"
                    + {0x8664: "x86_64", 0xAA64: "arm64"}[
                        struct.unpack_from("<H", pe_header, 4)[0]
                    ]
                )
    raise RuntimeError(f"Unrecognized executable format: {path}")


def verify_files(
    archive: Path, prefix: Path, target: str, formula_name: str = "enterprise-codex"
) -> Path:
    windows = target.startswith("windows-")
    binaries = ["codex", "codex-code-mode-host"]
    if windows:
        binaries += ["codex-command-runner", "codex-windows-sandbox-setup"]
        binaries = [f"{name}.exe" for name in binaries]
    elif target.startswith("linux-"):
        binaries += ["codex-resources/bwrap"]
    binary_dir = prefix if windows else prefix / "libexec"
    paths = {name: binary_dir / name for name in binaries}
    share = prefix if windows else prefix / "share" / formula_name
    paths.update({name: share / name for name in ("LICENSE", "NOTICE")})
    if target.startswith("linux-"):
        for name in ("bwrap-LICENSE", "bwrap-source.tar.gz"):
            paths[f"codex-resources/{name}"] = binary_dir / "codex-resources" / name

    # Compare installed bytes to the build artifact, not another local executable.
    if windows:
        with zipfile.ZipFile(archive) as package:
            expected = {
                name: hashlib.sha256(package.read(name)).hexdigest() for name in paths
            }
    else:
        with tarfile.open(archive) as package:
            expected = {}
            for name in paths:
                with package.extractfile(name) as member:
                    expected[name] = hashlib.file_digest(member, "sha256").hexdigest()
    for name, installed in paths.items():
        with installed.open("rb") as binary:
            actual = hashlib.file_digest(binary, "sha256").hexdigest()
        if actual != expected[name]:
            raise RuntimeError(
                f"Installed file differs from release archive: {installed}"
            )
        if name in binaries:
            if executable_platform(installed) != target:
                raise RuntimeError(f"Wrong executable architecture: {installed}")
            if not os.access(installed, os.X_OK):
                raise RuntimeError(f"Installed file is not executable: {installed}")
    return binary_dir


def smoke(
    prefix: Path,
    archive: Path,
    target: str,
    version: str,
    formula_name: str = "enterprise-codex",
) -> None:
    host_os = {"darwin": "macos", "linux": "linux", "win32": "windows"}[sys.platform]
    host_arch = {
        "arm64": "arm64",
        "aarch64": "arm64",
        "amd64": "x86_64",
        "x86_64": "x86_64",
    }[platform.machine().lower()]
    if f"{host_os}-{host_arch}" != target:
        raise RuntimeError(f"Installation test requires a native {target} runner")
    prefix = prefix.resolve()
    binary_dir = verify_files(archive, prefix, target, formula_name)
    suffix = ".exe" if os.name == "nt" else ""
    binary = binary_dir / f"codex{suffix}"
    command = shutil.which("codex")
    if command is None or Path(command).resolve() != binary.resolve():
        raise RuntimeError(f"codex resolves to {command}, expected {binary}")

    # Codex refuses to create PATH helpers under the system temp directory. Keep
    # the workspace outside it too, so temporary-directory write exceptions do
    # not invalidate the read-only sandbox check.
    with tempfile.TemporaryDirectory(
        prefix=".codex installed smoke ", dir=Path.home()
    ) as directory:
        root = Path(directory).resolve()
        if os.name == "nt":
            system32 = Path(os.environ["SystemRoot"]) / "System32"
            identity = subprocess.run(
                [str(system32 / "whoami.exe"), "/user", "/fo", "csv", "/nh"],
                capture_output=True,
                text=True,
                check=True,
                timeout=30,
            )
            user_sid = next(csv.reader(identity.stdout.splitlines()))[1]
            # Python 3.13's private directory ACL grants owner/admin access. An
            # elevated runner can create it with Administrators as the owner;
            # the restricted token cannot use that group. Give only this user
            # inheritable read/execute access, without granting sandbox writes.
            subprocess.run(
                [
                    str(system32 / "icacls.exe"),
                    str(root),
                    "/grant",
                    f"*{user_sid}:(OI)(CI)(RX)",
                ],
                check=True,
                timeout=30,
            )
        home = root / "codex home"
        home.mkdir()
        # Keep OS essentials, but no build paths, credentials, or existing Codex config.
        env = {
            key: value
            for key, value in os.environ.items()
            if key.upper()
            in {
                "SYSTEMROOT",
                "WINDIR",
                "COMSPEC",
                "TEMP",
                "TMP",
                "TMPDIR",
                "USERPROFILE",
                "USERNAME",
                "USER",
                "LOCALAPPDATA",
                "APPDATA",
            }
        }
        env["MESHAGENT_CODEX_HOME"] = str(home)
        env["RUN_MESHAGENT_CLOUD_SMOKE"] = "1"
        if os.name == "nt":
            env["PATH"] = os.path.join(os.environ["SystemRoot"], "System32")
        else:
            env["HOME"] = str(root)
            env["PATH"] = "/usr/bin:/bin:/usr/sbin:/sbin"

        def run(*args: str) -> str:
            result = subprocess.run(
                args,
                cwd=root,
                env=env,
                capture_output=True,
                text=True,
                timeout=120,
                check=False,
            )
            if result.returncode:
                for log in sorted((home / ".sandbox").glob("sandbox.*.log")):
                    print(
                        f"{log.name}:\n{log.read_text(errors='replace')[-16000:]}",
                        file=sys.stderr,
                    )
                raise RuntimeError(
                    f"{args} failed ({result.returncode}):\n{result.stdout}{result.stderr}"
                )
            return result.stdout

        if run(command, "--version").strip() != f"codex-cli {version}":
            raise RuntimeError("Installed CLI version does not match the release")
        if "MeshAgent Codex is maintained by MeshAgent" not in run(command, "--help"):
            raise RuntimeError("Installed CLI is missing MeshAgent branding")
        scripts = Path(__file__).resolve().parent
        run(
            sys.executable,
            str(scripts / "smoke-code-mode-host.py"),
            str(binary_dir / f"codex-code-mode-host{suffix}"),
        )
        if os.name == "nt":
            run(sys.executable, str(scripts / "smoke-windows-auth.py"), str(binary))

        # Force a real managed sandbox; an unrestricted default would give a false pass.
        (home / "config.toml").write_text(
            'sandbox_mode = "read-only"\n[windows]\nsandbox = "unelevated"\n'
        )
        marker = "installed-package-sandbox-ok"
        (root / "input.txt").write_text(marker + "\n")
        sandbox_command = (
            [
                os.path.join(os.environ["SystemRoot"], "System32", "cmd.exe"),
                "/d",
                "/c",
                "type input.txt",
            ]
            if os.name == "nt"
            else ["/bin/cat", "input.txt"]
        )
        if run(command, "sandbox", "--", *sandbox_command).strip() != marker:
            raise RuntimeError("Installed sandbox could not execute the test command")
        # A readable fixture must not make the read-only sandbox writable.
        write_command = (
            [str(system32 / "cmd.exe"), "/d", "/c", "echo unexpected>input.txt"]
            if os.name == "nt"
            else ["/bin/sh", "-c", "printf unexpected > input.txt"]
        )
        write_result = subprocess.run(
            [command, "sandbox", "--", *write_command],
            cwd=root,
            env=env,
            capture_output=True,
            text=True,
            timeout=120,
            check=False,
        )
        if (
            write_result.returncode == 0
            or (root / "input.txt").read_text() != marker + "\n"
        ):
            raise RuntimeError(
                "Installed read-only sandbox allowed a workspace write:\n"
                f"{write_result.stdout}{write_result.stderr}"
            )
    print(
        f"{target}: installed bytes, architecture, command, version, Code Mode, and sandbox passed"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prefix", required=True, type=Path)
    parser.add_argument("--archive", required=True, type=Path)
    parser.add_argument("--platform", required=True)
    parser.add_argument("--version", required=True)
    parser.add_argument("--homebrew-formula", default="enterprise-codex")
    args = parser.parse_args()
    smoke(args.prefix, args.archive, args.platform, args.version, args.homebrew_formula)
