"""Check packaged MeshAgent authentication with Windows' native home environment."""

import argparse
import os
import subprocess
import tempfile
from pathlib import Path


def smoke(binary: Path) -> None:
    if os.name != "nt":
        raise RuntimeError("This smoke test requires Windows")
    with tempfile.TemporaryDirectory(prefix="codex-auth-smoke-") as profile:
        codex_home = Path(profile) / ".meshagent" / "codex"
        codex_home.mkdir(parents=True)
        environment = os.environ.copy()
        environment.pop("HOME", None)
        environment["USERPROFILE"] = profile
        environment["MESHAGENT_CODEX_HOME"] = str(codex_home)
        result = subprocess.run(
            [str(binary.resolve()), "login", "status", "--provider=meshagent"],
            env=environment,
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
        # An empty profile must report signed out, not fail to find its home.
        if (
            result.returncode != 1
            or "MeshAgent: not logged in" not in result.stderr.splitlines()
        ):
            raise RuntimeError(
                f"Windows authentication smoke test failed ({result.returncode}):\n"
                f"{result.stdout}{result.stderr}"
            )
    print("Windows authentication passed: USERPROFILE works without HOME")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("binary", type=Path)
    smoke(parser.parse_args().binary)
