from __future__ import annotations

from pathlib import Path

import pytest

from .client import AppServerConfig, CodexBinResolverOps, resolve_codex_bin


def _ops(
    *,
    installed_codex_path,
    path_codex_path,
    existing_paths: set[Path] | None = None,
) -> CodexBinResolverOps:
    existing = existing_paths or set()
    return CodexBinResolverOps(
        installed_codex_path=installed_codex_path,
        path_codex_path=path_codex_path,
        path_exists=lambda path: path in existing,
    )


def test_resolve_codex_bin_uses_explicit_config() -> None:
    explicit_path = Path("/opt/codex")
    assert (
        resolve_codex_bin(
            AppServerConfig(codex_bin=str(explicit_path)),
            _ops(
                installed_codex_path=lambda: Path("/bundled/codex"),
                path_codex_path=lambda: Path("/usr/local/bin/codex"),
                existing_paths={explicit_path},
            ),
        )
        == explicit_path
    )


def test_resolve_codex_bin_uses_path_before_bundled_runtime() -> None:
    path_codex = Path("/usr/local/bin/codex")
    assert (
        resolve_codex_bin(
            AppServerConfig(),
            _ops(
                installed_codex_path=lambda: Path("/bundled/codex"),
                path_codex_path=lambda: path_codex,
            ),
        )
        == path_codex
    )


def test_resolve_codex_bin_falls_back_to_bundled_runtime() -> None:
    bundled_codex = Path("/bundled/codex")
    assert (
        resolve_codex_bin(
            AppServerConfig(),
            _ops(
                installed_codex_path=lambda: bundled_codex,
                path_codex_path=lambda: None,
            ),
        )
        == bundled_codex
    )


def test_resolve_codex_bin_reports_missing_runtime() -> None:
    with pytest.raises(FileNotFoundError, match="Unable to locate a Codex runtime"):
        resolve_codex_bin(
            AppServerConfig(),
            _ops(
                installed_codex_path=lambda: (_ for _ in ()).throw(
                    FileNotFoundError("Unable to locate a Codex runtime")
                ),
                path_codex_path=lambda: None,
            ),
        )
