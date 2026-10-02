"""Generate package-manager manifests from the exact release archives."""

import argparse
import hashlib
import json
import re
from pathlib import Path


def archive_sha256(path: Path) -> str:
    with path.open("rb") as archive:
        return hashlib.file_digest(archive, "sha256").hexdigest()


def homebrew(
    version: str,
    cli_version: str,
    base_url: str,
    artifacts: Path,
    output: Path,
    platforms: list[str],
    formula_version: str | None,
) -> None:
    archives = {}
    for platform in platforms:
        filename = f"enterprise-codex-{version}-{platform}.tar.gz"
        archives[platform] = (
            json.dumps(f"{base_url}/{filename}"),
            json.dumps(archive_sha256(artifacts / filename)),
        )

    class_name = "EnterpriseCodex"
    if formula_version:
        # Match Homebrew's Formulary.class_s conversion for versioned names.
        class_name += "AT" + re.sub(
            r"[.-]([a-z0-9])", lambda match: match[1].upper(), formula_version
        ).replace("+", "x")
    lines = [
        f"class {class_name} < Formula",
        '  desc "MeshAgent enterprise distribution of OpenAI Codex"',
        '  homepage "https://www.meshagent.com"',
        f"  version {json.dumps(version)}",
        '  license "Apache-2.0"',
    ]
    if formula_version:
        lines.extend(["", "  keg_only :versioned_formula"])
    for operating_system in dict.fromkeys(p.split("-")[0] for p in platforms):
        lines.extend(["", f"  on_{operating_system} do"])
        for architecture, brew_arch in (("arm64", "arm"), ("x86_64", "intel")):
            platform = f"{operating_system}-{architecture}"
            if platform not in archives:
                continue
            url, sha256 = archives[platform]
            lines.extend(
                [
                    f"    on_{brew_arch} do",
                    f"      url {url}",
                    f"      sha256 {sha256}",
                    "    end",
                ]
            )
        lines.append("  end")
    lines.extend(
        [
            "",
            "  def install",
            '    libexec.install "codex", "codex-code-mode-host"',
            '    libexec.install "codex-resources" if OS.linux?',
            '    bin.install_symlink libexec/"codex"',
            '    pkgshare.install "LICENSE", "NOTICE"',
            "  end",
            "",
            "  test do",
            f'    assert_match {json.dumps(f"codex-cli {cli_version}")}, shell_output("#{{bin}}/codex --version")',
            '    system libexec/"codex-code-mode-host", "--help"',
            "  end",
            "end",
            "",
        ]
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines), encoding="utf-8")


def winget(
    version: str, base_url: str, artifacts: Path, output: Path, identifier: str
) -> None:
    installers = []
    for platform, architecture in (
        ("windows-x86_64", "x64"),
        ("windows-arm64", "arm64"),
    ):
        filename = f"enterprise-codex-{version}-{platform}.zip"
        sha256 = archive_sha256(artifacts / filename).upper()
        installers.append(
            f"  - Architecture: {architecture}\n"
            f"    InstallerUrl: {json.dumps(f'{base_url}/{filename}')}\n"
            f"    InstallerSha256: {sha256}\n"
        )
    common = (
        f"PackageIdentifier: {json.dumps(identifier)}\n"
        f"PackageVersion: {json.dumps(version)}\n"
    )
    manifests = {
        f"{identifier}.yaml": "# yaml-language-server: $schema=https://aka.ms/winget-manifest.version.1.10.0.schema.json\n"
        + common
        + "DefaultLocale: en-US\nManifestType: version\nManifestVersion: 1.10.0\n",
        f"{identifier}.installer.yaml": "# yaml-language-server: $schema=https://aka.ms/winget-manifest.installer.1.10.0.schema.json\n"
        + common
        + "InstallerType: zip\n"
        + "NestedInstallerType: portable\n"
        + "NestedInstallerFiles:\n"
        + "  - RelativeFilePath: codex.exe\n"
        + "    PortableCommandAlias: codex\n"
        + "UpgradeBehavior: uninstallPrevious\n"
        + "Commands:\n  - codex\n"
        + "Installers:\n"
        + "".join(installers)
        + "ManifestType: installer\nManifestVersion: 1.10.0\n",
        f"{identifier}.locale.en-US.yaml": "# yaml-language-server: $schema=https://aka.ms/winget-manifest.defaultLocale.1.10.0.schema.json\n"
        + common
        + "PackageLocale: en-US\n"
        + "Publisher: MeshAgent\n"
        + "PublisherUrl: https://www.meshagent.com\n"
        + "PackageName: Enterprise Codex\n"
        + "PackageUrl: https://www.meshagent.com\n"
        + "License: Apache-2.0\n"
        + "ShortDescription: MeshAgent enterprise distribution of OpenAI Codex\n"
        + "Description: >-\n"
        + "  MeshAgent Codex is maintained by MeshAgent and is based on the\n"
        + "  open-source OpenAI Codex codebase. It is not an official OpenAI\n"
        + "  distribution. This package includes the Codex CLI, its JavaScript\n"
        + "  Code Mode host, and the Windows sandbox helpers.\n"
        + "ManifestType: defaultLocale\nManifestVersion: 1.10.0\n",
    }
    output.mkdir(parents=True, exist_ok=True)
    for filename, content in manifests.items():
        (output / filename).write_text(content, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("format", choices=("homebrew", "winget"))
    parser.add_argument("--version", required=True, help="Package and archive version")
    parser.add_argument(
        "--cli-version", help="Executable version, required for Homebrew"
    )
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--artifacts", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--identifier", default="MeshAgent.EnterpriseCodex")
    homebrew_targets = parser.add_mutually_exclusive_group()
    homebrew_targets.add_argument(
        "--homebrew-os",
        action="append",
        choices=("macos", "linux"),
        help="Limit a test formula to selected operating systems; defaults to both",
    )
    homebrew_targets.add_argument(
        "--homebrew-platform",
        action="append",
        choices=("macos-arm64", "macos-x86_64", "linux-arm64", "linux-x86_64"),
        help="Limit a test formula to selected OS/architecture pairs; defaults to all",
    )
    parser.add_argument(
        "--homebrew-version",
        help="Generate a keg-only enterprise-codex@<MeshAgent release> formula",
    )
    args = parser.parse_args()
    if args.homebrew_version and (
        args.format != "homebrew"
        or not re.fullmatch(r"[0-9][a-z0-9]*(?:[.+-][a-z0-9]+)*", args.homebrew_version)
        or not args.version.startswith(f"{args.homebrew_version}-meshagent.")
    ):
        parser.error("homebrew-version must match the MeshAgent release in --version")
    if not re.fullmatch(r"[0-9][A-Za-z0-9.+-]*", args.version):
        parser.error(
            "version must be a release version, such as 0.52.2-meshagent.0.153.4"
        )
    if args.format == "homebrew" and (
        not args.cli_version
        or not re.fullmatch(r"[0-9][A-Za-z0-9.+-]*", args.cli_version)
    ):
        parser.error("homebrew requires --cli-version with the executable's version")
    if not re.fullmatch(r"[A-Za-z0-9]+(?:\.[A-Za-z0-9]+)+", args.identifier):
        parser.error("identifier must have the form Publisher.Package")
    if not args.base_url.startswith("https://"):
        parser.error("base-url must be a public HTTPS download location")
    base_url = args.base_url.rstrip("/")
    if args.format == "homebrew":
        platforms = args.homebrew_platform or [
            f"{operating_system}-{architecture}"
            for operating_system in (args.homebrew_os or ["macos", "linux"])
            for architecture in ("arm64", "x86_64")
        ]
        homebrew(
            args.version,
            args.cli_version,
            base_url,
            args.artifacts,
            args.output,
            list(dict.fromkeys(platforms)),
            args.homebrew_version,
        )
    else:
        winget(args.version, base_url, args.artifacts, args.output, args.identifier)


if __name__ == "__main__":
    main()
