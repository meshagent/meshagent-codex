#!/usr/bin/env bash
# Run only on disposable CI runners: this installs and uninstalls a local test tap.
set -euo pipefail

formula_directory="$1"
archive="$2"
version="$3"
platform="$4"
meshagent_version="$5"
scripts="$(cd "$(dirname "$0")" && pwd)"
tap="meshagent/installation-test-${GITHUB_RUN_ID}-${GITHUB_RUN_ATTEMPT}"
formula_names=(enterprise-codex "enterprise-codex@$meshagent_version")
brew_prefix="$(brew --prefix)"
export HOMEBREW_NO_AUTO_UPDATE=1 HOMEBREW_NO_INSTALL_CLEANUP=1 HOMEBREW_NO_ANALYTICS=1
export HOMEBREW_NO_AUTOREMOVE=1
export HOMEBREW_CACHE="$RUNNER_TEMP/enterprise-codex-brew-cache"
export RUN_MESHAGENT_CLOUD_SMOKE=1

for name in "${formula_names[@]}"; do
  if brew list --formula --versions "$name" >/dev/null 2>&1; then
    echo "Installation tests require a runner without $name installed." >&2
    exit 1
  fi
done
if [[ -e "$brew_prefix/bin/codex" || -L "$brew_prefix/bin/codex" ]]; then
  echo 'Installation tests require a runner without a conflicting Codex command.' >&2
  exit 1
fi

tap_created=false
install_attempted=false
cleanup() {
  result=$?
  trap - EXIT
  if [[ "$install_attempted" == true ]]; then
    for name in "${formula_names[1]}" "${formula_names[0]}"; do
      if brew list --formula --versions "$name" >/dev/null 2>&1; then
        brew uninstall --formula "$tap/$name" || result=1
        if brew list --formula --versions "$name" >/dev/null 2>&1; then result=1; fi
        if [[ -e "$brew_prefix/opt/$name" || -L "$brew_prefix/opt/$name" ]]; then result=1; fi
      fi
    done
  fi
  if [[ -e "$brew_prefix/bin/codex" || -L "$brew_prefix/bin/codex" ]]; then result=1; fi
  if [[ "$tap_created" == true ]]; then brew untap "$tap" || result=1; fi
  exit "$result"
}
trap cleanup EXIT

brew tap-new --no-git "$tap"
tap_created=true
for name in "${formula_names[@]}"; do
  cp "$formula_directory/$name.rb" "$(brew --repository "$tap")/Formula/$name.rb"
done

# Keep latest installed while testing the keg-only version, so command collisions fail.
for name in "${formula_names[@]}"; do
  formula="$tap/$name"
  install_attempted=true
  brew install --formula "$formula"
  prefix="$(brew --prefix "$formula")"
  if [[ ! "$brew_prefix/bin/codex" -ef "$brew_prefix/opt/enterprise-codex/libexec/codex" ]]; then
    echo 'The latest formula must retain the global codex command.' >&2
    exit 1
  fi
  command_directory="$prefix/bin"
  if [[ "$name" == enterprise-codex ]]; then command_directory="$brew_prefix/bin"; fi
  brew test "$formula"
  PATH="$command_directory:$PATH" python3 "$scripts/smoke-installed.py" \
    --prefix "$prefix" --archive "$archive" --homebrew-formula "$name" \
    --platform "$platform" --version "$version"
done
