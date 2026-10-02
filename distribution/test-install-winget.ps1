# Run only on disposable CI runners: this installs and uninstalls a user package.
param(
    [Parameter(Mandatory)][string]$ManifestDirectory,
    [Parameter(Mandatory)][string]$Archive,
    [Parameter(Mandatory)][string]$Version,
    [Parameter(Mandatory)][string]$Platform
)
$ErrorActionPreference = 'Stop'
$env:RUN_MESHAGENT_CLOUD_SMOKE = '1'
$installation = Join-Path $env:RUNNER_TEMP 'Enterprise Codex installation'
$links = Join-Path $env:LOCALAPPDATA 'Microsoft\WinGet\Links'
$command = Join-Path $links 'codex.exe'

if ((Test-Path $installation) -or (Test-Path $command)) {
    throw 'Installation tests require a runner without a conflicting Codex installation.'
}

winget validate --manifest $ManifestDirectory --disable-interactivity
if ($LASTEXITCODE -ne 0) { throw 'WinGet manifest validation failed.' }
winget settings --enable LocalManifestFiles
if ($LASTEXITCODE -ne 0) { throw 'Could not enable local manifest installation.' }

$installed = $false
try {
    # Let WinGet choose the native installer; the smoke test checks its PE architecture.
    winget install --manifest $ManifestDirectory --scope user --location $installation `
        --accept-package-agreements --accept-source-agreements --disable-interactivity
    if ($LASTEXITCODE -ne 0) { throw 'WinGet installation failed.' }
    $installed = $true
    $env:PATH = "$links;$env:PATH"
    python "$PSScriptRoot/smoke-installed.py" --prefix $installation --archive $Archive `
        --platform $Platform --version $Version
    if ($LASTEXITCODE -ne 0) { throw 'Installed Windows package smoke test failed.' }
} finally {
    if ($installed) {
        winget uninstall --manifest $ManifestDirectory --scope user --purge `
            --accept-source-agreements --disable-interactivity
        if ($LASTEXITCODE -ne 0) { throw 'WinGet uninstall failed.' }
        if ((Test-Path $command) -or (Test-Path (Join-Path $installation 'codex.exe'))) {
            throw 'WinGet uninstall left the executable or command alias installed.'
        }
    }
}
