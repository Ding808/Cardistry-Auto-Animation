[CmdletBinding()]
param(
    [string]$Python,
    [string]$RuntimeArchive,
    [string]$ManoArchive,
    [string]$WilorCheckpoint,
    [string]$WilorConfig,
    [switch]$AcceptedRuntimeLicenses,
    [switch]$AcceptedModelLicenses,
    [switch]$RuntimeOnly,
    [switch]$CheckOnly,
    [switch]$NonInteractive
)

$ErrorActionPreference = 'Stop'
$pluginRoot = Split-Path -Parent $PSScriptRoot
$pipelineRoot = Join-Path $pluginRoot 'PythonPipeline'
$venvPython = Join-Path $pipelineRoot '.venv\Scripts\python.exe'

function Get-SavedLicenseAcceptance($LicenseRecord, [string]$RuntimeNoticeHash) {
    return @{
        Runtime = ($RuntimeNoticeHash.Length -eq 64 -and $LicenseRecord.runtime_license_notice_sha256 -eq $RuntimeNoticeHash)
        Models = ($LicenseRecord.accepted_model_licenses -is [bool] -and $LicenseRecord.accepted_model_licenses -eq $true -and $LicenseRecord.model_terms_commit -eq 'fcb911312a38fa8badd30d9656a167485d61b8f9')
    }
}

function Select-Asset([string]$Title, [string]$Filter) {
    Add-Type -AssemblyName System.Windows.Forms
    $dialog = New-Object System.Windows.Forms.OpenFileDialog
    try {
        $dialog.Title = $Title
        $dialog.Filter = $Filter
        $dialog.CheckFileExists = $true
        $dialog.Multiselect = $false
        if ($dialog.ShowDialog() -ne [System.Windows.Forms.DialogResult]::OK) {
            throw "No file selected: $Title. Setup was cancelled."
        }
        return $dialog.FileName
    } finally { $dialog.Dispose() }
}

try {
    if (-not [Environment]::Is64BitOperatingSystem) { throw 'Windows x64 is required.' }
    $licenseRecordPath = Join-Path $pipelineRoot '.runtime-cache\license-acceptance.json'
    $runtimeNoticePath = Join-Path $pluginRoot 'ThirdParty\MSVC\README.md'
    if (-not $CheckOnly -and (Test-Path -LiteralPath $licenseRecordPath)) {
        $licenseRecord = Get-Content -LiteralPath $licenseRecordPath -Raw | ConvertFrom-Json
        $noticeHash = (Get-FileHash -LiteralPath $runtimeNoticePath -Algorithm SHA256).Hash.ToLowerInvariant()
        $savedAcceptance = Get-SavedLicenseAcceptance $licenseRecord $noticeHash
        if ($savedAcceptance.Runtime) {
            $AcceptedRuntimeLicenses = $true
        }
        if ($savedAcceptance.Models) {
            $AcceptedModelLicenses = $true
        }
    }
    if (-not $CheckOnly -and -not $AcceptedRuntimeLicenses) {
        if ($NonInteractive) { throw 'Read ThirdParty/MSVC/README.md and pass -AcceptedRuntimeLicenses, or run Setup.cmd interactively.' }
        Write-Host 'The runtime contains separately licensed third-party libraries and Microsoft runtime components.'
        Write-Host 'The plugin license does not replace their terms. Microsoft components may only be used with this application.'
        Write-Host "Full runtime distribution terms: $runtimeNoticePath"
        Write-Host "Other notices: $(Join-Path $pluginRoot 'THIRD_PARTY_NOTICES.md')"
        while (-not $AcceptedRuntimeLicenses) {
            $runtimeChoice = Read-Host 'Type OPEN to view the full terms, ACCEPT to accept them, or press Enter to cancel'
            if ($runtimeChoice -ceq 'OPEN') {
                Start-Process -FilePath 'notepad.exe' -ArgumentList ('"' + $runtimeNoticePath + '"')
            } elseif ($runtimeChoice -ceq 'ACCEPT') {
                $AcceptedRuntimeLicenses = $true
            } else { throw 'Runtime terms were not accepted. Nothing was installed.' }
        }
    }
    if (-not $CheckOnly -and -not $RuntimeOnly) {
        if (-not $AcceptedModelLicenses) {
            if ($NonInteractive) { throw 'Use -AcceptedModelLicenses after reading the model terms, or use -RuntimeOnly.' }
            Write-Host 'Hand reconstruction requires your separately licensed MANO and WiLoR files.'
            Write-Host 'Read the applicable terms before continuing:'
            Write-Host '  https://mano.is.tue.mpg.de/'
            Write-Host '  https://github.com/rolpotamias/WiLoR/blob/fcb911312a38fa8badd30d9656a167485d61b8f9/license.txt'
            Write-Host '  https://github.com/vchoutas/smplx/blob/main/LICENSE'
            Write-Host 'These terms are separate from the plugin license and restrict use and redistribution.'
            $acceptance = Read-Host 'Type ACCEPT if you have read and accept the applicable terms. Otherwise press Enter for runtime-only setup'
            if ($acceptance -ceq 'ACCEPT') { $AcceptedModelLicenses = $true }
            else { $RuntimeOnly = $true }
        }
        if (-not $RuntimeOnly) {
            if (-not $ManoArchive -and (-not (Test-Path -LiteralPath (Join-Path $pipelineRoot 'models\MANO_LEFT.pkl')) -or -not (Test-Path -LiteralPath (Join-Path $pipelineRoot 'models\MANO_RIGHT.pkl')))) {
                if ($NonInteractive) { throw 'Provide -ManoArchive with your official MANO v1.2 ZIP.' }
                $ManoArchive = Select-Asset 'Select your official MANO v1.2 archive (left and right hands)' 'MANO archive (*.zip)|*.zip'
            }
            if (-not $WilorCheckpoint -and -not (Test-Path -LiteralPath (Join-Path $pipelineRoot 'models\wilor_final.ckpt'))) {
                if ($NonInteractive) { throw 'Provide -WilorCheckpoint with your authorized wilor_final.ckpt.' }
                $WilorCheckpoint = Select-Asset 'Select your authorized wilor_final.ckpt' 'WiLoR checkpoint (*.ckpt)|*.ckpt'
            }
            if (-not $WilorConfig -and -not (Test-Path -LiteralPath (Join-Path $pipelineRoot 'models\model_config.yaml'))) {
                if ($NonInteractive) { throw 'Provide -WilorConfig with the matching model_config.yaml.' }
                $WilorConfig = Select-Asset 'Select the matching model_config.yaml' 'Model configuration (*.yaml;*.yml)|*.yaml;*.yml'
            }
        }
    }
    foreach ($assetPath in @($RuntimeArchive, $ManoArchive, $WilorCheckpoint, $WilorConfig)) {
        if ($assetPath -and -not (Test-Path -LiteralPath $assetPath -PathType Leaf)) { throw "File not found: $assetPath" }
    }
    if (-not (Test-Path -LiteralPath $venvPython)) {
        if ($CheckOnly) { throw 'The isolated runtime is missing. Run Setup.cmd first.' }
        Write-Host 'Creating the plugin Python 3.10 environment...'
        if ($Python) { & $Python -m venv (Join-Path $pipelineRoot '.venv') }
        else { & py -3.10 -m venv (Join-Path $pipelineRoot '.venv') }
        if ($LASTEXITCODE -ne 0) { throw 'Could not create the environment. Install Python 3.10 x64, or pass -Python with its python.exe path.' }
    }
    & $venvPython -c 'import sys,struct; assert sys.version_info[:2] == (3,10) and struct.calcsize(chr(80)) == 8'
    if ($LASTEXITCODE -ne 0) { throw 'The plugin environment has an unsupported Python version.' }
    $setupArguments = @('-X', 'utf8', (Join-Path $pipelineRoot 'setup_runtime.py'))
    if ($RuntimeArchive) { $setupArguments += @('--runtime-archive', (Resolve-Path -LiteralPath $RuntimeArchive).Path) }
    if ($ManoArchive) { $setupArguments += @('--mano-archive', (Resolve-Path -LiteralPath $ManoArchive).Path) }
    if ($WilorCheckpoint) { $setupArguments += @('--wilor-checkpoint', (Resolve-Path -LiteralPath $WilorCheckpoint).Path) }
    if ($WilorConfig) { $setupArguments += @('--wilor-config', (Resolve-Path -LiteralPath $WilorConfig).Path) }
    if ($AcceptedModelLicenses) { $setupArguments += '--accepted-model-licenses' }
    if ($AcceptedRuntimeLicenses) { $setupArguments += '--accepted-runtime-licenses' }
    if ($RuntimeOnly) { $setupArguments += '--runtime-only' }
    if ($CheckOnly) { $setupArguments += '--check-only' }
    & $venvPython @setupArguments
    exit $LASTEXITCODE
} catch {
    Write-Host "Setup failed: $($_.Exception.Message)" -ForegroundColor Red
    exit 1
}
