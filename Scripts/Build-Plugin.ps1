param(
    [Parameter(Mandatory = $true)][string]$EngineRoot,
    [Parameter(Mandatory = $true)][string]$OutputDirectory
)
$ErrorActionPreference = 'Stop'
$buildEngine = (Resolve-Path -LiteralPath $EngineRoot).Path
$buildPlugin = Split-Path -Parent $PSScriptRoot
$buildOutput = [IO.Path]::GetFullPath($OutputDirectory)
if (Test-Path -LiteralPath $buildOutput) {
    if ((Get-ChildItem -LiteralPath $buildOutput -Force | Measure-Object).Count -gt 0) {
        throw 'OutputDirectory must be empty. Existing files will not be replaced.'
    }
}
$buildVersion = Get-Content -LiteralPath (Join-Path $buildEngine 'Engine\Build\Build.version') -Raw | ConvertFrom-Json
if ($buildVersion.MajorVersion -ne 5 -or $buildVersion.MinorVersion -ne 4) {
    throw 'This release targets Unreal Engine 5.4. Validation was performed with 5.4.4.'
}
$buildTool = Join-Path $buildEngine 'Engine\Build\BatchFiles\RunUAT.bat'
& $buildTool BuildPlugin "-Plugin=$buildPlugin\CardistryCapture.uplugin" "-Package=$buildOutput" -TargetPlatforms=Win64 -Rocket
if ($LASTEXITCODE -ne 0) { throw "Plugin compilation failed (exit $LASTEXITCODE)." }
Write-Output "Plugin compilation completed: $buildOutput"
Write-Output 'Run Scripts/Setup.cmd in the final installation directory to prepare video processing.'
