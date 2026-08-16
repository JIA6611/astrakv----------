[CmdletBinding()]
param(
    [string[]]$Files = @(),

    [string]$FilesCsv = "",

    [string]$User = "zyx",

    [string]$RemoteHost = "10.134.70.250",

    [int]$Port = 10000,

    [string]$RemoteRoot = "/home/zyx/astrakv-W",

    [switch]$DryRun
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

# The active DGX checkout is astrakv-W.  Normalize the previous mistaken
# checkout name so an old command line cannot silently sync to the wrong tree.
if ($RemoteRoot.TrimEnd("/") -eq "/home/zyx/astrakv2") {
    Write-Warning "RemoteRoot /home/zyx/astrakv2 is obsolete; using /home/zyx/astrakv-W."
    $RemoteRoot = "/home/zyx/astrakv-W"
}

$repoRoot = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)

function Normalize-RelativePath {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Value
    )

    $trimmed = $Value.Trim()
    if ([string]::IsNullOrWhiteSpace($trimmed)) {
        throw "File path cannot be empty."
    }
    return ($trimmed -replace "\\", "/").TrimStart("./")
}

function Resolve-FileSpec {
    param(
        [Parameter(Mandatory = $true)]
        [string]$RelativePath
    )

    $normalized = Normalize-RelativePath -Value $RelativePath
    $localPath = Join-Path $repoRoot ($normalized -replace "/", [IO.Path]::DirectorySeparatorChar)
    if (-not (Test-Path -LiteralPath $localPath -PathType Leaf)) {
        throw "Local file not found: $RelativePath"
    }
    $resolvedLocal = (Resolve-Path -LiteralPath $localPath).Path
    $remotePath = ($RemoteRoot.TrimEnd("/") + "/" + $normalized)
    $remoteDir = $remotePath -replace "/[^/]+$", ""
    [pscustomobject]@{
        RelativePath = $normalized
        LocalPath = $resolvedLocal
        RemotePath = $remotePath
        RemoteDir = $remoteDir
    }
}

function Quote-ForSingleQuotedShell {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Value
    )

    return "'" + $Value + "'"
}

$fileSpecs = @($Files)
if (-not [string]::IsNullOrWhiteSpace($FilesCsv)) {
    $fileSpecs += @($FilesCsv.Split(",") | ForEach-Object { $_.Trim() } | Where-Object { $_ })
}

if ($fileSpecs.Count -eq 0) {
    throw "At least one file must be provided via -Files or -FilesCsv."
}

$resolvedFiles = @()
foreach ($file in $fileSpecs) {
    $resolvedFiles += Resolve-FileSpec -RelativePath $file
}

$remoteDirs = @($resolvedFiles | ForEach-Object { $_.RemoteDir } | Sort-Object -Unique)
$remoteUserHost = "$User@$RemoteHost"

Write-Host "Repo root: $repoRoot"
Write-Host "Remote root: $RemoteRoot"
Write-Host "Files to sync:"
foreach ($item in $resolvedFiles) {
    Write-Host (" - {0}" -f $item.RelativePath)
}

if ($DryRun) {
    Write-Host "Dry run enabled; skipping ssh/scp."
    return
}

$quotedDirs = @($remoteDirs | ForEach-Object { Quote-ForSingleQuotedShell -Value $_ })
$mkdirScript = "mkdir -p -- " + ($quotedDirs -join " ")
& ssh "-p" $Port $remoteUserHost $mkdirScript
if ($LASTEXITCODE -ne 0) {
    throw "Failed to create remote directories on $remoteUserHost."
}

foreach ($item in $resolvedFiles) {
    & scp "-P" $Port $item.LocalPath ("{0}:{1}" -f $remoteUserHost, $item.RemotePath)
    if ($LASTEXITCODE -ne 0) {
        throw "Failed to copy $($item.RelativePath) to $remoteUserHost."
    }
}

Write-Host ""
Write-Host "Synced file list:"
foreach ($item in $resolvedFiles) {
    Write-Host (" - {0}" -f $item.RelativePath)
}
