<# 
.SYNOPSIS
  Fetch and unpack the official llama.cpp Vulkan build for Windows 11.
.DESCRIPTION
  Downloads a tagged llama.cpp Vulkan archive, verifies its SHA256 checksum (if provided or discoverable),
  and extracts binaries into ./bin. Optionally scaffolds config/models.toml.
.NOTES
  Run: Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
#>

[CmdletBinding(SupportsShouldProcess=$true)]
param(
  [string]$VersionTag = "b3532",
  [string]$DownloadUri = "",
  [string]$Sha256 = "",
  [switch]$Force
)

$ErrorActionPreference = "Stop"
try { [Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12 } catch {}

$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$repoRoot  = Resolve-Path (Join-Path $scriptDir "..")
$binDir    = Join-Path $repoRoot "bin"
$configDir = Join-Path $repoRoot "config"
$modelsConfigPath = Join-Path $configDir "models.toml"
$tempDir   = Join-Path ([System.IO.Path]::GetTempPath()) ("llama.cpp-" + [System.Guid]::NewGuid().ToString())

function Ensure-Directory {
  param([Parameter(Mandatory=$true)][string]$Path)
  if (-not (Test-Path -LiteralPath $Path)) {
    Write-Host ('[i] Creating directory: {0}' -f $Path) -ForegroundColor Cyan
    New-Item -ItemType Directory -Path $Path -Force | Out-Null
  }
}

function Resolve-ReleaseAsset {
  param([Parameter(Mandatory=$true)][string]$Tag)
  $releaseApi = ('https://api.github.com/repos/ggerganov/llama.cpp/releases/tags/{0}' -f $Tag)
  $headers = @{ 'User-Agent' = 'duel-of-minds-setup' }
  Write-Host ('[i] Resolving llama.cpp release {0} via {1}' -f $Tag, $releaseApi) -ForegroundColor Cyan
  $release = Invoke-RestMethod -Uri $releaseApi -Headers $headers
  $asset = $release.assets | Where-Object { $_.name -match '(?i)vulkan' -and $_.name -match '(?i)win' -and $_.name -match '\.zip$' } | Sort-Object name | Select-Object -First 1
  if (-not $asset) { throw ("Could not locate a Windows Vulkan *.zip asset in tag '{0}'. Provide -DownloadUri explicitly." -f $Tag) }
  $shaAsset = $release.assets | Where-Object { $_.name -match '\.sha256$' -and ( $_.name -replace '\.sha256$','') -eq $asset.name } | Select-Object -First 1
  return [pscustomobject]@{
    ZipUrl = $asset.browser_download_url
    ZipName = $asset.name
    ShaUrl = if ($shaAsset) { $shaAsset.browser_download_url } else { $null }
    ShaName = if ($shaAsset) { $shaAsset.name } else { $null }
  }
}

Ensure-Directory -Path $binDir
Ensure-Directory -Path $tempDir

try {
  $resolved = $null
  if ([string]::IsNullOrWhiteSpace($DownloadUri)) {
    $resolved = Resolve-ReleaseAsset -Tag $VersionTag
    $DownloadUri = $resolved.ZipUrl
    Write-Host ('[i] Selected asset: {0}' -f $resolved.ZipName) -ForegroundColor Cyan
    if ($resolved.ShaUrl) { Write-Host ('[i] Found sidecar checksum: {0}' -f $resolved.ShaName) -ForegroundColor Cyan }
  } else {
    Write-Host ('[i] Using explicit DownloadUri: {0}' -f $DownloadUri) -ForegroundColor Cyan
  }

  $archivePath = Join-Path $tempDir 'llama-vulkan-win-x64.zip'
  Write-Host '[i] Downloading archive...' -ForegroundColor Cyan
  Invoke-WebRequest -Uri $DownloadUri -OutFile $archivePath -UseBasicParsing

  if ([string]::IsNullOrWhiteSpace($Sha256) -and $resolved -and $resolved.ShaUrl) {
    $shaPath = Join-Path $tempDir $resolved.ShaName
    Write-Host '[i] Fetching SHA256 sidecar...' -ForegroundColor Cyan
    Invoke-WebRequest -Uri $resolved.ShaUrl -OutFile $shaPath -UseBasicParsing
    $firstLine = (Get-Content -Path $shaPath -TotalCount 1).Trim()
    if ($firstLine) { $Sha256 = ($firstLine -split '\s+')[0] }
    if (-not $Sha256 -or $Sha256.Length -lt 32) { Write-Host '[!] Sidecar .sha256 not plausible; skipping verification.' -ForegroundColor Yellow }
    else { Write-Host '[i] Using SHA256 from sidecar.' -ForegroundColor Cyan }
  }

  if ($Sha256) {
    $computed = (Get-FileHash -Path $archivePath -Algorithm SHA256).Hash.ToLowerInvariant()
    if ($computed -ne $Sha256.ToLowerInvariant()) { throw ("Checksum mismatch. Expected {0} but computed {1}." -f $Sha256, $computed) }
    Write-Host '[+] Checksum OK' -ForegroundColor Green
  } else {
    Write-Host '[!] No SHA256 provided or resolved; continuing without verification.' -ForegroundColor Yellow
  }

  $extractDir = Join-Path $tempDir 'extract'
  Ensure-Directory -Path $extractDir
  Write-Host '[i] Extracting archive...' -ForegroundColor Cyan
  Expand-Archive -Path $archivePath -DestinationPath $extractDir -Force

  $candidateBinaries = Get-ChildItem -Path $extractDir -Recurse -File -Include 'llama.cpp.exe','llama-server.exe','ggml-vulkan.dll','ggml*.dll'
  if (-not $candidateBinaries -or $candidateBinaries.Count -eq 0) { throw 'Archive did not contain expected llama.cpp Vulkan executables or ggml Vulkan DLLs.' }

  foreach ($file in $candidateBinaries) {
    $targetPath = Join-Path $binDir $file.Name
    if ((Test-Path -LiteralPath $targetPath) -and -not $Force) {
      Write-Host ('[!] Skipping existing {0}. Use -Force to overwrite.' -f $file.Name) -ForegroundColor Yellow
      continue
    }
    Copy-Item -LiteralPath $file.FullName -Destination $targetPath -Force
    Write-Host ('[+] Installed {0}' -f $file.Name) -ForegroundColor Green
  }

  if (-not (Test-Path -LiteralPath $modelsConfigPath)) {
    Write-Host '[i] Scaffolding config/models.toml' -ForegroundColor Cyan
    Ensure-Directory -Path $configDir
    $nl = [Environment]::NewLine
    $modelsConfigTemplate = @(
      '# Duel of Minds model registry (generated)',
      '#',
      '# Each [[models]] entry maps an alias to an absolute path and the expected SHA256 checksum',
      '# of a trusted GGUF file on your machine. Store weights outside the repo and update this file',
      '# with real metadata before launching the llamacpp backend.',
      '',
      '[[models]]',
      'alias = "example-deepseek-q5"',
      'path = "D:/ai/models/deepseek/deepseek-q5.gguf"',
      'sha256 = "replace-with-real-sha"'
    ) -join $nl
    Set-Content -Path $modelsConfigPath -Value $modelsConfigTemplate -Encoding UTF8
  }

  Write-Host '[✓] llama.cpp Vulkan installation complete.' -ForegroundColor Green
  Write-Host ('    Binaries available under: {0}' -f $binDir) -ForegroundColor Green
}
catch {
  # No nested subexpressions; no quotes inside the format braces.
  $msg = $_.Exception.Message
  Write-Error ('[x] Error: {0}' -f $msg)
  throw
}
finally {
  if (Test-Path -LiteralPath $tempDir) {
    try { Remove-Item -LiteralPath $tempDir -Recurse -Force -ErrorAction SilentlyContinue } catch {}
  }
}
