<#
.SYNOPSIS
    Fetch and unpack the official llama.cpp Vulkan build for Windows 11.
.DESCRIPTION
    Downloads a tagged llama.cpp Vulkan archive, verifies its SHA256 checksum (if provided),
    and extracts the binaries into the repo's ./bin directory so Duel of Minds can launch
    them directly when the Python bindings are unavailable.
    Optionally scaffolds config/models.toml if it does not yet exist.
.PARAMETER VersionTag
    Git tag from ggerganov/llama.cpp releases (e.g. "b3532").
.PARAMETER DownloadUri
    Optional explicit URI to the llama.cpp Vulkan archive. When omitted the script looks up the
    asset from the GitHub release metadata for VersionTag.
.PARAMETER Sha256
    Optional expected SHA256 hash for the downloaded archive. When supplied the archive hash must match.
.PARAMETER Force
    Overwrite any existing binaries in ./bin when set.
.EXAMPLE
    PS> ./scripts/Install-LlamaCppVulkan.ps1 -VersionTag b3532
.EXAMPLE
    PS> ./scripts/Install-LlamaCppVulkan.ps1 -DownloadUri "https://huggingface.co/ggml-org/models/resolve/main/llama.cpp/win/vulkan/llama-vulkan-win-x64.zip" -Sha256 "..."
.NOTES
    Run from a PowerShell prompt on Windows 11 with execution policy permitting local scripts:
        Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
#>

param(
    [string]$VersionTag = "b3532",
    [string]$DownloadUri = "",
    [string]$Sha256 = "",
    [switch]$Force
)

$ErrorActionPreference = "Stop"

$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$repoRoot = Resolve-Path (Join-Path $scriptDir "..")
$binDir   = Join-Path $repoRoot "bin"
$configDir = Join-Path $repoRoot "config"
$modelsConfigPath = Join-Path $configDir "models.toml"
$tempDir = Join-Path ([System.IO.Path]::GetTempPath()) ("llama.cpp-" + [System.Guid]::NewGuid().ToString())

function Ensure-Directory {
    param([string]$Path)
    if (-not (Test-Path -LiteralPath $Path)) {
        Write-Host "[+] Creating $Path" -ForegroundColor Green
        New-Item -ItemType Directory -Path $Path | Out-Null
    }
}

Ensure-Directory -Path $binDir
Ensure-Directory -Path $tempDir

try {
    if (-not $DownloadUri) {
        $releaseApi = "https://api.github.com/repos/ggerganov/llama.cpp/releases/tags/$VersionTag"
        Write-Host "[i] Resolving llama.cpp Vulkan asset from $releaseApi" -ForegroundColor Cyan
        $headers = @{ "User-Agent" = "duel-of-minds-setup" }
        $release = Invoke-RestMethod -Uri $releaseApi -Headers $headers
        $asset = $release.assets | Where-Object { $_.name -match "vulkan" -and $_.name -match "win" -and $_.name -match "zip$" } | Select-Object -First 1
        if (-not $asset) {
            throw "Could not locate a Vulkan Windows asset on release $VersionTag. Specify -DownloadUri explicitly."
        }
        $DownloadUri = $asset.browser_download_url
        if (-not $Sha256 -and $asset.name -match "(?i)sha256") {
            $Sha256 = ($asset.name -replace ".*sha256-", "")
        }
        Write-Host "[i] Selected asset: $($asset.name)" -ForegroundColor Cyan
    }

    $archivePath = Join-Path $tempDir "llama.cpp-vulkan.zip"
    Write-Host "[i] Downloading $DownloadUri" -ForegroundColor Cyan
    Invoke-WebRequest -Uri $DownloadUri -OutFile $archivePath

    if ($Sha256) {
        Write-Host "[i] Verifying SHA256 checksum" -ForegroundColor Cyan
        $hash = (Get-FileHash -Path $archivePath -Algorithm SHA256).Hash.ToLowerInvariant()
        if ($hash -ne $Sha256.ToLowerInvariant()) {
            throw "Checksum mismatch. Expected $Sha256 but found $hash."
        }
        Write-Host "[+] Checksum OK" -ForegroundColor Green
    } else {
        Write-Host "[!] No SHA256 provided. Consider supplying -Sha256 for integrity." -ForegroundColor Yellow
    }

    $extractDir = Join-Path $tempDir "extract"
    Ensure-Directory -Path $extractDir
    Write-Host "[i] Extracting archive" -ForegroundColor Cyan
    Expand-Archive -Path $archivePath -DestinationPath $extractDir -Force

    $candidateBinaries = Get-ChildItem -Path $extractDir -Recurse -Include "llama.cpp.exe", "llama-server.exe", "ggml-vulkan.dll", "ggml*.dll"
    if (-not $candidateBinaries) {
        throw "Archive did not contain expected llama.cpp Vulkan executables."
    }

    foreach ($file in $candidateBinaries) {
        $targetPath = Join-Path $binDir $file.Name
        if ((Test-Path $targetPath) -and -not $Force) {
            Write-Host "[!] Skipping existing $($file.Name). Use -Force to overwrite." -ForegroundColor Yellow
            continue
        }
        Copy-Item -Path $file.FullName -Destination $targetPath -Force
        Write-Host "[+] Installed $($file.Name)" -ForegroundColor Green
    }

    if (-not (Test-Path $modelsConfigPath)) {
        Write-Host "[i] Scaffolding config/models.toml" -ForegroundColor Cyan
        Ensure-Directory -Path $configDir
        $modelsConfigTemplate = @'
# Duel of Minds model registry (generated)
#
# Each [[models]] entry maps an alias to an absolute path and the expected SHA256 checksum
# of a trusted GGUF file on your machine. Store weights outside the repo and update this file
# with real metadata before launching the llamacpp backend.

[[models]]
alias = "example-deepseek-q5"
path = "D:/ai/models/deepseek/deepseek-q5.gguf"
sha256 = "replace-with-real-sha"
'@
        $modelsConfigTemplate | Set-Content -Path $modelsConfigPath -Encoding UTF8
    }

    Write-Host "[✓] llama.cpp Vulkan installation complete." -ForegroundColor Green
    Write-Host "    Binaries available under: $binDir" -ForegroundColor Green
}
finally {
    if (Test-Path $tempDir) {
        Remove-Item -Path $tempDir -Recurse -Force -ErrorAction SilentlyContinue
    }
}
