<#
.SYNOPSIS
    Append or update a GGUF model entry in config/models.toml with a verified SHA256 hash.
.DESCRIPTION
    Prompts for (or accepts) model metadata, computes the SHA256 checksum, and writes
    an entry compatible with Duel of Minds' llama.cpp backend configuration.
.PARAMETER Alias
    Friendly alias for the model (e.g. deepseek-q5).
.PARAMETER ModelPath
    Absolute path to the GGUF file. ~ expansion is supported.
.PARAMETER Force
    Overwrite an existing entry with the same alias when specified.
.EXAMPLE
    PS> ./scripts/New-ModelEntry.ps1 -Alias deepseek-q6 -ModelPath "D:/ai/models/deepseek/deepseek-q6.gguf"
#>

param(
    [string]$Alias = "",
    [string]$ModelPath = "",
    [switch]$Force
)

$ErrorActionPreference = "Stop"

$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$repoRoot = Resolve-Path (Join-Path $scriptDir "..")
$configPath = Join-Path $repoRoot "config/models.toml"

if (-not (Test-Path $configPath)) {
    throw "config/models.toml not found. Run Install-LlamaCppVulkan.ps1 first or create the file manually."
}

if (-not $Alias) {
    $Alias = Read-Host "Enter model alias (letters, numbers, dashes)"
}

if (-not $ModelPath) {
    $ModelPath = Read-Host "Enter full path to the GGUF model"
}

$expandedPath = [System.Environment]::ExpandEnvironmentVariables($ModelPath)
if ($expandedPath.StartsWith("~")) {
    $home = [Environment]::GetFolderPath("UserProfile")
    $expandedPath = (Join-Path $home $expandedPath.TrimStart("~", "\", "/"))
}

if (-not (Test-Path -LiteralPath $expandedPath)) {
    throw "Model path does not exist: $expandedPath"
}

$ModelPath = [System.IO.Path]::GetFullPath($expandedPath)

Write-Host "[i] Calculating SHA256 for $ModelPath" -ForegroundColor Cyan
$hash = (Get-FileHash -Path $ModelPath -Algorithm SHA256).Hash.ToLowerInvariant()

$toml = (Get-Content -Path $configPath -Raw).Replace("`r`n", "`n")
$entryHeader = "[[models]]`nalias = \"$Alias\""

if ($toml -match "\[\[models\]\]\s*`nalias\s*=\s*\"$Alias\"") {
    if (-not $Force) {
        throw "Alias '$Alias' already exists. Use -Force to overwrite."
    }
    Write-Host "[!] Replacing existing alias '$Alias'" -ForegroundColor Yellow
    $pattern = "\[\[models\]\]\s*`nalias\s*=\s*\"$Alias\"[\s\S]*?(?=\n\[\[models\]\]\s*`nalias|\z)"
    $replacement = "[[models]]`nalias = \"$Alias\"`npath = \"$ModelPath\"`nsha256 = \"$hash\"`n"
    $updated = [System.Text.RegularExpressions.Regex]::Replace($toml, $pattern, $replacement)
    Set-Content -Path $configPath -Value $updated -Encoding UTF8
} else {
    Write-Host "[+] Appending alias '$Alias'" -ForegroundColor Green
    Add-Content -Path $configPath -Value "`n[[models]]`nalias = \"$Alias\"`npath = \"$ModelPath\"`nsha256 = \"$hash\"`n"
}

Write-Host "[✓] Added $Alias with SHA256 $hash" -ForegroundColor Green
