<#
.SYNOPSIS
    Append or update a GGUF model entry in config/models.toml with a verified SHA256 hash.
.DESCRIPTION
    Accepts or prompts for model metadata, computes the SHA256 checksum, and writes
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

[CmdletBinding()]
param(
    [string]$Alias = "",
    [string]$ModelPath = "",
    [switch]$Force
)

$ErrorActionPreference = "Stop"

$scriptDir  = Split-Path -Parent $MyInvocation.MyCommand.Path
$repoRoot   = Resolve-Path (Join-Path $scriptDir "..")
$configPath = Join-Path $repoRoot "config/models.toml"

if (-not (Test-Path -LiteralPath $configPath)) {
    throw "config/models.toml not found. Run Install-LlamaCppVulkan.ps1 first or create the file manually."
}

if (-not $Alias) {
    $Alias = Read-Host "Enter model alias (letters, numbers, dashes)"
}
if ([string]::IsNullOrWhiteSpace($Alias)) {
    throw "Alias cannot be empty."
}

if (-not $ModelPath) {
    $ModelPath = Read-Host "Enter full path to the GGUF model"
}

# Expand ~ and environment variables
$expandedPath = [System.Environment]::ExpandEnvironmentVariables($ModelPath)
if ($expandedPath.StartsWith("~")) {
    $home = [Environment]::GetFolderPath("UserProfile")
    $expandedPath = Join-Path $home $expandedPath.TrimStart("~", "\", "/")
}

if (-not (Test-Path -LiteralPath $expandedPath)) {
    throw ("Model path does not exist: {0}" -f $expandedPath)
}

$ModelPath = [System.IO.Path]::GetFullPath($expandedPath)

Write-Host ("[i] Calculating SHA256 for {0}" -f $ModelPath) -ForegroundColor Cyan
$hash = (Get-FileHash -Path $ModelPath -Algorithm SHA256).Hash.ToLowerInvariant()

# Load and normalize line endings to LF (\n) for predictable regex behavior
$toml = (Get-Content -LiteralPath $configPath -Raw)
$toml = $toml -replace "`r`n","`n"

# Escape alias for regex safety
$aliasEsc = [System.Text.RegularExpressions.Regex]::Escape($Alias)

# Build entry text (use backtick-quote `")
$entryHeader = "[[models]]`nalias = `"$Alias`""
$entryBody   = "path = `"$ModelPath`"`nsha256 = `"$hash`"`n"
$entryFull   = "$entryHeader`n$entryBody"

# Pattern: existing block starting at [[models]] then alias="..."; up to next [[models]] alias or EOF
$pattern = "\[\[models\]\]\s*`nalias\s*=\s*`"$aliasEsc`"[\s\S]*?(?=(?:\n)\[\[models\]\]\s*`nalias|\z)"

if ($toml -match "\[\[models\]\]\s*`nalias\s*=\s*`"$aliasEsc`"") {
    if (-not $Force) {
        throw ("Alias '{0}' already exists. Use -Force to overwrite." -f $Alias)
    }
    Write-Host ("[!] Replacing existing alias '{0}'" -f $Alias) -ForegroundColor Yellow

    $updated = [System.Text.RegularExpressions.Regex]::Replace(
        $toml,
        $pattern,
        [System.Text.RegularExpressions.MatchEvaluator]{ param($m) $entryFull }
    )

    Set-Content -LiteralPath $configPath -Value $updated -Encoding UTF8
} else {
    Write-Host ("[+] Appending alias '{0}'" -f $Alias) -ForegroundColor Green
    if ($toml.Length -gt 0 -and -not $toml.EndsWith("`n")) {
        Add-Content -LiteralPath $configPath -Value "`n" -Encoding UTF8
    }
    Add-Content -LiteralPath $configPath -Value $entryFull -Encoding UTF8
}

Write-Host ("[OK] Added {0} with SHA256 {1}" -f $Alias, $hash) -ForegroundColor Green
