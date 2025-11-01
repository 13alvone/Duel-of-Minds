param(
    [Parameter(Mandatory = $true)]
    [string]$DatabasePath,

    [string]$OutputRoot = "outputs",

    [int[]]$RunIds = @(),

    [switch]$AllRuns,

    [int]$LimitTurns = 0,

    [int]$QuoteThreshold = 200,

    [int]$TcrStride = 3
)

$ErrorActionPreference = "Stop"
$matrixLabels = @("2h", "8h", "24h")
$timestamp = (Get-Date).ToUniversalTime().ToString("yyyyMMdd-HHmmss")

foreach ($label in $matrixLabels) {
    $arguments = @("eval", $DatabasePath, "--output-root", $OutputRoot, "--matrix-label", $label, "--timestamp", $timestamp)
    if ($LimitTurns -gt 0) {
        $arguments += @("--limit-turns", $LimitTurns)
    }
    if ($QuoteThreshold -gt 0) {
        $arguments += @("--quote-threshold", $QuoteThreshold)
    }
    if ($TcrStride -gt 0) {
        $arguments += @("--tcr-stride", $TcrStride)
    }
    if ($AllRuns.IsPresent) {
        $arguments += "--all"
    } elseif ($RunIds.Length -gt 0) {
        foreach ($rid in $RunIds) {
            $arguments += @("--run-id", $rid)
        }
    }

    Write-Host "[i] Running dom eval for matrix slot '$label'..." -ForegroundColor Cyan
    python .\duel_of_minds.py @arguments
    if ($LASTEXITCODE -ne 0) {
        throw "dom eval failed for matrix slot '$label'"
    }
}

Write-Host "[i] Evaluation matrix completed." -ForegroundColor Green
