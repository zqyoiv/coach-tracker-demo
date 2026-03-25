# Same behavior as process-video.sh — use this from Windows PowerShell (not Git Bash paths).
#
# Examples:
#   .\process-video.ps1 3-13 "C:\Users\vioyq\Desktop\Coach_Tracker\_Coach_Video" 3 4 5
#   $env:COACH_VIDEO_ROOT = "C:\Users\vioyq\Desktop\Coach_Tracker\_Coach_Video"
#   .\process-video.ps1 3-13 3 4 5
#
# Default video root if unset: $env:USERPROFILE\coach-raw-video

$ErrorActionPreference = "Stop"

if ($args.Count -lt 1) {
    Write-Host "Error: specify a date (e.g. .\process-video.ps1 3-13)" -ForegroundColor Red
    exit 1
}

$TargetDate = $args[0]
$rest = @()
if ($args.Count -gt 1) {
    $rest = $args[1..($args.Count - 1)]
}

$defaultRoot = Join-Path $env:USERPROFILE "coach-raw-video"
if ($env:COACH_VIDEO_ROOT) {
    $VideoRoot = $env:COACH_VIDEO_ROOT
} else {
    $VideoRoot = $defaultRoot
}
$Coaches = @()

if ($rest.Count -eq 0) {
    # keep VideoRoot
} elseif (Test-Path -LiteralPath $rest[0] -PathType Container) {
    $VideoRoot = $rest[0]
    if ($rest.Count -gt 1) {
        $Coaches = $rest[1..($rest.Count - 1)]
    }
} else {
    $Coaches = $rest
}

$BasePath = Join-Path $VideoRoot $TargetDate
$ScriptDir = $PSScriptRoot
$PythonScript = Join-Path $ScriptDir "coach-raw-video-concat-hours.py"

Write-Host "Starting processing for Date: $TargetDate"
Write-Host "   VIDEO_ROOT=$VideoRoot"
if ($Coaches.Count -gt 0) {
    Write-Host "   COACHES only: $($Coaches -join ' ')"
}

if (-not (Test-Path -LiteralPath $BasePath -PathType Container)) {
    Write-Host "Error: directory does not exist: $BasePath" -ForegroundColor Red
    exit 1
}

Write-Host "Step 1: Merging videos into 1-hour chunks..."
$pyArgs = @(
    $PythonScript,
    "--root", $VideoRoot,
    "--dates", $TargetDate
)
if ($Coaches.Count -gt 0) {
    $pyArgs += "--coaches"
    foreach ($c in $Coaches) { $pyArgs += $c }
}

& python @pyArgs
if ($LASTEXITCODE -ne 0) {
    Write-Host "Error: Python merge script failed." -ForegroundColor Red
    exit $LASTEXITCODE
}

function Clear-HourlyFolder {
    param([string]$CoachDirPath)
    $name = Split-Path $CoachDirPath -Leaf
    if (-not (Test-Path -LiteralPath $CoachDirPath -PathType Container)) {
        Write-Host "Warning: no folder $name. Skipping." -ForegroundColor Yellow
        return
    }
    Write-Host "Processing $name..."
    $hourly = Join-Path $CoachDirPath "hourly"
    if (-not (Test-Path -LiteralPath $hourly -PathType Container)) {
        Write-Host "Warning: no hourly folder in $name. Skipping cleanup." -ForegroundColor Yellow
        return
    }
    $hourMp4 = Get-ChildItem -LiteralPath $hourly -File -Filter "*.mp4" -ErrorAction SilentlyContinue
    if (-not $hourMp4 -or $hourMp4.Count -eq 0) {
        Write-Host "Warning: no merged videos in hourly for $name. Skipping cleanup." -ForegroundColor Yellow
        return
    }
    Get-ChildItem -LiteralPath $CoachDirPath -File -Filter "*.mp4" -ErrorAction SilentlyContinue | Remove-Item -Force
    Get-ChildItem -LiteralPath $hourly -File -Filter "*.mp4" | Move-Item -Destination $CoachDirPath -Force
    Remove-Item -LiteralPath $hourly -Force -Recurse
    Write-Host "Done $name"
}

Write-Host "Step 2: Cleaning up and moving files..."
if ($Coaches.Count -gt 0) {
    foreach ($n in $Coaches) {
        $dir = Join-Path $BasePath "Coach-$n"
        Clear-HourlyFolder -CoachDirPath $dir
    }
} else {
    Get-ChildItem -LiteralPath $BasePath -Directory -ErrorAction SilentlyContinue |
        Where-Object { $_.Name -match '^Coach-\d+$' } |
        ForEach-Object { Clear-HourlyFolder -CoachDirPath $_.FullName }
}

Write-Host "All tasks for $TargetDate are finished!"
