$ErrorActionPreference = "Stop"

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$RootDir = Join-Path $ScriptDir ".."

# Backend
Start-Job -ScriptBlock {
    param($RootDir)

    Set-Location $RootDir

    uv run --extra visualizer uvicorn visualizer.backend.app:app `
        --reload `
        --host 0.0.0.0 `
        --port 8000
} -ArgumentList $RootDir

# Frontend
Start-Job -ScriptBlock {
    param($RootDir)

    Set-Location (Join-Path $RootDir "frontend")

    npm run tauri dev
} -ArgumentList $RootDir

Get-Job
Wait-Job *