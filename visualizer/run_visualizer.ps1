$ErrorActionPreference = "Stop"

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$RootDir = Join-Path $ScriptDir ".."

# Backend
#
# IMPORTANT (GPU): `uv run` re-syncs the environment against uv.lock, which pins the plain
# (CPU-only) PyPI torch wheel. If you installed a CUDA-enabled torch build (see README.md ->
# "Running inference on GPU"), always use `--no-sync` here so that build isn't silently
# downgraded back to CPU on every launch.
Start-Job -ScriptBlock {
    param($RootDir)

    Set-Location $RootDir

    # uv run --no-sync --extra visualizer uvicorn visualizer.backend.app:app --reload --host 0.0.0.0 --port 8000
    uv run --no-sync --extra visualizer uvicorn visualizer.backend.app:app `
        --reload `
        --host 0.0.0.0 `
        --port 8000
} -ArgumentList $RootDir

# Frontend
Start-Job -ScriptBlock {
    param($RootDir)

    Set-Location (Join-Path $RootDir "visualizer\frontend")

    npm run tauri dev
} -ArgumentList $RootDir

Get-Job
Wait-Job *