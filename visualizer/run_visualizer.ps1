# Launches the RF-DETR visualizer backend (FastAPI) with auto-reload for local development.

$ErrorActionPreference = "Stop"

# Ir al directorio raíz del proyecto (equivalente a cd "$(dirname "$0")/..")
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location (Join-Path $ScriptDir "..")

uv run --extra visualizer uvicorn visualizer.backend.app:app `
    --reload `
    --host 0.0.0.0 `
    --port 8000