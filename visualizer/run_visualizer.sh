#!/usr/bin/env bash
# Launches the RF-DETR visualizer backend (FastAPI) with auto-reload for local development.
set -euo pipefail
cd "$(dirname "$0")/.."
uv run --extra visualizer uvicorn visualizer.backend.app:app --reload --host 0.0.0.0 --port 8000
