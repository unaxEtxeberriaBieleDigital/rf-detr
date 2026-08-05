#!/usr/bin/env bash
# Launches the RF-DETR visualizer backend (FastAPI) with auto-reload for local development.
#
# IMPORTANT (GPU): `uv run` re-syncs the environment against uv.lock, which pins the plain
# (CPU-only) PyPI torch wheel. If you installed a CUDA-enabled torch build (see README.md ->
# "Running inference on GPU"), always use `--no-sync` here so that build isn't silently
# downgraded back to CPU on every launch.
set -euo pipefail
cd "$(dirname "$0")/.."
uv run --no-sync --extra visualizer uvicorn visualizer.backend.app:app --reload --host 0.0.0.0 --port 8000
