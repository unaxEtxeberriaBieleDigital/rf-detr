# Runs tests/fiftyone_embeddings_app.py using the project's venv Python directly.
#
# ALWAYS use this wrapper (or `.venv\Scripts\python.exe` directly) instead of
# `uv run python tests\fiftyone_embeddings_app.py`. `uv run`/`uv sync` resync the
# environment against pyproject.toml's lockfile, which silently reverts the
# manually installed CUDA-enabled torch build back to the CPU-only pinned
# version, and can similarly clobber other manually installed packages
# (fiftyone, fiftyone-brain, pycocotools) that aren't tracked as project
# dependencies.
#
# Usage:
#   .\tests\run_fiftyone_app.ps1

$ErrorActionPreference = "Stop"

$repoRoot = Split-Path -Parent $PSScriptRoot
$python = Join-Path $repoRoot ".venv\Scripts\python.exe"
$script = Join-Path $repoRoot "tests\fiftyone_embeddings_app.py"

if (-not (Test-Path $python)) {
    throw "Could not find venv Python at '$python'. Run 'uv sync --all-groups' first."
}

& $python $script @args
exit $LASTEXITCODE
