# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

"""FastAPI backend for the RF-DETR data-cleaning / model-evaluation visualizer.

Run with:
    uv run uvicorn visualizer.backend.app:app --reload --port 8000

Typical flow for the frontend:
    1. GET  /api/v1/model-types, /api/v1/dataset-types   -> discover what's available
    2. POST /api/v1/jobs                                  -> kick off inference + evaluation
    3. GET  /api/v1/jobs/{job_id}                          -> poll until status == "done"
    4. GET  /api/v1/jobs/{job_id}/records?...              -> fetch (filtered) embedding records
    5. GET  /api/v1/jobs/{job_id}/images/{record_id}       -> fetch the underlying image
"""

import threading
import uuid
from pathlib import Path

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel as PydanticModel

# Importing these registers the concrete implementations into MODEL_REGISTRY /
# DATASET_REGISTRY via their `@register_model`/`@register_dataset` decorators. Adding a new
# model or dataset type only requires adding its import here (or an equivalent plugin
# discovery mechanism) - no other endpoint code needs to change.
from visualizer.backend.datasets import cocodetectiondataset  # noqa: F401
from visualizer.backend.datasets.basedataset import Split
from visualizer.backend.jobs import JOB_STORE, Job, run_job
from visualizer.backend.models import rfdetr  # noqa: F401
from visualizer.backend.registry import DATASET_REGISTRY, MODEL_REGISTRY

app = FastAPI(title="RF-DETR Visualizer API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class JobRequest(PydanticModel):
    dataset_path: str
    dataset_type: str
    model_path: str
    model_type: str
    splits: list[str] | None = None  # None = every split found in the dataset
    batch_size: int = 8
    iou_threshold: float = 0.5
    pca_components: int = 2


class JobStatusResponse(PydanticModel):
    id: str
    status: str
    error: str | None = None
    num_records: int = 0


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/api/v1/model-types")
def list_model_types() -> list[str]:
    return sorted(MODEL_REGISTRY)


@app.get("/api/v1/dataset-types")
def list_dataset_types() -> list[str]:
    return sorted(DATASET_REGISTRY)


@app.post("/api/v1/jobs", response_model=JobStatusResponse, status_code=202)
def create_job(request: JobRequest) -> JobStatusResponse:
    if request.dataset_type not in DATASET_REGISTRY:
        raise HTTPException(
            400, f"Unknown dataset_type '{request.dataset_type}'. Available: {sorted(DATASET_REGISTRY)}"
        )
    if request.model_type not in MODEL_REGISTRY:
        raise HTTPException(400, f"Unknown model_type '{request.model_type}'. Available: {sorted(MODEL_REGISTRY)}")

    dataset_cls = DATASET_REGISTRY[request.dataset_type]
    model_cls = MODEL_REGISTRY[request.model_type]

    try:
        dataset = dataset_cls(request.dataset_path)
        model = model_cls(request.model_path)
        splits = [_parse_split(name) for name in request.splits] if request.splits else list(dataset.splits.keys())
    except Exception as e:
        raise HTTPException(400, f"Could not initialize dataset/model: {e}") from e

    job = Job(id=str(uuid.uuid4()))
    JOB_STORE[job.id] = job

    thread = threading.Thread(
        target=run_job,
        args=(job, dataset, model, splits, request.batch_size, request.iou_threshold, request.pca_components),
        daemon=True,
    )
    thread.start()

    return JobStatusResponse(id=job.id, status=job.status)


@app.get("/api/v1/jobs/{job_id}", response_model=JobStatusResponse)
def get_job(job_id: str) -> JobStatusResponse:
    job = _get_job_or_404(job_id)
    return JobStatusResponse(id=job.id, status=job.status, error=job.error, num_records=len(job.records))


@app.get("/api/v1/jobs/{job_id}/records")
def get_job_records(
    job_id: str,
    split: str | None = None,
    status: str | None = None,
    class_id: int | None = None,
    limit: int = Query(default=200, le=2000),
    offset: int = 0,
):
    job = _get_job_or_404(job_id)
    if job.status != "done":
        raise HTTPException(409, f"Job is not finished yet (status='{job.status}')")

    records = job.records
    if split is not None:
        records = [r for r in records if r.split.lower() == split.lower()]
    if status is not None:
        records = [r for r in records if r.status == status]
    if class_id is not None:
        records = [
            r
            for r in records
            if (r.prediction is not None and r.prediction.class_id == class_id)
            or (r.ground_truth is not None and r.ground_truth.class_id == class_id)
        ]

    return records[offset : offset + limit]


@app.get("/api/v1/jobs/{job_id}/images/{record_id}")
def get_record_image(job_id: str, record_id: str) -> FileResponse:
    job = _get_job_or_404(job_id)
    record = next((r for r in job.records if r.id == record_id), None)
    if record is None:
        raise HTTPException(404, f"Record not found: {record_id}")

    image_path = Path(record.image_path)
    if not image_path.exists():
        raise HTTPException(404, f"Image file not found on disk: {image_path}")

    return FileResponse(image_path)


def _get_job_or_404(job_id: str) -> Job:
    job = JOB_STORE.get(job_id)
    if job is None:
        raise HTTPException(404, f"Job not found: {job_id}")
    return job


def _parse_split(name: str) -> Split:
    try:
        return Split[name.strip().upper()]
    except KeyError:
        raise HTTPException(400, f"Unknown split '{name}'. Expected one of: {[s.name for s in Split]}") from None
