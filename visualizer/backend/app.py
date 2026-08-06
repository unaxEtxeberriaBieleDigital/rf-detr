# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

"""FastAPI backend for the RF-DETR data-cleaning / model-evaluation visualizer.

Run with:
    uv run uvicorn visualizer.backend.app:app --reload --port 8000

Typical flow for the frontend:
    1. GET  /api/v1/model-types, /api/v1/dataset-types     -> discover what's available
    2. GET  /api/v1/check-dataset?path=…                   -> check for existing DB
    3a. POST /api/v1/jobs                                   -> kick off inference + evaluation
    3b. POST /api/v1/jobs/load                              -> load an existing DB (skip inference)
    4. GET  /api/v1/jobs/{job_id}                           -> poll until status == "done"
    5. GET  /api/v1/jobs/{job_id}/records?…                 -> fetch (filtered) embedding records
    6. GET  /api/v1/jobs/{job_id}/images/{record_id}        -> fetch the underlying image
    7. POST /api/v1/jobs/{job_id}/pca?components=2          -> compute PCA on demand
"""

import json
import threading
import uuid
from pathlib import Path

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel as PydanticModel

from rfdetr.utilities.logger import get_logger

from visualizer.backend.datasets import cocodetectiondataset  # noqa: F401
from visualizer.backend.datasets.basedataset import Split
from visualizer.backend.jobs import JOB_STORE, Job, run_job
from visualizer.backend.models import rfdetr  # noqa: F401
from visualizer.backend.registry import DATASET_REGISTRY, MODEL_REGISTRY
from visualizer.backend.store import DB_FILENAME, JobStore

logger = get_logger()

app = FastAPI(title="RF-DETR Visualizer API", version="2.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------


class JobRequest(PydanticModel):
    dataset_path: str
    dataset_type: str
    model_path: str
    model_type: str
    splits: list[str] | None = None
    batch_size: int = 8
    iou_threshold: float = 0.5


class LoadJobRequest(PydanticModel):
    dataset_path: str


class JobStatusResponse(PydanticModel):
    id: str
    status: str
    error: str | None = None
    num_records: int = 0
    categories: dict[int, str] = {}
    num_images_total: int = 0
    num_images_processed: int = 0
    has_pca: bool = False
    pca_components: int | None = None


class ImagePathPageResponse(PydanticModel):
    image_paths: list[str]
    total_images: int
    offset: int
    limit: int
    has_more: bool


class RecordsByImagePathsRequest(PydanticModel):
    image_paths: list[str]
    split: str | None = None


class CheckDatasetResponse(PydanticModel):
    has_db: bool
    num_records: int = 0
    has_pca: bool = False
    pca_components: int | None = None
    status: str | None = None


class PcaStatusResponse(PydanticModel):
    updated: int
    components: int


# ---------------------------------------------------------------------------
# Startup: re-register persisted jobs
# ---------------------------------------------------------------------------


@app.on_event("startup")
def _reload_persisted_jobs() -> None:  # noqa: ANN202
    """Scan the known dataset paths for existing DB files and re-register done jobs.

    This makes jobs survive backend restarts.  Any DB whose stored status is
    ``running`` or ``pending`` (i.e. the process was killed mid-run) is marked
    ``error`` so the frontend doesn't wait forever.
    """
    # We only know about DBs that are explicitly listed – there's no global registry
    # of dataset paths.  Instead, we expose a /jobs/load endpoint that the frontend
    # calls when the user provides a dataset path with an existing DB.
    pass


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


# ---------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------


@app.get("/api/v1/model-types")
def list_model_types() -> list[str]:
    return sorted(MODEL_REGISTRY)


@app.get("/api/v1/dataset-types")
def list_dataset_types() -> list[str]:
    return sorted(DATASET_REGISTRY)


# ---------------------------------------------------------------------------
# Dataset check
# ---------------------------------------------------------------------------


@app.get("/api/v1/check-dataset", response_model=CheckDatasetResponse)
def check_dataset(path: str) -> CheckDatasetResponse:
    """Check whether *path* already has a computed embeddings DB.

    Args:
        path: Absolute path to the dataset root directory.

    Returns:
        :class:`CheckDatasetResponse` with ``has_db=True`` and summary if found.
    """
    if not JobStore.db_exists(path):
        return CheckDatasetResponse(has_db=False)

    store = JobStore(path)
    status = store.get_meta("status")
    return CheckDatasetResponse(
        has_db=True,
        num_records=store.record_count(),
        has_pca=store.has_pca(),
        pca_components=store.pca_components(),
        status=status,
    )


# ---------------------------------------------------------------------------
# Job creation (inference)
# ---------------------------------------------------------------------------


@app.post("/api/v1/jobs", response_model=JobStatusResponse, status_code=202)
def create_job(request: JobRequest) -> JobStatusResponse:
    """Create and start a new inference job.

    Inference results are written to ``rfdetr_visualizer.db`` in the dataset
    root, replacing any existing DB at that path.

    Args:
        request: Job creation parameters.

    Returns:
        Initial :class:`JobStatusResponse` with ``status="pending"``.
    """
    if request.dataset_type not in DATASET_REGISTRY:
        raise HTTPException(
            400,
            f"Unknown dataset_type '{request.dataset_type}'. "
            f"Available: {sorted(DATASET_REGISTRY)}",
        )
    if request.model_type not in MODEL_REGISTRY:
        raise HTTPException(
            400,
            f"Unknown model_type '{request.model_type}'. "
            f"Available: {sorted(MODEL_REGISTRY)}",
        )

    dataset_cls = DATASET_REGISTRY[request.dataset_type]
    model_cls = MODEL_REGISTRY[request.model_type]

    try:
        dataset = dataset_cls(request.dataset_path)
        model = model_cls(request.model_path)
        splits = (
            [_parse_split(name) for name in request.splits]
            if request.splits
            else list(dataset.splits.keys())
        )
    except Exception as e:
        logger.error(
            f"Could not initialise dataset/model for a new job: {e}", exc_info=True
        )
        raise HTTPException(400, f"Could not initialise dataset/model: {e}") from e

    store = JobStore(request.dataset_path)
    store.create_tables()
    store.set_meta("dataset_path", request.dataset_path)
    store.set_meta("categories", json.dumps(dataset.categories))

    job = Job(id=str(uuid.uuid4()), store=store)
    job.categories = dataset.categories
    JOB_STORE[job.id] = job

    logger.info(
        f"Created job {job.id}: dataset_type='{request.dataset_type}' "
        f"({request.dataset_path}), model_type='{request.model_type}' "
        f"({request.model_path})"
    )

    thread = threading.Thread(
        target=run_job,
        args=(job, dataset, model, splits, request.batch_size, request.iou_threshold),
        daemon=True,
    )
    thread.start()

    return JobStatusResponse(id=job.id, status=job.status)


# ---------------------------------------------------------------------------
# Load existing job (skip inference)
# ---------------------------------------------------------------------------


@app.post("/api/v1/jobs/load", response_model=JobStatusResponse)
def load_job(request: LoadJobRequest) -> JobStatusResponse:
    """Load an existing DB from *dataset_path* without re-running inference.

    Args:
        request: Contains the ``dataset_path`` pointing to a directory with an
            existing ``rfdetr_visualizer.db``.

    Returns:
        :class:`JobStatusResponse` reflecting the stored job state.
    """
    if not JobStore.db_exists(request.dataset_path):
        raise HTTPException(
            404,
            f"No existing DB found at '{request.dataset_path}'. "
            "Run inference first via POST /api/v1/jobs.",
        )

    store = JobStore(request.dataset_path)
    status = store.get_meta("status") or "done"

    # If a crash left the DB in a running/pending state, surface it as error.
    if status in ("running", "pending"):
        status = "error"
        store.set_meta("status", status)
        store.set_meta("error", "Job was interrupted (backend restarted mid-run)")

    raw_categories = store.get_meta("categories")
    categories: dict[int, str] = {}
    if raw_categories:
        raw = json.loads(raw_categories) if isinstance(raw_categories, str) else raw_categories
        categories = {int(k): v for k, v in raw.items()}

    job_id = str(uuid.uuid4())
    job = Job(id=job_id, store=store, status=status, categories=categories)  # type: ignore[arg-type]
    job.num_images_total = store.get_meta("num_images_total") or 0
    job.num_images_processed = store.get_meta("num_images_processed") or 0
    JOB_STORE[job_id] = job

    logger.info(
        f"Loaded existing job {job_id} from '{request.dataset_path}' "
        f"({store.record_count()} records)"
    )

    return JobStatusResponse(
        id=job_id,
        status=status,
        num_records=store.record_count(),
        categories=categories,
        num_images_total=job.num_images_total,
        num_images_processed=job.num_images_processed,
        has_pca=store.has_pca(),
        pca_components=store.pca_components(),
    )


# ---------------------------------------------------------------------------
# Job status
# ---------------------------------------------------------------------------


@app.get("/api/v1/jobs/{job_id}", response_model=JobStatusResponse)
def get_job(job_id: str) -> JobStatusResponse:
    job = _get_job_or_404(job_id)
    return JobStatusResponse(
        id=job.id,
        status=job.status,
        error=job.error,
        num_records=job.store.record_count(),
        categories=job.categories,
        num_images_total=job.num_images_total,
        num_images_processed=job.num_images_processed,
        has_pca=job.store.has_pca(),
        pca_components=job.store.pca_components(),
    )


# ---------------------------------------------------------------------------
# PCA (on-demand)
# ---------------------------------------------------------------------------


class ReductionRequest(PydanticModel):
    record_ids: list[str] | None = None
    algorithm: str = "pca"  # "pca" | "tsne" | "umap"
    perplexity: float = 30.0
    n_neighbors: int = 15
    min_dist: float = 0.1


@app.post("/api/v1/jobs/{job_id}/pca", response_model=PcaStatusResponse)
def compute_reduction(
    job_id: str,
    components: int = Query(default=2, ge=2, le=3),
    body: ReductionRequest = ReductionRequest(),
) -> PcaStatusResponse:
    """Compute dimensionality reduction over a set of raw embeddings for this job.

    Supports PCA (incremental, memory-safe), t-SNE (batch, O(n log n)), and
    UMAP (batch, faster than t-SNE for large datasets; requires ``umap-learn``).

    When *record_ids* is provided in the request body, the reduction is fitted
    and applied only to that subset.

    Args:
        job_id: Job identifier.
        components: Number of output dimensions (2 or 3).
        body: Request body with algorithm, optional record_ids, and hyperparams.

    Returns:
        :class:`PcaStatusResponse` with the number of updated records.
    """
    job = _get_job_or_404(job_id)
    if job.status != "done":
        raise HTTPException(409, f"Job is not finished yet (status='{job.status}')")
    try:
        updated = job.store.compute_reduction(
            components,
            algorithm=body.algorithm,
            record_ids=body.record_ids,
            perplexity=body.perplexity,
            n_neighbors=body.n_neighbors,
            min_dist=body.min_dist,
        )
    except (ValueError, RuntimeError) as exc:
        raise HTTPException(422, str(exc)) from exc
    return PcaStatusResponse(updated=updated, components=components)


# ---------------------------------------------------------------------------
# Records
# ---------------------------------------------------------------------------


@app.get("/api/v1/jobs/{job_id}/records")
def get_job_records(
    job_id: str,
    split: str | None = None,
    status: str | None = None,
    class_id: int | None = None,
    limit: int = Query(default=200, le=2000),
    offset: int = 0,
) -> list[dict]:
    job = _get_job_or_404(job_id)
    if job.status != "done":
        raise HTTPException(409, f"Job is not finished yet (status='{job.status}')")
    return job.store.get_records(
        split=split, status=status, class_id=class_id, limit=limit, offset=offset
    )


@app.get("/api/v1/jobs/{job_id}/image-paths", response_model=ImagePathPageResponse)
def get_job_image_paths(
    job_id: str,
    split: str | None = None,
    limit: int = Query(default=60, ge=1, le=500),
    offset: int = Query(default=0, ge=0),
) -> ImagePathPageResponse:
    job = _get_job_or_404(job_id)
    if job.status != "done":
        raise HTTPException(409, f"Job is not finished yet (status='{job.status}')")
    paths, total = job.store.get_image_paths(split=split, limit=limit, offset=offset)
    has_more = offset + len(paths) < total
    return ImagePathPageResponse(
        image_paths=paths,
        total_images=total,
        offset=offset,
        limit=limit,
        has_more=has_more,
    )


@app.post("/api/v1/jobs/{job_id}/records/by-image-paths")
def get_job_records_by_image_paths(
    job_id: str, payload: RecordsByImagePathsRequest
) -> list[dict]:
    job = _get_job_or_404(job_id)
    if job.status != "done":
        raise HTTPException(409, f"Job is not finished yet (status='{job.status}')")
    return job.store.get_records_by_image_paths(
        image_paths=payload.image_paths, split=payload.split
    )


# ---------------------------------------------------------------------------
# Image serving
# ---------------------------------------------------------------------------


@app.get("/api/v1/jobs/{job_id}/images/{record_id}")
def get_record_image(job_id: str, record_id: str) -> FileResponse:
    job = _get_job_or_404(job_id)
    image_path_str = job.store.get_image_path_for_record(record_id)
    if image_path_str is None:
        raise HTTPException(404, f"Record not found: {record_id}")
    image_path = Path(image_path_str)
    if not image_path.exists():
        raise HTTPException(404, f"Image file not found on disk: {image_path}")
    return FileResponse(image_path)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _get_job_or_404(job_id: str) -> Job:
    job = JOB_STORE.get(job_id)
    if job is None:
        raise HTTPException(404, f"Job not found: {job_id}")
    return job


def _parse_split(name: str) -> Split:
    try:
        return Split[name.strip().upper()]
    except KeyError:
        raise HTTPException(
            400,
            f"Unknown split '{name}'. Expected one of: {[s.name for s in Split]}",
        ) from None
