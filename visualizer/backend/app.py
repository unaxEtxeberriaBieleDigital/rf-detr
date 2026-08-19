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
    7. POST /api/v1/jobs/{job_id}/dimensionality_reduction?components=2 -> compute reduction on demand
"""

import base64
import json
import threading
import uuid
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel as PydanticModel

from rfdetr.utilities.logger import get_logger
from visualizer.backend.datasets import cocodetectiondataset  # noqa: F401
from visualizer.backend.datasets.basedataset import Split
from visualizer.backend.evaluator import Match
from visualizer.backend.jobs import (
    JOB_STORE,
    Job,
    release_active_job,
    run_job,
    try_register_active_job,
)
from visualizer.backend.metrics import get_metrics_for_dataset
from visualizer.backend.models import rfdetr  # noqa: F401
from visualizer.backend.prediction import Prediction
from visualizer.backend.registry import DATASET_REGISTRY, MODEL_REGISTRY, SEMANTIC_SEARCH_SOURCE_REGISTRY
from visualizer.backend.semantic_search import SEARCH_JOB_STORE, SearchJob, run_semantic_search
from visualizer.backend.semantic_search import sources as semantic_search_sources  # noqa: F401
from visualizer.backend.semantic_search.sources.basesource import BaseSemanticSearchSource
from visualizer.backend.store import JobStore

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
    resume: bool = False


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
    num_images_remaining: int = 0
    can_resume: bool = False
    has_dimensionality_reduction: bool = False
    dimensionality_reduction_components: int | None = None


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
    has_dimensionality_reduction: bool = False
    dimensionality_reduction_components: int | None = None
    status: str | None = None
    num_images_total: int = 0
    num_images_processed: int = 0
    num_images_remaining: int = 0
    can_resume: bool = False


class DimensionalityReductionStatusResponse(PydanticModel):
    updated: int
    components: int


class MetricDefinitionResponse(PydanticModel):
    name: str
    display_name: str
    description: str
    metric_type: str


class EvaluationMetricsResponse(PydanticModel):
    dataset_type: str
    metrics: dict[str, Any]
    metric_definitions: list[MetricDefinitionResponse]
    cached: bool
    calculated_at: str | None = None
    applied_class_thresholds: dict[int, float] | None = None
    applied_record_ids: list[str] | None = None
    applied_record_count: int | None = None


class EvaluationRequest(PydanticModel):
    class_thresholds: dict[int, float] | None = None
    record_ids: list[str] | None = None


class OptimalThresholdResponse(PydanticModel):
    dataset_type: str
    metric_name: str
    class_id: int | None = None
    threshold: float
    metric_value: float
    num_thresholds: int


class SemanticSearchRequest(PydanticModel):
    query_record_id: str
    search_path: str
    k: int = 20
    model_path: str
    model_type: str
    source_type: str = "default"


class SemanticSearchResultDTO(PydanticModel):
    image_path: str
    bbox: tuple[float, float, float, float] | None
    confidence: float
    class_id: int
    distance: float
    preview_data_url: str


class SemanticSearchStatusResponse(PydanticModel):
    id: str
    parent_job_id: str
    query_record_id: str
    query_image_path: str
    search_path: str
    k: int
    status: str
    error: str | None = None
    num_images_total: int = 0
    num_images_processed: int = 0
    results: list[SemanticSearchResultDTO] | None = None


# ---------------------------------------------------------------------------
# Startup: re-register persisted jobs
# ---------------------------------------------------------------------------


@app.on_event("startup")
def _reload_persisted_jobs() -> None:
    """Scan the known dataset paths for existing DB files and re-register done jobs.

    This makes jobs survive backend restarts.  Any DB whose stored status is ``running`` or ``pending`` (i.e. the
    process was killed mid-run) is marked ``error`` so the frontend doesn't wait forever.
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
    components = store.dimensionality_reduction_components()
    has_reduction = store.has_dimensionality_reduction()
    num_images_total = int(store.get_meta("num_images_total") or 0)
    num_images_processed = int(store.get_meta("num_images_processed") or 0)
    can_resume = _store_can_resume(store, status)
    return CheckDatasetResponse(
        has_db=True,
        num_records=store.record_count(),
        has_dimensionality_reduction=has_reduction,
        dimensionality_reduction_components=components,
        status=status,
        num_images_total=num_images_total,
        num_images_processed=num_images_processed,
        num_images_remaining=_num_images_remaining(num_images_total, num_images_processed),
        can_resume=can_resume,
    )


# ---------------------------------------------------------------------------
# Job creation (inference)
# ---------------------------------------------------------------------------


@app.post("/api/v1/jobs", response_model=JobStatusResponse, status_code=202)
def create_job(request: JobRequest) -> JobStatusResponse:
    """Create and start a new inference job.

    Inference results are written to ``rfdetr_visualizer.db`` in the dataset
    root. Fresh runs overwrite prior inference artifacts; ``resume=True``
    continues an interrupted compatible run in the existing DB.

    Args:
        request: Job creation parameters.

    Returns:
        Initial :class:`JobStatusResponse` with ``status="pending"``.
    """
    if request.dataset_type not in DATASET_REGISTRY:
        raise HTTPException(
            400,
            f"Unknown dataset_type '{request.dataset_type}'. Available: {sorted(DATASET_REGISTRY)}",
        )
    if request.model_type not in MODEL_REGISTRY:
        raise HTTPException(
            400,
            f"Unknown model_type '{request.model_type}'. Available: {sorted(MODEL_REGISTRY)}",
        )

    dataset_cls = DATASET_REGISTRY[request.dataset_type]
    model_cls = MODEL_REGISTRY[request.model_type]

    try:
        dataset = dataset_cls(request.dataset_path)
        splits = [_parse_split(name) for name in request.splits] if request.splits else list(dataset.splits.keys())
    except Exception as e:
        logger.error(f"Could not initialise dataset for a new job: {e}", exc_info=True)
        raise HTTPException(400, f"Could not initialise dataset: {e}") from e

    db_already_exists = JobStore.db_exists(request.dataset_path)
    if request.resume and not db_already_exists:
        raise HTTPException(
            409,
            "Resume was requested, but no existing visualizer DB was found for this dataset.",
        )
    store = JobStore(request.dataset_path)
    store.create_tables()
    split_names = [split.name for split in splits]
    requested_run_config = _build_run_config(request, split_names)
    job_id = str(uuid.uuid4())

    existing_active_job_id = try_register_active_job(request.dataset_path, job_id)
    if existing_active_job_id is not None:
        active_job = JOB_STORE.get(existing_active_job_id)
        if active_job is None:
            release_active_job(request.dataset_path, existing_active_job_id)
            existing_active_job_id = try_register_active_job(request.dataset_path, job_id)
            active_job = JOB_STORE.get(existing_active_job_id) if existing_active_job_id else None
        if existing_active_job_id is not None:
            if request.resume and store.get_run_config() == requested_run_config and active_job is not None:
                return _job_to_response(active_job)
            raise HTTPException(
                409,
                f"An inference job is already running for this dataset (job_id={existing_active_job_id}).",
            )

    try:
        if request.resume:
            if not store.has_progress_tracking():
                raise HTTPException(
                    409,
                    "Resume was requested, but this DB does not support persisted progress. Start a fresh run instead.",
                )

            stored_status = store.get_meta("status")
            if stored_status == "done":
                raise HTTPException(
                    409,
                    "Resume was requested, but the stored run is already complete. "
                    "Load it or start a fresh run instead.",
                )
            if stored_status not in {"pending", "running", "error"}:
                raise HTTPException(
                    409,
                    "Resume was requested, but no interrupted run state was found in this DB.",
                )

            stored_run_config = store.get_run_config()
            if stored_run_config != requested_run_config:
                raise HTTPException(
                    409,
                    "Resume configuration does not match the interrupted run. "
                    "Use the same dataset_type, model_path, model_type, splits, batch_size, and iou_threshold.",
                )
            num_images_total = int(store.get_meta("num_images_total") or 0)
            num_images_processed = store.processed_image_count(split_names=split_names)
        else:
            store.reset_for_rerun()
            num_images_total = 0
            num_images_processed = 0

        store.set_meta("dataset_path", request.dataset_path)
        store.set_meta("dataset_type", request.dataset_type)
        store.set_meta("model_path", request.model_path)
        store.set_meta("model_type", request.model_type)
        store.set_meta("categories", dataset.categories)
        store.set_run_config(requested_run_config)
        store.enable_progress_tracking()
        store.set_meta("status", "pending")
        store.set_meta("error", None)
        store.set_meta("num_images_total", num_images_total)
        store.set_meta("num_images_processed", num_images_processed)

        try:
            model = model_cls(request.model_path)
        except Exception as e:
            logger.error(f"Could not initialise model for a new job: {e}", exc_info=True)
            raise HTTPException(400, f"Could not initialise model: {e}") from e

        job = Job(id=job_id, store=store)
        job.categories = dataset.categories
        job.num_images_total = num_images_total
        job.num_images_processed = num_images_processed
        JOB_STORE[job.id] = job

        logger.info(
            f"Created job {job.id}: dataset_type='{request.dataset_type}' "
            f"({request.dataset_path}), model_type='{request.model_type}' "
            f"({request.model_path}), resume={request.resume}"
        )

        thread = threading.Thread(
            target=run_job,
            args=(
                job,
                dataset,
                model,
                splits,
                request.batch_size,
                request.iou_threshold,
                request.resume,
            ),
            daemon=True,
        )
        thread.start()
    except Exception:
        JOB_STORE.pop(job_id, None)
        release_active_job(request.dataset_path, job_id)
        raise

    return _job_to_response(job)


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
            f"No existing DB found at '{request.dataset_path}'. Run inference first via POST /api/v1/jobs.",
        )

    store = JobStore(request.dataset_path)
    status = store.get_meta("status") or "done"
    can_resume = _store_can_resume(store, status)
    error = store.get_meta("error")

    # If a crash left the DB in a running/pending state, surface it as error.
    if status in ("running", "pending"):
        status = "error"
        error = "Job was interrupted (backend restarted mid-run)"
        store.set_meta("status", status)
        store.set_meta("error", error)

    raw_categories = store.get_meta("categories")
    categories: dict[int, str] = {}
    if raw_categories:
        raw = json.loads(raw_categories) if isinstance(raw_categories, str) else raw_categories
        categories = {int(k): v for k, v in raw.items()}

    job_id = str(uuid.uuid4())
    job = Job(
        id=job_id,
        store=store,
        status=status,  # type: ignore[arg-type]
        categories=categories,
        error=error if isinstance(error, str) else None,
    )
    job.num_images_total = store.get_meta("num_images_total") or 0
    job.num_images_processed = store.get_meta("num_images_processed") or 0
    JOB_STORE[job_id] = job

    logger.info(f"Loaded existing job {job_id} from '{request.dataset_path}' ({store.record_count()} records)")
    if status == "error":
        job.error = error if isinstance(error, str) else "Job was interrupted"

    response = _job_to_response(job)
    response.can_resume = can_resume
    return response


# ---------------------------------------------------------------------------
# Job status
# ---------------------------------------------------------------------------


@app.get("/api/v1/jobs/{job_id}", response_model=JobStatusResponse)
def get_job(job_id: str) -> JobStatusResponse:
    job = _get_job_or_404(job_id)
    return _job_to_response(job)


# ---------------------------------------------------------------------------
# Dimensionality reduction (on-demand)
# ---------------------------------------------------------------------------


class ReductionRequest(PydanticModel):
    record_ids: list[str] | None = None
    algorithm: str = "pca"  # "pca" | "tsne" | "umap"
    perplexity: float = 30.0
    n_neighbors: int = 15
    min_dist: float = 0.1


@app.post(
    "/api/v1/jobs/{job_id}/dimensionality_reduction",
    response_model=DimensionalityReductionStatusResponse,
)
def compute_dimensionality_reduction(
    job_id: str,
    components: int = Query(default=2, ge=2, le=3),
    body: ReductionRequest = ReductionRequest(),
) -> DimensionalityReductionStatusResponse:
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
        :class:`DimensionalityReductionStatusResponse` with the number of updated records.
    """
    return _compute_dimensionality_reduction(job_id=job_id, components=components, body=body)


def _compute_dimensionality_reduction(
    job_id: str,
    components: int,
    body: ReductionRequest,
) -> DimensionalityReductionStatusResponse:
    """Shared implementation for dimensionality-reduction endpoints."""
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
    return DimensionalityReductionStatusResponse(updated=updated, components=components)


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
    return job.store.get_records(split=split, status=status, class_id=class_id, limit=limit, offset=offset)


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
def get_job_records_by_image_paths(job_id: str, payload: RecordsByImagePathsRequest) -> list[dict]:
    job = _get_job_or_404(job_id)
    if job.status != "done":
        raise HTTPException(409, f"Job is not finished yet (status='{job.status}')")
    return job.store.get_records_by_image_paths(image_paths=payload.image_paths, split=payload.split)


# ---------------------------------------------------------------------------
# Evaluation metrics (on-demand + cached)
# ---------------------------------------------------------------------------


def _calculate_job_evaluation(
    job_id: str,
    class_thresholds: str | None = Query(default=None),
    record_ids: str | None = Query(default=None),
) -> EvaluationMetricsResponse:
    """Compute or return cached evaluation metrics for this job."""
    try:
        job = _get_job_or_404(job_id)
        if job.status != "done":
            raise HTTPException(409, f"Job is not finished yet (status='{job.status}')")

        dataset_type = _get_dataset_type_for_job(job)
        metric_defs = get_metrics_for_dataset(dataset_type)
        if not metric_defs:
            raise HTTPException(422, f"Dataset type '{dataset_type}' has no registered metrics.")
        calculator = _get_metrics_calculator_for_job(job, dataset_type)
        normalized_class_thresholds = _parse_class_thresholds_query(
            class_thresholds,
            allowed_class_ids=set(job.categories),
        )
        normalized_record_ids = _parse_record_ids_query(record_ids)

        metric_definition_responses = [
            MetricDefinitionResponse(
                name=m.name,
                display_name=m.display_name,
                description=m.description,
                metric_type=m.metric_type.value,
            )
            for m in metric_defs
        ]

        if normalized_class_thresholds is None and normalized_record_ids is None:
            cached = job.store.get_cached_metrics(job.id, dataset_type)
            if cached is not None:
                metrics, calculated_at = cached
                return EvaluationMetricsResponse(
                    dataset_type=dataset_type,
                    metrics=metrics,
                    metric_definitions=metric_definition_responses,
                    cached=True,
                    calculated_at=calculated_at,
                )

        matches, applied_record_ids = _load_matches_for_job(job, record_ids=normalized_record_ids)
        metrics = calculator.calculate(matches, class_thresholds=normalized_class_thresholds)

        calculated_at = None
        if normalized_class_thresholds is None and normalized_record_ids is None:
            job.store.cache_metrics(job.id, dataset_type, metrics)
            cached_after_write = job.store.get_cached_metrics(job.id, dataset_type)
            calculated_at = cached_after_write[1] if cached_after_write is not None else None

        return EvaluationMetricsResponse(
            dataset_type=dataset_type,
            metrics=metrics,
            metric_definitions=metric_definition_responses,
            cached=False,
            calculated_at=calculated_at,
            applied_class_thresholds=normalized_class_thresholds,
            applied_record_ids=applied_record_ids,
            applied_record_count=(len(applied_record_ids) if applied_record_ids is not None else None),
        )
    except HTTPException:
        raise
    except Exception as exc:
        logger.error("Evaluation endpoint failed", exc_info=True)
        raise HTTPException(500, f"Evaluation failed: {exc}") from exc


@app.get("/api/v1/jobs/{job_id}/evaluation", response_model=EvaluationMetricsResponse)
def get_job_evaluation(
    job_id: str,
    class_thresholds: str | None = Query(default=None),
    record_ids: str | None = Query(default=None),
) -> EvaluationMetricsResponse:
    """Compute evaluation metrics from optional URL-encoded filters."""
    return _calculate_job_evaluation(job_id, class_thresholds, record_ids)


@app.post("/api/v1/jobs/{job_id}/evaluation", response_model=EvaluationMetricsResponse)
def post_job_evaluation(
    job_id: str,
    request: EvaluationRequest,
) -> EvaluationMetricsResponse:
    """Compute evaluation metrics from filters sent in a JSON request body."""
    class_thresholds = (
        json.dumps(request.class_thresholds) if request.class_thresholds is not None else None
    )
    record_ids = json.dumps(request.record_ids) if request.record_ids is not None else None
    return _calculate_job_evaluation(job_id, class_thresholds, record_ids)


@app.get("/api/v1/jobs/{job_id}/optimal-threshold", response_model=OptimalThresholdResponse)
def get_optimal_threshold(
    job_id: str,
    metric: str = Query(default="f1"),
    num_thresholds: int = Query(default=100, ge=10, le=1000),
    class_id: int | None = Query(default=None),
) -> OptimalThresholdResponse:
    """Find confidence threshold that maximizes one optimizable metric."""
    try:
        job = _get_job_or_404(job_id)
        if job.status != "done":
            raise HTTPException(409, f"Job is not finished yet (status='{job.status}')")

        dataset_type = _get_dataset_type_for_job(job)
        calculator = _get_metrics_calculator_for_job(job, dataset_type)
        if class_id is not None and class_id not in job.categories:
            raise HTTPException(422, f"Unknown class_id '{class_id}' for this job.")
        matches, _ = _load_matches_for_job(job)
        try:
            threshold = calculator.get_optimal_threshold(
                metric_name=metric,
                matches=matches,
                num_thresholds=num_thresholds,
                class_id=class_id,
            )
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from exc

        threshold_scope_matches = _filter_matches_for_threshold_optimization(matches, class_id)
        scoped_thresholds = _build_uniform_class_thresholds(threshold_scope_matches, threshold, class_id)
        metric_value_raw = calculator.calculate(
            threshold_scope_matches,
            class_thresholds=scoped_thresholds,
        ).get(metric, 0.0)
        if isinstance(metric_value_raw, list) or isinstance(metric_value_raw, dict):
            metric_value = 0.0
        else:
            metric_value = float(metric_value_raw)

        return OptimalThresholdResponse(
            dataset_type=dataset_type,
            metric_name=metric,
            class_id=class_id,
            threshold=float(threshold),
            metric_value=metric_value,
            num_thresholds=num_thresholds,
        )
    except HTTPException:
        raise
    except Exception as exc:
        logger.error("Optimal-threshold endpoint failed", exc_info=True)
        raise HTTPException(500, f"Optimal threshold failed: {exc}") from exc


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
# Semantic search (nearest-neighbour over an arbitrary folder)
# ---------------------------------------------------------------------------


@app.post(
    "/api/v1/jobs/{job_id}/semantic-search",
    response_model=SemanticSearchStatusResponse,
    status_code=202,
)
def create_semantic_search(job_id: str, request: SemanticSearchRequest) -> SemanticSearchStatusResponse:
    """Start a background nearest-neighbour search over ``request.search_path``.

    The query embedding is looked up server-side from ``request.query_record_id`` (always
    the full-dimensionality raw embedding, regardless of whether PCA has been computed),
    so only the record id needs to travel over the wire.

    Args:
        job_id: The visualizer job that owns the query record.
        request: Search parameters (query record, target folder, k, model to run).

    Returns:
        Initial :class:`SemanticSearchStatusResponse` with ``status="pending"``.
    """
    job = _get_job_or_404(job_id)
    if request.model_type not in MODEL_REGISTRY:
        raise HTTPException(
            400,
            f"Unknown model_type '{request.model_type}'. Available: {sorted(MODEL_REGISTRY)}",
        )
    if request.source_type not in SEMANTIC_SEARCH_SOURCE_REGISTRY:
        raise HTTPException(
            400,
            f"Unknown source_type '{request.source_type}'. Available: {sorted(SEMANTIC_SEARCH_SOURCE_REGISTRY)}",
        )

    query_embedding = job.store.get_raw_embedding(request.query_record_id)
    if query_embedding is None:
        raise HTTPException(404, f"No raw embedding found for record: {request.query_record_id}")

    search_folder = Path(request.search_path)
    if not search_folder.exists() or not search_folder.is_dir():
        raise HTTPException(400, f"Search folder not found: {request.search_path}")

    model_cls = MODEL_REGISTRY[request.model_type]
    try:
        model = model_cls(request.model_path)
    except Exception as e:
        logger.error(f"Could not initialise model for semantic search: {e}", exc_info=True)
        raise HTTPException(400, f"Could not initialise model: {e}") from e

    query_image_path = job.store.get_image_path_for_record(request.query_record_id) or ""
    search_job = SearchJob(
        id=str(uuid.uuid4()),
        parent_job_id=job_id,
        query_record_id=request.query_record_id,
        query_image_path=query_image_path,
        search_path=request.search_path,
        k=request.k,
        source_type=request.source_type,
    )
    SEARCH_JOB_STORE[search_job.id] = search_job

    logger.info(
        f"Created semantic search {search_job.id} for job {job_id}: "
        f"query_record_id='{request.query_record_id}', search_path='{request.search_path}', "
        f"k={request.k}, source_type='{request.source_type}'"
    )

    source_cls = SEMANTIC_SEARCH_SOURCE_REGISTRY[request.source_type]
    source = source_cls()

    thread = threading.Thread(
        target=run_semantic_search,
        args=(search_job, model, request.model_type, query_embedding, source),
        daemon=True,
    )
    thread.start()

    return _search_job_to_response(search_job)


@app.get(
    "/api/v1/jobs/{job_id}/semantic-search",
    response_model=list[SemanticSearchStatusResponse],
)
def list_semantic_searches(job_id: str) -> list[SemanticSearchStatusResponse]:
    """List all semantic-search jobs started for *job_id* (most recent last).

    Lets the frontend reattach to running/finished searches (e.g. after closing and reopening the semantic-search panel,
    or after a page refresh) without losing track of progress, as long as the backend process is still alive.
    """
    _get_job_or_404(job_id)
    matching = [j for j in SEARCH_JOB_STORE.values() if j.parent_job_id == job_id]
    return [_search_job_to_response(j, include_results=False) for j in matching]


@app.get(
    "/api/v1/jobs/{job_id}/semantic-search/{search_id}",
    response_model=SemanticSearchStatusResponse,
)
def get_semantic_search(job_id: str, search_id: str) -> SemanticSearchStatusResponse:
    _get_job_or_404(job_id)
    search_job = _get_search_job_or_404(search_id)
    return _search_job_to_response(search_job, include_results=True)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _num_images_remaining(num_images_total: int, num_images_processed: int) -> int:
    """Return the non-negative number of images left to process."""
    return max(num_images_total - num_images_processed, 0)


def _store_can_resume(store: JobStore, status: str | None) -> bool:
    """Return True when *store* represents an interrupted resumable run."""
    return bool(status in {"pending", "running", "error"} and store.has_progress_tracking())


def _build_run_config(request: JobRequest, split_names: list[str]) -> dict[str, Any]:
    """Return the normalized persisted run configuration for one request."""
    return {
        "dataset_type": request.dataset_type,
        "model_path": request.model_path,
        "model_type": request.model_type,
        "splits": split_names,
        "batch_size": int(request.batch_size),
        "iou_threshold": float(request.iou_threshold),
    }


def _job_to_response(job: Job) -> JobStatusResponse:
    """Convert an in-memory job to the frontend-facing status response."""
    components = job.store.dimensionality_reduction_components()
    has_reduction = job.store.has_dimensionality_reduction()
    num_images_total = int(job.num_images_total or job.store.get_meta("num_images_total") or 0)
    num_images_processed = int(job.num_images_processed or job.store.get_meta("num_images_processed") or 0)
    error = job.error
    if error is None and job.status == "error":
        stored_error = job.store.get_meta("error")
        error = str(stored_error) if stored_error is not None else None

    return JobStatusResponse(
        id=job.id,
        status=job.status,
        error=error,
        num_records=job.store.record_count(),
        categories=job.categories,
        num_images_total=num_images_total,
        num_images_processed=num_images_processed,
        num_images_remaining=_num_images_remaining(num_images_total, num_images_processed),
        can_resume=_store_can_resume(job.store, job.status),
        has_dimensionality_reduction=has_reduction,
        dimensionality_reduction_components=components,
    )


def _get_job_or_404(job_id: str) -> Job:
    job = JOB_STORE.get(job_id)
    if job is None:
        raise HTTPException(404, f"Job not found: {job_id}")
    return job


def _get_dataset_type_for_job(job: Job) -> str:
    dataset_type = job.store.get_meta("dataset_type")
    if isinstance(dataset_type, str) and dataset_type in DATASET_REGISTRY:
        return dataset_type

    # Backward-compatible fallback for DBs created before dataset_type metadata existed.
    if len(DATASET_REGISTRY) == 1:
        return next(iter(DATASET_REGISTRY))
    if "coco_detection" in DATASET_REGISTRY:
        return "coco_detection"
    raise HTTPException(
        422,
        "Could not determine dataset_type for this job. Re-run inference with the latest backend.",
    )


def _get_metrics_calculator_for_job(job: Job, dataset_type: str):
    # Build the calculator from already-loaded job metadata so this endpoint
    # works with existing DBs without forcing dataset re-instantiation.
    if dataset_type == "coco_detection":
        if not job.categories:
            raw_categories = job.store.get_meta("categories")
            categories: dict[int, str] = {}
            if raw_categories:
                raw = json.loads(raw_categories) if isinstance(raw_categories, str) else raw_categories
                categories = {int(k): v for k, v in raw.items()}
            if not categories:
                raise HTTPException(
                    422,
                    "No category metadata available for evaluation. Load/create the job again to refresh metadata.",
                )
            job.categories = categories

        from visualizer.backend.metrics.coco_detection_metrics import (  # local import to keep startup light
            COCODetectionMetricsCalculator,
        )

        return COCODetectionMetricsCalculator(job.categories)

    dataset_cls = DATASET_REGISTRY.get(dataset_type)
    if dataset_cls is None:
        raise HTTPException(422, f"Unknown dataset_type '{dataset_type}'.")
    dataset_path = job.store.get_meta("dataset_path")
    if not dataset_path:
        raise HTTPException(422, "Missing dataset_path metadata in job DB.")
    try:
        dataset = dataset_cls(dataset_path)
    except Exception as exc:
        raise HTTPException(
            422,
            f"Could not initialize dataset '{dataset_type}' from stored path: {exc}",
        ) from exc
    if not hasattr(dataset, "get_metrics_calculator"):
        raise HTTPException(422, f"Dataset type '{dataset_type}' does not implement metrics.")
    return dataset.get_metrics_calculator()


def _row_to_match(row: dict[str, Any]) -> Match:
    pred_bbox = None
    if row["pred_x1"] is not None:
        pred_bbox = (
            float(row["pred_x1"]),
            float(row["pred_y1"]),
            float(row["pred_x2"]),
            float(row["pred_y2"]),
        )
    prediction = (
        Prediction(
            class_id=int(row["pred_class_id"]),
            confidence=float(row["pred_confidence"] or 0.0),
            bbox=pred_bbox,
        )
        if row["pred_class_id"] is not None
        else None
    )

    gt_bbox = None
    if row["gt_x1"] is not None:
        gt_bbox = (
            float(row["gt_x1"]),
            float(row["gt_y1"]),
            float(row["gt_x2"]),
            float(row["gt_y2"]),
        )
    ground_truth = (
        Prediction(
            class_id=int(row["gt_class_id"]),
            confidence=float(row["gt_confidence"] or 1.0),
            bbox=gt_bbox,
        )
        if row["gt_class_id"] is not None
        else None
    )
    return Match(
        prediction=prediction,
        embedding=None,
        ground_truth=ground_truth,
        status=str(row["status"]),
    )


def _load_matches_for_job(
    job: Job,
    record_ids: list[str] | None = None,
) -> tuple[list[Match], list[str] | None]:
    """Load evaluation matches for a job, optionally scoped to selected records."""
    rows = job.store.get_evaluation_rows(record_ids=record_ids)
    applied_record_ids = _validate_record_ids_selection(record_ids, rows)
    return [_row_to_match(row) for row in rows], applied_record_ids


def _parse_class_thresholds_query(
    raw_class_thresholds: str | None,
    allowed_class_ids: set[int],
) -> dict[int, float] | None:
    """Parse and validate a JSON-encoded per-class threshold mapping."""
    if raw_class_thresholds is None:
        return None

    try:
        parsed = json.loads(raw_class_thresholds)
    except json.JSONDecodeError as exc:
        raise HTTPException(
            422,
            "Invalid class_thresholds: expected a JSON object mapping class ids to thresholds.",
        ) from exc

    if parsed is None:
        return None
    if not isinstance(parsed, dict):
        raise HTTPException(
            422,
            "Invalid class_thresholds: expected a JSON object mapping class ids to thresholds.",
        )

    normalized: dict[int, float] = {}
    for raw_class_id, raw_threshold in parsed.items():
        try:
            class_id = int(raw_class_id)
        except (TypeError, ValueError) as exc:
            raise HTTPException(
                422,
                "Invalid class_thresholds: all keys must be integer class ids.",
            ) from exc

        if class_id not in allowed_class_ids:
            raise HTTPException(
                422,
                f"Invalid class_thresholds: unknown class_id '{class_id}'.",
            )

        try:
            threshold = float(raw_threshold)
        except (TypeError, ValueError) as exc:
            raise HTTPException(
                422,
                f"Invalid class_thresholds: threshold for class_id '{class_id}' must be numeric.",
            ) from exc

        if not 0.0 <= threshold <= 1.0:
            raise HTTPException(
                422,
                f"Invalid class_thresholds: threshold for class_id '{class_id}' must be between 0 and 1.",
            )

        normalized[class_id] = threshold

    return normalized or None


def _parse_record_ids_query(raw_record_ids: str | None) -> list[str] | None:
    """Parse and normalize a JSON-encoded record-id selection."""
    if raw_record_ids is None:
        return None

    try:
        parsed = json.loads(raw_record_ids)
    except json.JSONDecodeError as exc:
        raise HTTPException(
            422,
            "Invalid record_ids: expected a JSON array of record id strings.",
        ) from exc

    if not isinstance(parsed, list):
        raise HTTPException(
            422,
            "Invalid record_ids: expected a JSON array of record id strings.",
        )

    normalized: list[str] = []
    seen_record_ids: set[str] = set()
    for raw_record_id in parsed:
        if not isinstance(raw_record_id, str):
            raise HTTPException(
                422,
                "Invalid record_ids: every entry must be a string.",
            )
        if raw_record_id not in seen_record_ids:
            normalized.append(raw_record_id)
            seen_record_ids.add(raw_record_id)

    return normalized


def _validate_record_ids_selection(
    requested_record_ids: list[str] | None,
    rows: list[dict[str, Any]],
) -> list[str] | None:
    """Validate that every requested record id exists in the evaluation store."""
    if requested_record_ids is None:
        return None

    matched_record_ids = {str(row["id"]) for row in rows}
    unknown_record_ids = [record_id for record_id in requested_record_ids if record_id not in matched_record_ids]
    if unknown_record_ids:
        raise HTTPException(
            422,
            f"Invalid record_ids: unknown record ids {unknown_record_ids}.",
        )

    return requested_record_ids


def _filter_matches_for_threshold_optimization(
    matches: list[Match],
    class_id: int | None,
) -> list[Match]:
    """Filter predictions and false negatives belonging to one class."""
    if class_id is None:
        return list(matches)
    return [
        match
        for match in matches
        if (match.prediction is not None and match.prediction.class_id == class_id)
        or (match.prediction is None and match.ground_truth is not None and match.ground_truth.class_id == class_id)
    ]


def _build_uniform_class_thresholds(
    matches: list[Match],
    threshold: float,
    class_id: int | None,
) -> dict[int, float]:
    """Build a threshold mapping for either one class or all predicted classes."""
    if class_id is not None:
        return {class_id: float(threshold)}
    return {match.prediction.class_id: float(threshold) for match in matches if match.prediction is not None}


def _get_search_job_or_404(search_id: str) -> SearchJob:
    search_job = SEARCH_JOB_STORE.get(search_id)
    if search_job is None:
        raise HTTPException(404, f"Semantic search not found: {search_id}")
    return search_job


def _get_semantic_search_source(source_type: str) -> BaseSemanticSearchSource:
    source_cls = SEMANTIC_SEARCH_SOURCE_REGISTRY.get(source_type)
    if source_cls is None:
        raise HTTPException(400, f"Unknown source_type '{source_type}'")
    return source_cls()


def _search_job_to_response(search_job: SearchJob, include_results: bool = False) -> SemanticSearchStatusResponse:
    results = None
    if include_results and search_job.status == "done":
        source = _get_semantic_search_source(search_job.source_type)
        results = []
        for r in search_job.results:
            try:
                preview = source.render_result_preview(r)
            except FileNotFoundError as e:
                logger.warning(f"Skipping semantic-search result, could not render preview: {e}")
                continue
            preview_data_url = f"data:{preview.media_type};base64,{base64.b64encode(preview.content).decode('ascii')}"
            results.append(
                SemanticSearchResultDTO(
                    image_path=r.image_path,
                    bbox=preview.bbox,
                    confidence=r.prediction.confidence,
                    class_id=r.prediction.class_id,
                    distance=r.distance,
                    preview_data_url=preview_data_url,
                )
            )
    return SemanticSearchStatusResponse(
        id=search_job.id,
        parent_job_id=search_job.parent_job_id,
        query_record_id=search_job.query_record_id,
        query_image_path=search_job.query_image_path,
        search_path=search_job.search_path,
        k=search_job.k,
        status=search_job.status,
        error=search_job.error,
        num_images_total=search_job.num_images_total,
        num_images_processed=search_job.num_images_processed,
        results=results,
    )


def _parse_split(name: str) -> Split:
    try:
        return Split[name.strip().upper()]
    except KeyError:
        raise HTTPException(
            400,
            f"Unknown split '{name}'. Expected one of: {[s.name for s in Split]}",
        ) from None
