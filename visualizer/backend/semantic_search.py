# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

"""Semantic (nearest-neighbour) search over an arbitrary folder of images.

Given a query embedding (taken from an existing prediction in a visualizer job), this
module runs the model over every image found recursively under a user-chosen folder
(which need not be one of the dataset's splits) and keeps the ``k`` images whose closest
detection is nearest to the query, ranked by cosine distance. Each image contributes at
most one result (its closest-matching detection), so the same image never appears twice
among the neighbours.

Per-image inference results (embeddings + predictions) are cached in a small SQLite
database at the root of the searched folder (see ``visualizer.backend.search_cache``), so
re-running a search against the same folder with the same model only needs to recompute
cosine distances -- inference is skipped entirely for already-scanned images.

Runs in a background thread, mirroring ``visualizer.backend.jobs``: the search keeps
progressing even if no client is polling it, and the frontend can reattach to it (by
``search_id``) at any time to read the current progress or final results.
"""

import heapq
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import numpy as np

from rfdetr.utilities.logger import get_logger
from visualizer.backend.datasets.basedataset import SUPPORTED_IMAGE_EXTENSIONS
from visualizer.backend.models.basemodel import BaseModel
from visualizer.backend.prediction import Prediction
from visualizer.backend.search_cache import SearchCache

logger = get_logger()

SearchStatus = Literal["pending", "running", "done", "error"]

# Number of images sent to the model per inference call.
_BATCH_SIZE = 8


@dataclass
class SearchResult:
    """One nearest-neighbour hit: a single detection in some image under the search folder."""

    image_path: str
    bbox: tuple[float, float, float, float] | None
    confidence: float
    class_id: int
    distance: float


@dataclass
class SearchJob:
    """State for one semantic-search run, kept fully in memory."""

    id: str
    parent_job_id: str
    query_record_id: str
    query_image_path: str
    search_path: str
    k: int
    status: SearchStatus = "pending"
    error: str | None = None
    num_images_total: int = 0
    num_images_processed: int = 0
    results: list[SearchResult] = field(default_factory=list)


# Maps search_id -> SearchJob. Kept alongside visualizer.backend.jobs.JOB_STORE; a search
# job outlives any particular frontend tab as long as the backend process is alive.
SEARCH_JOB_STORE: dict[str, SearchJob] = {}


def iter_images_recursive(folder: Path) -> Iterator[Path]:
    """Yield every supported image file under *folder*, recursively, in sorted order."""
    for path in sorted(folder.rglob("*")):
        if path.is_file() and path.suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS:
            yield path


def run_semantic_search(
    search_job: SearchJob,
    model: BaseModel,
    model_type: str,
    query_embedding: list[float],
) -> None:
    """Run the nearest-neighbour search for *search_job* and mutate it in place.

    Intended to run in a background thread; ``search_job`` is polled from the
    request-handling thread via the ``/semantic-search/{search_id}`` endpoint.

    Args:
        search_job: The search job whose state this call fills in.
        model: Model used to extract per-query embeddings and predictions for the
            images under ``search_job.search_path``.
        model_type: The model type/registry key, stored in the on-disk cache purely as
            informational metadata.
        query_embedding: The raw (full-dimensionality) embedding to search for.
    """
    search_job.status = "running"
    logger.info(
        f"[search {search_job.id}] starting: folder='{search_job.search_path}', k={search_job.k}"
    )
    try:
        folder = Path(search_job.search_path)
        if not folder.exists() or not folder.is_dir():
            raise ValueError(f"Search folder not found: {folder}")

        cache = SearchCache(folder, model_path=str(model.model_path), model_type=model_type)

        images = list(iter_images_recursive(folder))
        search_job.num_images_total = len(images)
        logger.info(f"[search {search_job.id}] found {len(images)} image(s) to scan")

        query_vec = np.asarray(query_embedding, dtype=np.float32)
        query_norm = float(np.linalg.norm(query_vec)) or 1.0

        # Min-of-max-heap of size k, keyed by *negative* distance so the worst
        # (largest-distance) kept result sits at heap[0] and can be evicted in O(log k).
        # A monotonically increasing tie-breaker avoids ever comparing SearchResult objects.
        heap: list[tuple[float, int, SearchResult]] = []
        tie_breaker = 0

        def consider_image(image_path: Path, detections: list[tuple[Prediction, list[float]]]) -> None:
            nonlocal tie_breaker
            best_result: SearchResult | None = None
            for pred, embedding in detections:
                vec = np.asarray(embedding, dtype=np.float32)
                distance = _cosine_distance(query_vec, query_norm, vec)
                if best_result is None or distance < best_result.distance:
                    best_result = SearchResult(
                        image_path=str(image_path),
                        bbox=pred.bbox,
                        confidence=pred.confidence,
                        class_id=pred.class_id,
                        distance=distance,
                    )
            if best_result is None:
                return
            tie_breaker += 1
            if len(heap) < search_job.k:
                heapq.heappush(heap, (-best_result.distance, tie_breaker, best_result))
            elif -heap[0][0] > best_result.distance:
                heapq.heapreplace(heap, (-best_result.distance, tie_breaker, best_result))

        processed = 0
        num_cache_hits = 0
        pending: list[Path] = []
        for image_path in images:
            image_key = str(image_path)
            if cache.is_scanned(image_key):
                num_cache_hits += 1
                consider_image(image_path, cache.get_cached(image_key))
                processed += 1
                search_job.num_images_processed = processed
                continue
            pending.append(image_path)

        for batch_start in range(0, len(pending), _BATCH_SIZE):
            batch_paths = pending[batch_start : batch_start + _BATCH_SIZE]
            embeddings, predictions = model.get_batch_embeddings(batch_paths)

            for image_path, image_embeddings, image_predictions in zip(
                batch_paths, embeddings, predictions
            ):
                detections: list[tuple[Prediction, list[float]]] = []
                if image_embeddings is not None and len(image_embeddings) > 0:
                    vecs = image_embeddings.detach().cpu().numpy().astype(np.float32)
                    detections = [
                        (pred, vec.tolist()) for vec, pred in zip(vecs, image_predictions)
                    ]
                cache.store(str(image_path), detections)
                consider_image(image_path, detections)

            processed += len(batch_paths)
            search_job.num_images_processed = processed
            logger.info(
                f"[search {search_job.id}] {processed}/{len(images)} image(s) scanned"
            )

        logger.info(
            f"[search {search_job.id}] {num_cache_hits}/{len(images)} image(s) served from cache"
        )

        heap.sort(key=lambda entry: -entry[0])  # ascending distance
        search_job.results = [entry[2] for entry in heap]
        search_job.status = "done"
        logger.info(
            f"[search {search_job.id}] done: kept {len(search_job.results)} neighbour(s) "
            f"out of {len(images)} image(s) scanned"
        )
    except Exception as e:
        logger.error(f"[search {search_job.id}] failed: {e}", exc_info=True)
        search_job.error = str(e)
        search_job.status = "error"


def _cosine_distance(query_vec: np.ndarray, query_norm: float, vec: np.ndarray) -> float:
    """Return ``1 - cosine_similarity(query_vec, vec)``, in ``[0, 2]`` (0 = identical)."""
    vec_norm = float(np.linalg.norm(vec)) or 1.0
    similarity = float(np.dot(query_vec, vec) / (query_norm * vec_norm))
    return 1.0 - similarity


__all__ = ["SearchJob", "SearchResult", "SEARCH_JOB_STORE", "run_semantic_search", "iter_images_recursive"]
