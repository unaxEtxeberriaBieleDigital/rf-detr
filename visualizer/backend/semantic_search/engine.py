# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

"""Semantic (nearest-neighbour) search over an arbitrary folder, via a pluggable source.

Given a query embedding (taken from an existing prediction in a visualizer job), this
module runs the model over every unit of work produced by a :class:`BaseSemanticSearchSource`
(e.g. one unit per image, or one unit per tile of a large image) and keeps the ``k``
results whose closest detection is nearest to the query, ranked by cosine distance. Units
sharing the same ``group_key`` (e.g. every tile of the same source image) contribute at
most one result -- their single best-matching detection -- so a group never appears twice
among the neighbours.

Per-unit inference results (embeddings + predictions) are cached in a small SQLite
database at the root of the searched folder (see ``visualizer.backend.semantic_search.cache``),
so re-running a search against the same folder with the same model only needs to recompute
cosine distances -- inference is skipped entirely for already-scanned units.

Runs in a background thread, mirroring ``visualizer.backend.jobs``: the search keeps
progressing even if no client is polling it, and the frontend can reattach to it (by
``search_id``) at any time to read the current progress or final results.
"""

import heapq
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import numpy as np

from rfdetr.utilities.logger import get_logger
from visualizer.backend.models.basemodel import BaseModel
from visualizer.backend.prediction import Prediction
from visualizer.backend.semantic_search.sources.basesource import BaseSemanticSearchSource
from visualizer.backend.semantic_search.cache import SearchCache

logger = get_logger()

SearchStatus = Literal["pending", "running", "done", "error"]

# Number of units sent to the model per inference call.
_BATCH_SIZE = 8


@dataclass
class SearchResult:
    """One nearest-neighbour hit: the best-matching detection within one result group."""

    image_path: str
    prediction: Prediction
    distance: float
    # The id of the ScanUnit this detection came from (e.g. a plain image path, or a tile
    # id like "img.tif::tile_0_0_560_560"). Sources use this to re-render exactly the same
    # unit that was fed to the model when producing a preview, instead of an arbitrary crop
    # around the bbox -- see e.g. TiledImageSource.render_result_preview.
    unit_id: str


@dataclass
class SearchJob:
    """State for one semantic-search run, kept fully in memory."""

    id: str
    parent_job_id: str
    query_record_id: str
    query_image_path: str
    search_path: str
    k: int
    source_type: str = "default"
    status: SearchStatus = "pending"
    error: str | None = None
    num_images_total: int = 0
    num_images_processed: int = 0
    results: list[SearchResult] = field(default_factory=list)


# Maps search_id -> SearchJob. Kept alongside visualizer.backend.jobs.JOB_STORE; a search
# job outlives any particular frontend tab as long as the backend process is alive.
SEARCH_JOB_STORE: dict[str, SearchJob] = {}


def run_semantic_search(
    search_job: SearchJob,
    model: BaseModel,
    model_type: str,
    query_embedding: list[float],
    source: BaseSemanticSearchSource,
) -> None:
    """Run the nearest-neighbour search for *search_job* and mutate it in place.

    Intended to run in a background thread; ``search_job`` is polled from the
    request-handling thread via the ``/semantic-search/{search_id}`` endpoint.

    Args:
        search_job: The search job whose state this call fills in.
        model: Model used to extract per-unit embeddings and predictions for the units
            produced by ``source`` over ``search_job.search_path``.
        model_type: The model type/registry key, stored in the on-disk cache purely as
            informational metadata.
        query_embedding: The raw (full-dimensionality) embedding to search for.
        source: Decides how ``search_job.search_path`` is scanned into inference units
            (see :class:`BaseSemanticSearchSource`).
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

        units = list(source.iter_scan_units(folder, model))
        search_job.num_images_total = len(units)
        logger.info(f"[search {search_job.id}] found {len(units)} unit(s) to scan")

        query_vec = np.asarray(query_embedding, dtype=np.float32)
        query_norm = float(np.linalg.norm(query_vec)) or 1.0

        # Best result seen so far per group_key (e.g. per source image), so units sharing a
        # group (like tiles of the same image) still contribute at most one final result.
        best_by_group: dict[str, SearchResult] = {}

        def consider_unit(
            unit_id: str, group_key: str, detections: list[tuple[Prediction, list[float]]]
        ) -> None:
            best_result: SearchResult | None = None
            for pred, embedding in detections:
                vec = np.asarray(embedding, dtype=np.float32)
                distance = _cosine_distance(query_vec, query_norm, vec)
                if best_result is None or distance < best_result.distance:
                    prediction = Prediction(
                        bbox=pred.bbox,
                        confidence=pred.confidence,
                        class_id=pred.class_id,
                    )
                    best_result = SearchResult(
                        image_path=group_key,
                        prediction=prediction,
                        distance=distance,
                        unit_id=unit_id,
                    )
            if best_result is None:
                return
            existing = best_by_group.get(group_key)
            if existing is None or best_result.distance < existing.distance:
                best_by_group[group_key] = best_result

        processed = 0
        num_cache_hits = 0
        pending = []
        for unit in units:
            if cache.is_scanned(unit.id):
                num_cache_hits += 1
                consider_unit(unit.id, unit.group_key, cache.get_cached(unit.id))
                processed += 1
                search_job.num_images_processed = processed
                continue
            pending.append(unit)

        for batch_start in range(0, len(pending), _BATCH_SIZE):
            batch_units = pending[batch_start : batch_start + _BATCH_SIZE]
            batch_detections = source.process_batch(model, batch_units)

            for unit, detections in zip(batch_units, batch_detections):
                cache.store(unit.id, detections)
                consider_unit(unit.id, unit.group_key, detections)

            processed += len(batch_units)
            search_job.num_images_processed = processed
            logger.info(f"[search {search_job.id}] {processed}/{len(units)} unit(s) scanned")

        logger.info(
            f"[search {search_job.id}] {num_cache_hits}/{len(units)} unit(s) served from cache"
        )

        top_k = heapq.nsmallest(search_job.k, best_by_group.values(), key=lambda r: r.distance)
        search_job.results = top_k
        search_job.status = "done"
        logger.info(
            f"[search {search_job.id}] done: kept {len(search_job.results)} neighbour(s) "
            f"out of {len(best_by_group)} group(s) scanned"
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


__all__ = ["SearchJob", "SearchResult", "SEARCH_JOB_STORE", "run_semantic_search"]
