# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

"""Semantic (nearest-neighbour) search over an arbitrary folder, via a pluggable source.

Given a query embedding (taken from an existing prediction in a visualizer job), this module runs the model over every
unit of work produced by a :class:`BaseSemanticSearchSource` (e.g. one unit per image, or one unit per tile of a large
image) and keeps the ``k`` results whose closest detection is nearest to the query, ranked by cosine distance. Units
sharing the same ``group_key`` (e.g. every tile of the same source image) contribute at most one result -- their single
best-matching detection -- so a group never appears twice among the neighbours.

Per-unit inference results (embeddings + predictions) are cached in a small SQLite database at the root of the searched
folder (see ``visualizer.backend.semantic_search.cache``), so re-running a search against the same folder with the same
model only needs to recompute cosine distances -- inference is skipped entirely for already-scanned units.

Runs in a background thread, mirroring ``visualizer.backend.jobs``: the search keeps progressing even if no client is
polling it, and the frontend can reattach to it (by ``search_id``) at any time to read the current progress or final
results.
"""

import heapq
import threading
from pathlib import Path

import numpy as np
from PIL import Image

from rfdetr.utilities.logger import get_logger
from visualizer.backend.models.basemodel import BaseModel
from visualizer.backend.semantic_search.cache import SearchCache
from visualizer.backend.semantic_search.sources.basesource import BaseSemanticSearchSource
from visualizer.backend.semantic_search.types import ScanUnit, SearchJob, SearchResult
from visualizer.backend.shared_types.prediction import Prediction

logger = get_logger()

# Number of units sent to the model per inference call.
_BATCH_SIZE = 8


# Maps search_id -> SearchJob. Kept alongside visualizer.backend.jobs.JOB_STORE; a search
# job outlives any particular frontend tab as long as the backend process is alive.
SEARCH_JOB_STORE: dict[str, SearchJob] = {}
SEARCH_JOB_STORE_LOCK = threading.Lock()


def try_register_active_search(search_job: SearchJob) -> bool:
    """Atomically register a search if no other search is pending or running.

    Args:
        search_job: The new search job to register.

    Returns:
        ``True`` when the job was registered, otherwise ``False``.
    """
    with SEARCH_JOB_STORE_LOCK:
        if any(search.status in {"pending", "running"} for search in SEARCH_JOB_STORE.values()):
            return False
        SEARCH_JOB_STORE[search_job.id] = search_job
        return True


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
    logger.info(f"[search {search_job.id}] starting: folder='{search_job.search_path}', k={search_job.k}")
    try:
        folder = Path(search_job.search_path)
        if not folder.exists() or not folder.is_dir():
            raise ValueError(f"Search folder not found: {folder}")

        cache = SearchCache(
            folder,
            model_path=str(model.model_path),
            model_type=model_type,
            source_signature=source.cache_signature(model),
        )

        manifests = cache.list_group_manifests()
        scan_plan: list[tuple[Path, int, int, int, bool]] = []
        num_units = 0
        for path in source.iter_group_paths(folder):
            try:
                stat = path.stat()
                group_key = str(path)
                manifest = manifests.get(group_key)
                if (
                    manifest is not None
                    and manifest.file_size == stat.st_size
                    and manifest.file_mtime_ns == stat.st_mtime_ns
                ):
                    cache_hit = True
                    unit_count = manifest.unit_count
                else:
                    cache_hit = False
                    unit_count = source.get_num_units_for_group(path, model)
            except OSError as e:
                logger.warning(f"[search {search_job.id}] skipping image '{path}': {e}")
                continue
            scan_plan.append((path, stat.st_size, stat.st_mtime_ns, unit_count, cache_hit))
            num_units += unit_count
        search_job.num_images_total = num_units
        logger.info(f"[search {search_job.id}] found {num_units} unit(s) to scan")

        query_vec = np.asarray(query_embedding, dtype=np.float32)
        query_norm = float(np.linalg.norm(query_vec)) or 1.0

        # Best result seen so far per group_key (e.g. per source image), so units sharing a
        # group (like tiles of the same image) still contribute at most one final result.
        best_by_group: dict[str, SearchResult] = {}

        def consider_unit(unit_id: str, group_key: str, detections: list[tuple[Prediction, list[float]]]) -> None:
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

        # Actualizar la lista expuesta a la API con el Top-K actual
        def update_top_k() -> None:
            search_job.results = heapq.nsmallest(search_job.k, best_by_group.values(), key=lambda r: r.distance)

        processed = 0
        num_cache_hits = 0
        invalid_groups: set[str] = set()

        # Buffer de unidades pendientes de inferencia. Se vacía en cuanto llega a
        # _BATCH_SIZE, de modo que nunca hay más de un lote de unidades en memoria.
        pending: list[tuple[ScanUnit, str]] = []
        incomplete_groups: dict[str, list[int]] = {}
        group_file_stats: dict[str, tuple[int, int]] = {}

        def flush_pending() -> None:
            """Run inference over the buffered units, then cache and score their detections."""
            nonlocal processed
            if not pending:
                return
            all_batch_entries = list(pending)
            batch_entries = all_batch_entries
            pending.clear()
            try:
                batch_detections = source.process_batch(model, [entry[0] for entry in batch_entries])
            except OSError:
                # A batch-level image read error can be caused by one bad file. Retry per
                # unit so valid inputs in the same batch are not lost with it.
                batch_detections = []
                valid_entries: list[tuple[ScanUnit, str]] = []
                failed_entries: list[tuple[ScanUnit, str]] = []
                for entry in batch_entries:
                    unit, cache_group_key = entry
                    try:
                        unit_detections = source.process_batch(model, [unit])
                    except OSError as e:
                        if not _is_unreadable_image_file(unit.inference_input):
                            raise
                        logger.warning(f"[search {search_job.id}] skipping image '{cache_group_key}': {e}")
                        invalid_groups.add(cache_group_key)
                        cache.invalidate_group(cache_group_key)
                        best_by_group.pop(unit.group_key, None)
                        failed_entries.append(entry)
                        continue
                    if len(unit_detections) != 1:
                        raise ValueError(
                            f"Semantic-search source returned {len(unit_detections)} result sets for a single unit"
                        )
                    valid_entries.append(entry)
                    batch_detections.extend(unit_detections)
                for _, cache_group_key in failed_entries:
                    incomplete_groups[cache_group_key][0] -= 1
                batch_entries = valid_entries

            batch_units = [entry[0] for entry in batch_entries]
            if len(batch_detections) != len(batch_units):
                raise ValueError(
                    f"Semantic-search source returned {len(batch_detections)} result sets "
                    f"for a batch of {len(batch_units)} units"
                )
            for (unit, cache_group_key), detections in zip(batch_entries, batch_detections):
                if cache_group_key not in invalid_groups:
                    cache.store_unit(unit.id, cache_group_key, detections)
                    consider_unit(unit.id, unit.group_key, detections)
                incomplete_groups[cache_group_key][0] -= 1
                if incomplete_groups[cache_group_key][0] == 0 and cache_group_key not in invalid_groups:
                    file_size, file_mtime_ns = group_file_stats[cache_group_key]
                    cache.mark_group_complete(
                        cache_group_key,
                        file_size,
                        file_mtime_ns,
                        incomplete_groups[cache_group_key][1],
                    )

            processed += len(all_batch_entries)
            search_job.num_images_processed = processed
            logger.info(f"[search {search_job.id}] {processed}/{num_units} unit(s) scanned")

            # Por cada lote procesado por la red neuronal actualiza
            update_top_k()

        # Los grupos válidos se leen directamente desde SQLite. Solo un miss materializa
        # sus ScanUnit, lo que evita abrir o decodificar imágenes cacheadas (también tiled).
        for path, file_size, file_mtime_ns, expected_units, cache_hit in scan_plan:
            if search_job.status == "cancelled":
                logger.info(
                    f"[search {search_job.id}] cancelled during scan ({processed}/{num_units} unit(s) scanned)."
                )
                break

            cache_group_key = str(path)
            if cache_hit:
                try:
                    path.stat()
                except OSError as e:
                    logger.warning(f"[search {search_job.id}] skipping image '{path}': {e}")
                    num_units -= expected_units
                    search_job.num_images_total = num_units
                    cache.invalidate_group(cache_group_key)
                    continue
                cached_units = cache.get_cached_group(cache_group_key)
                if len(cached_units) == expected_units:
                    for unit_id, detections in cached_units:
                        consider_unit(unit_id, cache_group_key, detections)
                        processed += 1
                        num_cache_hits += 1
                        search_job.num_images_processed = processed
                        if processed % 50 == 0:
                            update_top_k()
                    continue

                logger.warning(
                    f"[search {search_job.id}] invalid cache manifest for '{cache_group_key}': "
                    f"expected {expected_units} unit(s), found {len(cached_units)}"
                )
                try:
                    replacement_count = source.get_num_units_for_group(path, model)
                except OSError as e:
                    logger.warning(f"[search {search_job.id}] skipping image '{path}': {e}")
                    num_units -= expected_units
                    search_job.num_images_total = num_units
                    cache.invalidate_group(cache_group_key)
                    continue
                num_units += replacement_count - expected_units
                search_job.num_images_total = num_units
                expected_units = replacement_count

            cache.invalidate_group(cache_group_key)
            incomplete_groups[cache_group_key] = [expected_units, expected_units]
            group_file_stats[cache_group_key] = (file_size, file_mtime_ns)
            if expected_units == 0:
                cache.mark_group_complete(cache_group_key, file_size, file_mtime_ns, 0)
                continue

            generated_units = 0
            try:
                for unit in source.iter_scan_units_for_group(path, model):
                    if search_job.status == "cancelled":
                        break
                    pending.append((unit, cache_group_key))
                    generated_units += 1
                    if len(pending) >= _BATCH_SIZE:
                        flush_pending()
            except OSError as e:
                if generated_units > 0:
                    raise
                logger.warning(f"[search {search_job.id}] skipping image '{path}': {e}")
                num_units -= expected_units
                search_job.num_images_total = num_units
                cache.invalidate_group(cache_group_key)
                incomplete_groups.pop(cache_group_key, None)
                group_file_stats.pop(cache_group_key, None)
                continue

            if search_job.status == "cancelled":
                break
            if generated_units != expected_units:
                raise ValueError(
                    f"Semantic-search source counted {expected_units} unit(s) for '{path}' "
                    f"but generated {generated_units}"
                )

        # Último lote incompleto (descartamos lo pendiente si se ha cancelado la búsqueda)
        if search_job.status != "cancelled":
            flush_pending()

        logger.info(f"[search {search_job.id}] {num_cache_hits}/{num_units} unit(s) served from cache")

        # Actualización final
        update_top_k()

        if search_job.status != "cancelled":
            search_job.status = "done"
            logger.info(
                f"[search {search_job.id}] done: kept {len(search_job.results)} neighbour(s) "
                f"out of {len(best_by_group)} group(s) scanned"
            )
        else:
            logger.info(
                f"[search {search_job.id}] cancelled: {processed}/{num_units} unit(s) scanned, "
                f"kept {len(search_job.results)} neighbour(s) before stopping"
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


def _is_unreadable_image_file(inference_input: str | Path | np.ndarray) -> bool:
    """Return whether a path-like inference input cannot be decoded as an image."""
    if not isinstance(inference_input, (str, Path)):
        return False
    try:
        with Image.open(inference_input) as image:
            image.load()
    except OSError:
        return True
    return False


__all__ = [
    "SearchJob",
    "SearchResult",
    "SEARCH_JOB_STORE",
    "SEARCH_JOB_STORE_LOCK",
    "try_register_active_search",
    "run_semantic_search",
]
