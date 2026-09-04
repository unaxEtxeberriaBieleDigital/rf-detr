# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

from pathlib import Path
import threading

from rfdetr.utilities.logger import get_logger
from visualizer.backend.datasets.basedataset import BaseDataset, Split
from visualizer.backend.evaluator import match_detections
from visualizer.backend.dataset_inference_store import DatasetInferenceStore
from visualizer.backend.inference.types import DatasetInferenceJobStatus, EmbeddingRecord
from visualizer.backend.models.basemodel import BaseModel

logger = get_logger()

# Maps dataset_inference_job_id -> Job.  The actual records live in the Job's SQLite DB, not in RAM.
DATASET_INFERENCE_JOB_STORE: dict[str, DatasetInferenceJobStatus] = {}
_ACTIVE_DATASET_INFERENCE_JOB_IDS_BY_PATH: dict[str, str] = {}
_ACTIVE_DATASET_INFERENCE_JOB_IDS_LOCK = threading.Lock()


def dataset_inference_job_path_key(dataset_path: str | Path) -> str:
    """Return a normalized key used for one dataset/dataset_inference_job DB path."""
    return str(Path(dataset_path).resolve())


def try_register_active_dataset_inference_job(dataset_path: str | Path, dataset_inference_job_id: str) -> str | None:
    """Reserve *dataset_path* for one active inference dataset_inference_job.

    Args:
        dataset_path: Dataset root directory.
        dataset_inference_job_id: In-memory dataset_inference_job identifier attempting to run.

    Returns:
        Existing active dataset_inference_job id when the dataset is already reserved, otherwise ``None``.
    """
    key = dataset_inference_job_path_key(dataset_path)
    with _ACTIVE_DATASET_INFERENCE_JOB_IDS_LOCK:
        existing_dataset_inference_job_id = _ACTIVE_DATASET_INFERENCE_JOB_IDS_BY_PATH.get(key)
        if existing_dataset_inference_job_id is not None:
            return existing_dataset_inference_job_id
        _ACTIVE_DATASET_INFERENCE_JOB_IDS_BY_PATH[key] = dataset_inference_job_id
    return None


def release_active_dataset_inference_job(dataset_path: str | Path, dataset_inference_job_id: str) -> None:
    """Release a dataset-path reservation held by *dataset_inference_job_id*."""
    key = dataset_inference_job_path_key(dataset_path)
    with _ACTIVE_DATASET_INFERENCE_JOB_IDS_LOCK:
        if _ACTIVE_DATASET_INFERENCE_JOB_IDS_BY_PATH.get(key) == dataset_inference_job_id:
            del _ACTIVE_DATASET_INFERENCE_JOB_IDS_BY_PATH[key]


def get_active_dataset_inference_job_id(dataset_path: str | Path) -> str | None:
    """Return the active in-memory dataset_inference_job id for *dataset_path*, if any."""
    with _ACTIVE_DATASET_INFERENCE_JOB_IDS_LOCK:
        return _ACTIVE_DATASET_INFERENCE_JOB_IDS_BY_PATH.get(dataset_inference_job_path_key(dataset_path))


def run_dataset_inference_job(
    dataset_inference_job: DatasetInferenceJobStatus,
    dataset: BaseDataset,
    model: BaseModel,
    splits: list[Split],
    batch_size: int,
    iou_threshold: float,
    resume: bool = False,
) -> None:
    """Run inference over *splits*, match predictions to ground truth, and persist to SQLite.

    Raw embeddings (full dimensionality) are written to the DB per batch so that memory
    usage stays bounded. Dimensionality reduction is NOT performed here; it is
    triggered on demand via the
    ``POST /api/v1/dataset_inference_jobs/{dataset_inference_job_id}/dimensionality_reduction`` endpoint.

    Mutates ``dataset_inference_job`` in place (``status``, ``error``, ``num_images_*``) so callers can poll
    it from another thread.  Intended to be run in a background thread, not on the
    request-handling thread.

    Args:
        dataset_inference_job: The dataset_inference_job whose state this call fills in.
        dataset: Dataset providing image batches and ground truth for each split.
        model: Model used to extract per-query embeddings and predictions.
        splits: Dataset splits to process.
        batch_size: Number of images per inference batch.
        iou_threshold: Minimum IoU for a prediction to be matched to a ground truth box.
        resume: Whether to resume an interrupted run from persisted per-image progress.
    """
    dataset_inference_job.status = "running"
    dataset_inference_job.error = None
    dataset_inference_job.store.set_meta("status", "running")
    dataset_inference_job.store.set_meta("error", None)
    logger.info(
        f"[dataset_inference_job {dataset_inference_job.id}] starting: splits={[s.name for s in splits]}, "
        f"batch_size={batch_size}, iou_threshold={iou_threshold}, resume={resume}"
    )
    try:
        split_names = [split.name for split in splits]
        processed_image_paths = (
            dataset_inference_job.store.get_processed_image_paths(split_names=split_names) if resume else set()
        )

        dataset_inference_job.num_images_total = 0
        dataset_inference_job.num_images_processed = 0
        for split in splits:
            for image_path in dataset.iter_split(split):
                dataset_inference_job.num_images_total += 1
                if str(image_path) in processed_image_paths:
                    dataset_inference_job.num_images_processed += 1

        dataset_inference_job.store.set_meta("num_images_total", dataset_inference_job.num_images_total)
        dataset_inference_job.store.set_meta("num_images_processed", dataset_inference_job.num_images_processed)
        logger.info(
            f"[dataset_inference_job {dataset_inference_job.id}] found {dataset_inference_job.num_images_total} image(s) across {len(splits)} split(s); "
            f"{dataset_inference_job.num_images_processed} already processed"
        )

        total_records = dataset_inference_job.store.record_count()

        for split in splits:
            logger.info(f"[dataset_inference_job {dataset_inference_job.id}] processing split '{split.name}'")
            for batch in dataset.iter_batches(split, batch_size):
                pending_batch = [image_path for image_path in batch if str(image_path) not in processed_image_paths]
                if not pending_batch:
                    continue

                embeddings, predictions = model.get_batch_embeddings(pending_batch)

                batch_records: list[EmbeddingRecord] = []
                processed_batch_images: list[tuple[str, str]] = []
                for image_path, image_embeddings, image_predictions in zip(
                    pending_batch, embeddings, predictions
                ):
                    ground_truths = dataset.get_ground_truth(image_path)
                    matches = match_detections(
                        image_predictions, image_embeddings, ground_truths, iou_threshold
                    )

                    for match_idx, match in enumerate(matches):
                        record = EmbeddingRecord(
                            id=f"{split.name}:{Path(image_path).name}:{match_idx}",
                            image_path=str(image_path),
                            split=split.name,
                            embedding=match.embedding,
                            prediction=match.prediction,
                            ground_truth=match.ground_truth,
                            status=match.status,
                        )
                        batch_records.append(record)

                    processed_batch_images.append((str(image_path), split.name))

                # Write batch atomically – records, per-image progress, and counters together.
                dataset_inference_job.num_images_processed += len(processed_batch_images)
                dataset_inference_job.store.persist_batch(
                    batch_records,
                    processed_batch_images,
                    num_images_processed=dataset_inference_job.num_images_processed,
                )
                processed_image_paths.update(image_path for image_path, _ in processed_batch_images)
                total_records += len(batch_records)
                logger.info(
                    f"[dataset_inference_job {dataset_inference_job.id}] {dataset_inference_job.num_images_processed}/{dataset_inference_job.num_images_total} "
                    f"image(s) processed ({total_records} record(s) so far)"
                )

        dataset_inference_job.status = "done"
        dataset_inference_job.store.set_meta("status", "done")
        logger.info(
            f"[dataset_inference_job {dataset_inference_job.id}] done: {total_records} record(s) from "
            f"{dataset_inference_job.num_images_total} image(s). Run dimensionality reduction to enable "
            "the scatter plot."
        )
    except Exception as e:
        logger.error(f"[dataset_inference_job {dataset_inference_job.id}] failed: {e}", exc_info=True)
        dataset_inference_job.error = str(e)
        dataset_inference_job.status = "error"
        dataset_inference_job.store.set_meta("status", "error")
        dataset_inference_job.store.set_meta("error", str(e))
    finally:
        release_active_dataset_inference_job(dataset_inference_job.store.dataset_path, dataset_inference_job.id)
