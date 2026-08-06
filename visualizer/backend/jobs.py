# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from rfdetr.utilities.logger import get_logger
from visualizer.backend.datasets.basedataset import BaseDataset, Split
from visualizer.backend.embeddingrecord import EmbeddingRecord
from visualizer.backend.evaluator import match_detections
from visualizer.backend.models.basemodel import BaseModel
from visualizer.backend.store import JobStore

logger = get_logger()

JobStatus = Literal["pending", "running", "done", "error"]


@dataclass
class Job:
    """State for one embedding-extraction-and-evaluation run, backed by a SQLite DB."""

    id: str
    store: JobStore
    status: JobStatus = "pending"
    error: str | None = None
    categories: dict[int, str] = field(default_factory=dict)
    num_images_total: int = 0
    num_images_processed: int = 0


# Maps job_id -> Job.  The actual records live in the Job's SQLite DB, not in RAM.
JOB_STORE: dict[str, Job] = {}


def run_job(
    job: Job,
    dataset: BaseDataset,
    model: BaseModel,
    splits: list[Split],
    batch_size: int,
    iou_threshold: float,
) -> None:
    """Run inference over *splits*, match predictions to ground truth, and persist to SQLite.

    Raw embeddings (full dimensionality) are written to the DB per batch so that memory
    usage stays bounded.  PCA is NOT performed here; it is triggered on demand via the
    ``POST /api/v1/jobs/{job_id}/pca`` endpoint.

    Mutates ``job`` in place (``status``, ``error``, ``num_images_*``) so callers can poll
    it from another thread.  Intended to be run in a background thread, not on the
    request-handling thread.

    Args:
        job: The job whose state this call fills in.
        dataset: Dataset providing image batches and ground truth for each split.
        model: Model used to extract per-query embeddings and predictions.
        splits: Dataset splits to process.
        batch_size: Number of images per inference batch.
        iou_threshold: Minimum IoU for a prediction to be matched to a ground truth box.
    """
    job.status = "running"
    job.store.set_meta("status", "running")
    logger.info(
        f"[job {job.id}] starting: splits={[s.name for s in splits]}, "
        f"batch_size={batch_size}, iou_threshold={iou_threshold}"
    )
    try:
        job.num_images_total = sum(len(list(dataset.iter_split(split))) for split in splits)
        job.store.set_meta("num_images_total", job.num_images_total)
        logger.info(
            f"[job {job.id}] found {job.num_images_total} image(s) across {len(splits)} split(s)"
        )

        total_records = 0

        for split in splits:
            logger.info(f"[job {job.id}] processing split '{split.name}'")
            for batch in dataset.iter_batches(split, batch_size):
                embeddings, predictions = model.get_batch_embeddings(batch)

                batch_records: list[EmbeddingRecord] = []
                for image_path, image_embeddings, image_predictions in zip(
                    batch, embeddings, predictions
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

                # Write batch to DB immediately – no accumulation in RAM.
                job.store.insert_records(batch_records)
                total_records += len(batch_records)
                job.num_images_processed += len(batch)
                job.store.set_meta("num_images_processed", job.num_images_processed)
                logger.info(
                    f"[job {job.id}] {job.num_images_processed}/{job.num_images_total} "
                    f"image(s) processed ({total_records} record(s) so far)"
                )

        job.status = "done"
        job.store.set_meta("status", "done")
        logger.info(
            f"[job {job.id}] done: {total_records} record(s) from "
            f"{job.num_images_total} image(s).  Run PCA to enable the scatter plot."
        )
    except Exception as e:
        logger.error(f"[job {job.id}] failed: {e}", exc_info=True)
        job.error = str(e)
        job.status = "error"
        job.store.set_meta("status", "error")
        job.store.set_meta("error", str(e))
