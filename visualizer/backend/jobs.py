# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import numpy as np
from sklearn.decomposition import PCA

from rfdetr.utilities.logger import get_logger
from visualizer.backend.datasets.basedataset import BaseDataset, Split
from visualizer.backend.embeddingrecord import EmbeddingRecord
from visualizer.backend.evaluator import match_detections
from visualizer.backend.models.basemodel import BaseModel

logger = get_logger()

JobStatus = Literal["pending", "running", "done", "error"]


@dataclass
class Job:
    """In-memory state for one embedding-extraction-and-evaluation run."""

    id: str
    status: JobStatus = "pending"
    error: str | None = None
    records: list[EmbeddingRecord] = field(default_factory=list)
    categories: dict[int, str] = field(default_factory=dict)
    num_images_total: int = 0
    num_images_processed: int = 0


# Simple in-memory job store. Fine for a single-process dev server; swap for a persistent
# store (e.g. sqlite, redis) if the API needs to survive restarts or run multi-process.
JOB_STORE: dict[str, Job] = {}


def run_job(
    job: Job,
    dataset: BaseDataset,
    model: BaseModel,
    splits: list[Split],
    batch_size: int,
    iou_threshold: float,
    pca_components: int,
) -> None:
    """Runs inference over `splits`, matches predictions to ground truth, and reduces embeddings with PCA.

    Mutates `job` in place (`status`, `records`, `error`) so callers can poll it from another
    thread. Intended to be run in a background thread/task, not on the request-handling thread.

    Args:
        job: The job whose state this call fills in.
        dataset: Dataset providing image batches and ground truth for each split.
        model: Model used to extract per-query embeddings and predictions.
        splits: Dataset splits to process.
        batch_size: Number of images per inference batch.
        iou_threshold: Minimum IoU for a prediction to be matched to a ground truth box.
        pca_components: Target dimensionality for the PCA projection of the embeddings.
    """
    job.status = "running"
    logger.info(
        f"[job {job.id}] starting: splits={[s.name for s in splits]}, batch_size={batch_size}, "
        f"iou_threshold={iou_threshold}, pca_components={pca_components}"
    )
    try:
        # Count images up front (cheap directory listing) so progress can be reported as a
        # fraction of the total instead of just a running count.
        job.num_images_total = sum(len(list(dataset.iter_split(split))) for split in splits)
        logger.info(f"[job {job.id}] found {job.num_images_total} image(s) across {len(splits)} split(s)")

        records: list[EmbeddingRecord] = []
        raw_embeddings: list[list[float]] = []
        embedded_record_indices: list[int] = []

        for split in splits:
            logger.info(f"[job {job.id}] processing split '{split.name}'")
            for batch in dataset.iter_batches(split, batch_size):
                embeddings, predictions = model.get_batch_embeddings(batch)

                for image_path, image_embeddings, image_predictions in zip(batch, embeddings, predictions):
                    ground_truths = dataset.get_ground_truth(image_path)
                    matches = match_detections(image_predictions, image_embeddings, ground_truths, iou_threshold)

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
                        if match.embedding is not None:
                            embedded_record_indices.append(len(records))
                            raw_embeddings.append(match.embedding)
                        records.append(record)

                job.num_images_processed += len(batch)
                logger.info(
                    f"[job {job.id}] {job.num_images_processed}/{job.num_images_total} image(s) processed "
                    f"({len(records)} record(s) so far)"
                )

        if raw_embeddings:
            n_components = min(pca_components, len(raw_embeddings), len(raw_embeddings[0]))
            logger.info(f"[job {job.id}] reducing {len(raw_embeddings)} embedding(s) to {n_components}D with PCA")
            coords = PCA(n_components=n_components).fit_transform(np.array(raw_embeddings, dtype=np.float32))
            for record_idx, coord in zip(embedded_record_indices, coords.tolist()):
                records[record_idx].embedding = coord

        job.records = records
        job.status = "done"
        logger.info(f"[job {job.id}] done: {len(records)} record(s) from {job.num_images_total} image(s)")
    except Exception as e:
        logger.error(f"[job {job.id}] failed: {e}", exc_info=True)
        job.error = str(e)
        job.status = "error"
