# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

"""Data types used by dataset inference."""

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal

from visualizer.backend.shared_types.prediction import Prediction

if TYPE_CHECKING:
    from visualizer.backend.dataset_inference_store import DatasetInferenceStore

DatasetInferenceJobState = Literal["pending", "running", "done", "error"]


@dataclass
class DatasetInferenceJobStatus:
    """State for one embedding-extraction and evaluation run."""

    id: str
    store: "DatasetInferenceStore"
    status: DatasetInferenceJobState = "pending"
    error: str | None = None
    categories: dict[int, str] = field(default_factory=dict)
    num_images_total: int = 0
    num_images_processed: int = 0


@dataclass
class EmbeddingRecord:
    """Persisted embedding, prediction, and ground-truth record."""

    id: str
    image_path: str
    split: str
    embedding: list[float] | None
    prediction: Prediction | None
    ground_truth: Prediction | None
    status: Literal["tp", "fp", "fn", "misclassified", "correct", "incorrect"]


__all__ = ["DatasetInferenceJobState", "DatasetInferenceJobStatus", "EmbeddingRecord"]
