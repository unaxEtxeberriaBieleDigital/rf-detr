# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

from dataclasses import dataclass
from typing import Literal

from visualizer.backend.prediction import Prediction


@dataclass
class EmbeddingRecord:
    id: str
    image_path: str
    split: str
    embedding: list[float] | None  # None for false negatives (no query produced this box)
    prediction: Prediction | None
    ground_truth: Prediction | None
    status: Literal["tp", "fp", "fn", "misclassified", "correct", "incorrect"]
