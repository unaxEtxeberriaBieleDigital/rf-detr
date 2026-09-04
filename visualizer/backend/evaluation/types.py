# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

"""Data types used by model evaluation."""

from dataclasses import dataclass

from visualizer.backend.shared_types.prediction import Prediction


@dataclass
class Match:
    """One resolved prediction and ground-truth pairing."""

    prediction: Prediction | None
    embedding: list[float] | None
    ground_truth: Prediction | None
    status: str


__all__ = ["Match"]
