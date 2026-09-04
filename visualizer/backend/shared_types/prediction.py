# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

"""Model prediction data type shared across backend domains."""

from dataclasses import dataclass


@dataclass
class Prediction:
    """One classification or object-detection prediction."""

    class_id: int
    confidence: float
    bbox: tuple[float, float, float, float] | None = None


__all__ = ["Prediction"]
