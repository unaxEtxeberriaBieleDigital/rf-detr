# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

from dataclasses import dataclass


@dataclass
class Prediction:
    class_id: int
    confidence: float
    bbox: tuple[float, float, float, float] | None = None  # None en clasificación
