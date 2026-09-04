# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

"""Data types used by the metrics subsystem."""

from dataclasses import dataclass
from enum import Enum


class MetricType(Enum):
    """Categorizes how a metric should be rendered."""

    SCALAR = "scalar"
    CURVE = "curve"
    MATRIX = "matrix"


@dataclass
class MetricDefinition:
    """Describes a metric that can be calculated."""

    name: str
    display_name: str
    description: str
    metric_type: MetricType


__all__ = ["MetricType", "MetricDefinition"]
