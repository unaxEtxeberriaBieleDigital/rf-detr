# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from visualizer.backend.evaluator import Match


class MetricType(Enum):
    """Categorizes how a metric should be rendered."""
    SCALAR = "scalar"  # Single number: mAP, accuracy, etc.
    CURVE = "curve"  # Line chart: x,y pairs (threshold vs mAP)
    MATRIX = "matrix"  # 2D data: confusion matrix


@dataclass
class MetricDefinition:
    """Describes a single metric that can be calculated.
    
    Attributes:
        name: Unique identifier (snake_case), used in API and storage.
        display_name: Human-readable title shown in UI.
        description: Long-form explanation for tooltips.
        metric_type: How the metric should be visualized (scalar/curve/matrix).
    """
    name: str
    display_name: str
    description: str
    metric_type: MetricType


# MetricValue can be:
# - float: for scalar metrics (mAP, accuracy, etc.)
# - list[tuple[float, float]]: for curve metrics (threshold, value) pairs
# - list[list[float]]: for matrix metrics (confusion matrix as 2D array)
MetricValue = float | list[tuple[float, float]] | list[list[float]] | dict[str, Any]


class MetricsCalculator(ABC):
    """Abstract base class for dataset-specific metrics calculation.
    
    Each dataset type (COCO detection, classification, etc.) provides its own
    implementation to define which metrics are available and how to calculate them.
    
    Example:
        class COCODetectionMetricsCalculator(MetricsCalculator):
            def supported_metrics(self):
                return [
                    MetricDefinition("mAP50", "Mean Average Precision @IoU=0.5", ...),
                    ...
                ]
            
            def calculate(self, matches):
                return {"mAP50": 0.85, "accuracy": 0.92, ...}
    """

    @abstractmethod
    def supported_metrics(self) -> list[MetricDefinition]:
        """Return list of all metrics this calculator can compute.
        
        Returns:
            List of MetricDefinition objects describing available metrics.
        """
        pass

    @abstractmethod
    def calculate(self, matches: list["Match"]) -> dict[str, MetricValue]:
        """Calculate all supported metrics from detection matches.
        
        Args:
            matches: List of Match objects from match_detections() in evaluator.py
            
        Returns:
            Dictionary mapping metric name (str) to MetricValue (float, curve, or matrix).
            Only includes metrics that were successfully calculated.
        """
        pass

    @abstractmethod
    def get_optimal_threshold(
        self,
        metric_name: str,
        matches: list["Match"],
        num_thresholds: int = 100,
    ) -> float:
        """Find confidence threshold that maximizes a given metric.
        
        Iterates through candidate thresholds [0.0 ... 1.0] and finds the one
        that produces the highest value for the target metric.
        
        Args:
            metric_name: Name of metric to optimize for (e.g., "mAP50").
            matches: List of Match objects (predictions already sorted by confidence).
            num_thresholds: Number of thresholds to test between 0 and 1.
            
        Returns:
            Optimal confidence threshold as float [0.0, 1.0].
            
        Raises:
            ValueError: If metric_name is not supported or cannot be optimized.
        """
        pass
