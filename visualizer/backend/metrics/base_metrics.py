# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

from visualizer.backend.metrics.types import MetricDefinition, MetricType

if TYPE_CHECKING:
    from visualizer.backend.evaluation.types import Match


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
    def calculate(
        self,
        matches: list["Match"],
        class_thresholds: dict[int, float] | None = None,
    ) -> dict[str, MetricValue]:
        """Calculate all supported metrics from detection matches.

        Args:
            matches: List of Match objects from match_detections() in evaluator.py
            class_thresholds: Optional mapping of predicted class id to confidence
                threshold. Predictions below their class threshold are excluded
                before metrics are calculated.

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
        class_id: int | None = None,
    ) -> float:
        """Find confidence threshold that maximizes a given metric.

        Iterates through candidate thresholds [0.0 ... 1.0] and finds the one
        that produces the highest value for the target metric.

        Args:
            metric_name: Name of metric to optimize for (e.g., "mAP50").
            matches: List of Match objects (predictions already sorted by confidence).
            num_thresholds: Number of thresholds to test between 0 and 1.
            class_id: Optional predicted class id to optimize. When provided,
                only matches for predictions of that class are thresholded,
                while false negatives remain in scope.

        Returns:
            Optimal confidence threshold as float [0.0, 1.0].

        Raises:
            ValueError: If metric_name is not supported or cannot be optimized.
        """
        pass
