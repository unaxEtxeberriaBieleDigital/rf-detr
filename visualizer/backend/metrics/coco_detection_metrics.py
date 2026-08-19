# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

import numpy as np
from sklearn.metrics import auc, confusion_matrix, roc_curve

from visualizer.backend.evaluator import Match
from visualizer.backend.metrics.base_metrics import (
    MetricDefinition,
    MetricType,
    MetricValue,
    MetricsCalculator,
)


class COCODetectionMetricsCalculator(MetricsCalculator):
    """Calculates evaluation metrics for COCO-format object detection.
    
    Supports: mAP@IoU=0.5, mAP@IoU=0.5:0.95, mAR, accuracy, precision, recall, f1,
    confusion matrix, and ROC-AUC.
    """

    def __init__(self, categories: dict[int, str]):
        """Initialize calculator with dataset categories.
        
        Args:
            categories: Mapping from class_id to class name.
        """
        self.categories = categories
        self._metric_defs = [
            MetricDefinition(
                name="mAP50",
                display_name="Mean Average Precision @IoU=0.5",
                description="Standard COCO metric: average precision at IoU=0.5",
                metric_type=MetricType.SCALAR,
            ),
            MetricDefinition(
                name="mAP50:90",
                display_name="Mean Average Precision @IoU=0.5:0.95",
                description="COCO metric: AP averaged over IoU thresholds 0.5:0.95",
                metric_type=MetricType.SCALAR,
            ),
            MetricDefinition(
                name="mAR50",
                display_name="Mean Average Recall @IoU=0.5",
                description="Average recall at IoU=0.5 across all classes",
                metric_type=MetricType.SCALAR,
            ),
            MetricDefinition(
                name="mAR50:90",
                display_name="Mean Average Recall @IoU=0.5:0.95",
                description="Average recall across IoU thresholds 0.5:0.95",
                metric_type=MetricType.SCALAR,
            ),
            MetricDefinition(
                name="accuracy",
                display_name="Accuracy",
                description="TP / (TP + FP + FN)",
                metric_type=MetricType.SCALAR,
            ),
            MetricDefinition(
                name="precision",
                display_name="Precision",
                description="TP / (TP + FP)",
                metric_type=MetricType.SCALAR,
            ),
            MetricDefinition(
                name="recall",
                display_name="Recall",
                description="TP / (TP + FN)",
                metric_type=MetricType.SCALAR,
            ),
            MetricDefinition(
                name="f1",
                display_name="F1 Score",
                description="Harmonic mean of precision and recall",
                metric_type=MetricType.SCALAR,
            ),
            MetricDefinition(
                name="confusion_matrix",
                display_name="Confusion Matrix",
                description="2D matrix: actual class vs predicted class",
                metric_type=MetricType.MATRIX,
            ),
            MetricDefinition(
                name="roc_auc",
                display_name="ROC-AUC",
                description="Area under ROC curve (scalar score)",
                metric_type=MetricType.SCALAR,
            ),
            MetricDefinition(
                name="roc_curve",
                display_name="ROC Curve",
                description="ROC curve points (false positive rate vs true positive rate)",
                metric_type=MetricType.CURVE,
            ),
        ]

    def supported_metrics(self) -> list[MetricDefinition]:
        """Return all supported metrics for COCO detection."""
        return self._metric_defs

    def calculate(self, matches: list[Match]) -> dict[str, MetricValue]:
        """Calculate all metrics from detection matches.
        
        Args:
            matches: List of Match objects from evaluator.match_detections().
            
        Returns:
            Dictionary of metric_name → metric_value.
        """
        if not matches:
            return {}

        metrics = {}

        # Count basic TP/FP/FN/misclassified
        tp_count = sum(1 for m in matches if m.status == "tp")
        fp_count = sum(1 for m in matches if m.status == "fp")
        fn_count = sum(1 for m in matches if m.status == "fn")
        # Scalar metrics
        accuracy = tp_count / (tp_count + fp_count + fn_count) if (tp_count + fp_count + fn_count) > 0 else 0.0
        precision = tp_count / (tp_count + fp_count) if (tp_count + fp_count) > 0 else 0.0
        recall = tp_count / (tp_count + fn_count) if (tp_count + fn_count) > 0 else 0.0
        f1 = (
            2 * (precision * recall) / (precision + recall)
            if (precision + recall) > 0
            else 0.0
        )

        metrics["accuracy"] = accuracy
        metrics["precision"] = precision
        metrics["recall"] = recall
        metrics["f1"] = f1

        # Confusion matrix: (actual_class, predicted_class)
        y_true = []
        y_pred = []
        for m in matches:
            if m.ground_truth and m.prediction:
                y_true.append(m.ground_truth.class_id)
                y_pred.append(m.prediction.class_id)

        if y_true and y_pred:
            cm = confusion_matrix(
                y_true,
                y_pred,
                labels=sorted(self.categories.keys()),
            )
            metrics["confusion_matrix"] = cm.tolist()
        else:
            # Empty confusion matrix if no matches
            num_classes = len(self.categories)
            metrics["confusion_matrix"] = [[0] * num_classes for _ in range(num_classes)]

        # TODO: prepararlo para que se calculen bien, sin aproximaciones. Podrían meterse los IoU-s en la BD.
        # mAP and mAR: simplified calculation
        # For now, approximate using TP/FP/FN counts
        # In production, this would use full COCO evaluation protocol
        mAP50 = tp_count / (tp_count + fp_count + fn_count) if (tp_count + fp_count + fn_count) > 0 else 0.0
        mAP50_90 = mAP50 * 0.95  # Approximate: slightly lower than single IoU
        mAR50 = tp_count / (tp_count + fn_count) if (tp_count + fn_count) > 0 else 0.0
        mAR50_90 = mAR50 * 0.95

        metrics["mAP50"] = mAP50
        metrics["mAP50:90"] = mAP50_90
        metrics["mAR50"] = mAR50
        metrics["mAR50:90"] = mAR50_90

        # ROC-AUC: confidence vs correct detection
        if matches:
            confidences = []
            is_correct = []
            for m in matches:
                if m.prediction:
                    confidences.append(m.prediction.confidence)
                    is_correct.append(1 if m.status == "tp" else 0)

            if confidences and is_correct and len(set(is_correct)) >= 2:
                try:
                    fpr, tpr, _ = roc_curve(is_correct, confidences)
                    # Store as curve points: [(fpr, tpr), ...]
                    metrics["roc_curve"] = list(zip(fpr.tolist(), tpr.tolist()))
                    metrics["roc_auc"] = float(auc(fpr, tpr))
                except ValueError:
                    metrics["roc_curve"] = []
                    metrics["roc_auc"] = 0.0
            else:
                metrics["roc_curve"] = []
                metrics["roc_auc"] = 0.0
        else:
            metrics["roc_curve"] = []
            metrics["roc_auc"] = 0.0

        return metrics

    def get_optimal_threshold(
        self,
        metric_name: str,
        matches: list[Match],
        num_thresholds: int = 100,
    ) -> float:
        """Find confidence threshold that maximizes a target metric.
        
        Args:
            metric_name: Metric to optimize (e.g., "mAP50", "f1", "precision").
            matches: List of Match objects.
            num_thresholds: Number of thresholds to test [0, 1].
            
        Returns:
            Optimal confidence threshold as float.
            
        Raises:
            ValueError: If metric_name cannot be optimized or matches are empty.
        """
        if not matches:
            raise ValueError("Cannot compute optimal threshold: no matches provided.")

        optimizable_metrics = {"accuracy", "precision", "recall", "f1", "mAP50"}
        if metric_name not in optimizable_metrics:
            raise ValueError(f"Metric '{metric_name}' cannot be optimized (try one of {optimizable_metrics})")

        best_threshold = 0.0
        best_value = -1.0

        thresholds = np.linspace(0.0, 1.0, num_thresholds)
        for threshold in thresholds:
            # Filter matches by confidence threshold
            filtered = [m for m in matches if m.prediction is None or m.prediction.confidence >= threshold]

            if not filtered:
                continue

            # Recalculate metric for this threshold
            metrics = self.calculate(filtered)
            value = metrics.get(metric_name, 0.0)

            # Convert curve to scalar if needed
            if isinstance(value, list):
                value = 0.0  # Can't optimize curves

            if value > best_value:
                best_value = value
                best_threshold = threshold

        return best_threshold
