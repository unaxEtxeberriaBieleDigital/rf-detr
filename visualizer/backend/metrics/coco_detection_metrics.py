# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

import numpy as np
from sklearn.metrics import auc, confusion_matrix, roc_curve

from visualizer.backend.evaluation.types import Match
from visualizer.backend.metrics.base_metrics import (
    MetricDefinition,
    MetricsCalculator,
    MetricType,
    MetricValue,
)

# COCO evaluates AP/AR over IoU thresholds 0.50:0.05:0.95.
_COCO_IOU_THRESHOLDS = np.linspace(0.5, 0.95, 10)

# COCO interpolates the precision/recall curve over 101 evenly spaced recall levels.
_COCO_RECALL_LEVELS = np.linspace(0.0, 1.0, 101)

# Records written before the IoU column existed only know they cleared the matching
# threshold, so they are scored at that threshold and no higher.
_LEGACY_IOU = 0.5


# TODO: Quitar referencias legacy
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

    @staticmethod
    def _apply_class_thresholds(
        matches: list[Match],
        class_thresholds: dict[int, float] | None = None,
    ) -> list[Match]:
        """Apply per-class confidence thresholds to prediction-backed matches.

        Predictions below their configured threshold are removed. When a removed prediction had an attached ground
        truth, the ground truth is preserved as a false negative so recall-sensitive metrics remain accurate.
        """
        if not class_thresholds:
            return list(matches)

        filtered_matches: list[Match] = []
        for match in matches:
            if match.prediction is None:
                filtered_matches.append(match)
                continue

            threshold = class_thresholds.get(match.prediction.class_id)
            if threshold is None or match.prediction.confidence >= threshold:
                filtered_matches.append(match)
                continue

            if match.ground_truth is not None:
                filtered_matches.append(
                    Match(
                        prediction=None,
                        embedding=None,
                        ground_truth=match.ground_truth,
                        status="fn",
                    )
                )

        return filtered_matches

    @staticmethod
    def _filter_matches_for_predicted_class(matches: list[Match], class_id: int) -> list[Match]:
        """Return predictions and false negatives belonging to one class."""
        return [
            match
            for match in matches
            if (match.prediction is not None and match.prediction.class_id == class_id)
            or (match.prediction is None and match.ground_truth is not None and match.ground_truth.class_id == class_id)
        ]

    @staticmethod
    def _average_precision(is_true_positive: np.ndarray, num_ground_truths: int) -> float:
        """Return 101-point interpolated average precision for one class.

        Args:
            is_true_positive: Binary flags for detections already sorted by descending
                confidence, where 1 marks a true positive.
            num_ground_truths: Number of ground-truth boxes for the class.

        Returns:
            Average precision in ``[0, 1]``.
        """
        if num_ground_truths == 0 or is_true_positive.size == 0:
            return 0.0

        cumulative_true_positives = np.cumsum(is_true_positive)
        cumulative_detections = np.arange(1, is_true_positive.size + 1)
        precisions = cumulative_true_positives / cumulative_detections
        recalls = cumulative_true_positives / num_ground_truths

        # Make precision monotonically decreasing so each recall level uses the best
        # precision achievable at or beyond it.
        precisions = np.maximum.accumulate(precisions[::-1])[::-1]

        interpolated = np.zeros_like(_COCO_RECALL_LEVELS)
        insert_positions = np.searchsorted(recalls, _COCO_RECALL_LEVELS, side="left")
        valid = insert_positions < precisions.size
        interpolated[valid] = precisions[insert_positions[valid]]
        return float(interpolated.mean())

    def _mean_average_precision_recall(
        self,
        matches: list[Match],
        iou_threshold: float,
    ) -> tuple[float, float]:
        """Return (mAP, mAR) across classes at one IoU threshold.

        A detection counts as a true positive when it was paired with a ground truth of
        the same class and that pairing's IoU clears *iou_threshold*. Because the
        evaluator already enforces one-to-one greedy assignment, raising the threshold
        only removes pairings, so ground-truth boxes are never double-counted.

        Args:
            matches: Detection matches to score.
            iou_threshold: Minimum IoU for a pairing to count as a true positive.

        Returns:
            Tuple of ``(mean_average_precision, mean_average_recall)``.
        """
        average_precisions: list[float] = []
        average_recalls: list[float] = []

        for class_id in sorted(self.categories.keys()):
            num_ground_truths = sum(
                1 for match in matches if match.ground_truth is not None and match.ground_truth.class_id == class_id
            )
            if num_ground_truths == 0:
                continue

            class_detections = [
                match for match in matches if match.prediction is not None and match.prediction.class_id == class_id
            ]
            class_detections.sort(
                key=lambda match: match.prediction.confidence,  # type: ignore[union-attr]
                reverse=True,
            )

            flags = np.array(
                [
                    1
                    if (
                        match.ground_truth is not None
                        and match.ground_truth.class_id == class_id
                        and (_LEGACY_IOU if match.iou is None else match.iou) >= iou_threshold
                    )
                    else 0
                    for match in class_detections
                ],
                dtype=np.int64,
            )

            average_precisions.append(self._average_precision(flags, num_ground_truths))
            average_recalls.append(float(flags.sum()) / num_ground_truths)

        if not average_precisions:
            return 0.0, 0.0
        return float(np.mean(average_precisions)), float(np.mean(average_recalls))

    def calculate(
        self,
        matches: list[Match],
        class_thresholds: dict[int, float] | None = None,
    ) -> dict[str, MetricValue]:
        """Calculate all metrics from detection matches.

        Args:
            matches: List of Match objects from evaluator.match_detections().
            class_thresholds: Optional mapping of predicted class id to
                confidence threshold.

        Returns:
            Dictionary of metric_name → metric_value.
        """
        matches = self._apply_class_thresholds(matches, class_thresholds)
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
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

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

        # mAP / mAR from the per-record IoU persisted by the evaluator, following the
        # COCO protocol (per-class AP with 101-point interpolation, averaged over classes
        # and over the IoU thresholds 0.50:0.05:0.95).
        precision_by_threshold: list[float] = []
        recall_by_threshold: list[float] = []
        for iou_threshold in _COCO_IOU_THRESHOLDS:
            mean_ap, mean_ar = self._mean_average_precision_recall(matches, float(iou_threshold))
            precision_by_threshold.append(mean_ap)
            recall_by_threshold.append(mean_ar)

        metrics["mAP50"] = precision_by_threshold[0]
        metrics["mAP50:90"] = float(np.mean(precision_by_threshold))
        metrics["mAR50"] = recall_by_threshold[0]
        metrics["mAR50:90"] = float(np.mean(recall_by_threshold))

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
                    false_positive_rate, true_positive_rate, _ = roc_curve(is_correct, confidences)
                    # Store as curve points: [(false_positive_rate, true_positive_rate), ...]
                    metrics["roc_curve"] = list(
                        zip(
                            false_positive_rate.tolist(),
                            true_positive_rate.tolist(),
                        )
                    )
                    metrics["roc_auc"] = float(auc(false_positive_rate, true_positive_rate))
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
        class_id: int | None = None,
    ) -> float:
        """Find confidence threshold that maximizes a target metric.

        Args:
            metric_name: Metric to optimize (e.g., "mAP50", "f1", "precision").
            matches: List of Match objects.
            num_thresholds: Number of thresholds to test [0, 1].
            class_id: Optional predicted class id to optimize. When provided,
                only predictions of that class are thresholded, while false
                negatives remain in scope.

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

        candidate_matches = (
            self._filter_matches_for_predicted_class(matches, class_id) if class_id is not None else list(matches)
        )
        if not candidate_matches:
            raise ValueError("Cannot compute optimal threshold: no relevant matches provided.")

        best_threshold = 0.0
        best_value = -1.0

        thresholds = np.linspace(0.0, 1.0, num_thresholds)
        for threshold in thresholds:
            if class_id is None:
                class_thresholds = {
                    match.prediction.class_id: float(threshold)
                    for match in candidate_matches
                    if match.prediction is not None
                }
            else:
                class_thresholds = {class_id: float(threshold)}
            filtered = self._apply_class_thresholds(candidate_matches, class_thresholds)

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
