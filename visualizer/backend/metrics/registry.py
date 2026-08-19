# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

from visualizer.backend.metrics.base_metrics import MetricDefinition, MetricType


# Global registry of all metrics that any dataset can compute
AVAILABLE_METRICS: dict[str, MetricDefinition] = {
    "mAP50": MetricDefinition(
        name="mAP50",
        display_name="Mean Average Precision @IoU=0.5",
        description="Standard COCO metric: average precision at IoU threshold 0.5",
        metric_type=MetricType.SCALAR,
    ),
    "mAP50:90": MetricDefinition(
        name="mAP50:90",
        display_name="Mean Average Precision @IoU=0.5:0.95",
        description="COCO metric: average precision across IoU thresholds from 0.5 to 0.95",
        metric_type=MetricType.SCALAR,
    ),
    "mAR50": MetricDefinition(
        name="mAR50",
        display_name="Mean Average Recall @IoU=0.5",
        description="Recall metric at IoU threshold 0.5",
        metric_type=MetricType.SCALAR,
    ),
    "mAR50:90": MetricDefinition(
        name="mAR50:90",
        display_name="Mean Average Recall @IoU=0.5:0.95",
        description="Recall metric across IoU thresholds from 0.5 to 0.95",
        metric_type=MetricType.SCALAR,
    ),
    "roc_auc": MetricDefinition(
        name="roc_auc",
        display_name="ROC-AUC",
        description="Area under the Receiver Operating Characteristic curve",
        metric_type=MetricType.SCALAR,
    ),
    "roc_curve": MetricDefinition(
        name="roc_curve",
        display_name="ROC Curve",
        description="ROC curve points as (false positive rate, true positive rate)",
        metric_type=MetricType.CURVE,
    ),
    "accuracy": MetricDefinition(
        name="accuracy",
        display_name="Accuracy",
        description="Overall detection accuracy (TP / (TP + FP + FN))",
        metric_type=MetricType.SCALAR,
    ),
    "precision": MetricDefinition(
        name="precision",
        display_name="Precision",
        description="Precision: TP / (TP + FP)",
        metric_type=MetricType.SCALAR,
    ),
    "recall": MetricDefinition(
        name="recall",
        display_name="Recall",
        description="Recall: TP / (TP + FN)",
        metric_type=MetricType.SCALAR,
    ),
    "f1": MetricDefinition(
        name="f1",
        display_name="F1 Score",
        description="Harmonic mean of precision and recall",
        metric_type=MetricType.SCALAR,
    ),
    "confusion_matrix": MetricDefinition(
        name="confusion_matrix",
        display_name="Confusion Matrix",
        description="2D matrix showing True Positives, False Positives, False Negatives per class",
        metric_type=MetricType.MATRIX,
    ),
}


# Maps dataset types to which metrics they support
# Format: dataset_type_string → list of metric names
DATASET_METRICS_MAPPING: dict[str, list[str]] = {
    "coco_detection": [
        "mAP50",
        "mAP50:90",
        "mAR50",
        "mAR50:90",
        "accuracy",
        "precision",
        "recall",
        "f1",
        "confusion_matrix",
        "roc_auc",
        "roc_curve",
    ],
}


def get_metrics_for_dataset(dataset_type: str) -> list[MetricDefinition]:
    """Get list of metric definitions for a specific dataset type.
    
    Args:
        dataset_type: String identifier of dataset (e.g., "coco_detection").
        
    Returns:
        List of MetricDefinition objects for that dataset type.
        Empty list if dataset type is unknown.
    """
    metric_names = DATASET_METRICS_MAPPING.get(dataset_type, [])
    return [AVAILABLE_METRICS[name] for name in metric_names if name in AVAILABLE_METRICS]
