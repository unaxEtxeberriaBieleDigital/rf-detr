# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

"""Pydantic request and response schemas for the visualizer API."""

from typing import Any

from pydantic import BaseModel


class JobRequest(BaseModel):
    """Parameters for starting dataset inference."""

    dataset_path: str
    dataset_type: str
    model_path: str
    model_type: str
    splits: list[str] | None = None
    batch_size: int = 8
    iou_threshold: float = 0.5
    resume: bool = False


class LoadJobRequest(BaseModel):
    """Parameters for loading persisted dataset inference."""

    dataset_path: str


class JobStatusResponse(BaseModel):
    """Status of a dataset inference job."""

    id: str
    status: str
    error: str | None = None
    num_records: int = 0
    categories: dict[int, str] = {}
    num_images_total: int = 0
    num_images_processed: int = 0
    num_images_remaining: int = 0
    can_resume: bool = False
    has_dimensionality_reduction: bool = False
    dimensionality_reduction_components: int | None = None


class ImagePathPageResponse(BaseModel):
    """Paginated image paths belonging to a job."""

    image_paths: list[str]
    total_images: int
    offset: int
    limit: int
    has_more: bool


class RecordsByImagePathsRequest(BaseModel):
    """Filter for retrieving records by image path."""

    image_paths: list[str]
    split: str | None = None


class CheckDatasetResponse(BaseModel):
    """Summary of persisted dataset inference state."""

    has_db: bool
    num_records: int = 0
    has_dimensionality_reduction: bool = False
    dimensionality_reduction_components: int | None = None
    status: str | None = None
    num_images_total: int = 0
    num_images_processed: int = 0
    num_images_remaining: int = 0
    can_resume: bool = False


class DimensionalityReductionStatusResponse(BaseModel):
    """Result of a dimensionality-reduction request."""

    updated: int
    components: int


class MetricDefinitionResponse(BaseModel):
    """Metric metadata exposed by the API."""

    name: str
    display_name: str
    description: str
    metric_type: str


class EvaluationMetricsResponse(BaseModel):
    """Calculated evaluation metrics."""

    dataset_type: str
    metrics: dict[str, Any]
    metric_definitions: list[MetricDefinitionResponse]
    cached: bool
    calculated_at: str | None = None
    applied_class_thresholds: dict[int, float] | None = None
    applied_record_ids: list[str] | None = None
    applied_record_count: int | None = None


class EvaluationRequest(BaseModel):
    """Optional filters for metric calculation."""

    class_thresholds: dict[int, float] | None = None
    record_ids: list[str] | None = None


class OptimalThresholdResponse(BaseModel):
    """Best threshold found for a metric."""

    dataset_type: str
    metric_name: str
    class_id: int | None = None
    threshold: float
    metric_value: float
    num_thresholds: int


class SemanticSearchRequest(BaseModel):
    """Parameters for starting semantic search."""

    query_record_id: str
    search_path: str
    k: int = 20
    model_path: str
    model_type: str
    source_type: str = "default"


class SemanticSearchResultDTO(BaseModel):
    """A semantic-search result returned by the API."""

    image_path: str
    bbox: tuple[float, float, float, float] | None
    confidence: float
    class_id: int
    distance: float
    preview_data_url: str


class SemanticSearchStatusResponse(BaseModel):
    """Status of a semantic-search job."""

    id: str
    parent_job_id: str
    query_record_id: str
    query_image_path: str
    search_path: str
    k: int
    status: str
    error: str | None = None
    num_images_total: int = 0
    num_images_processed: int = 0
    results: list[SemanticSearchResultDTO] | None = None


class ReductionRequest(BaseModel):
    """Parameters for dimensionality reduction."""

    record_ids: list[str] | None = None
    algorithm: str = "pca"
    perplexity: float = 30.0
    n_neighbors: int = 15
    min_dist: float = 0.1


__all__ = [
    "CheckDatasetResponse",
    "DimensionalityReductionStatusResponse",
    "EvaluationMetricsResponse",
    "EvaluationRequest",
    "ImagePathPageResponse",
    "JobRequest",
    "JobStatusResponse",
    "LoadJobRequest",
    "MetricDefinitionResponse",
    "OptimalThresholdResponse",
    "RecordsByImagePathsRequest",
    "ReductionRequest",
    "SemanticSearchRequest",
    "SemanticSearchResultDTO",
    "SemanticSearchStatusResponse",
]
