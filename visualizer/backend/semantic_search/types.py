# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

"""Data types used by semantic search."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import numpy as np

from visualizer.backend.shared_types.prediction import Prediction

SearchStatus = Literal["pending", "running", "done", "error", "cancelled"]


@dataclass
class ScanUnit:
    """One inference-ready unit produced by a semantic-search source."""

    id: str
    group_key: str
    inference_input: str | Path | np.ndarray


@dataclass
class SearchResultPreview:
    """In-memory preview rendered for one semantic-search result."""

    content: bytes
    media_type: str
    bbox: tuple[float, float, float, float] | None


@dataclass
class SearchResult:
    """One nearest-neighbour hit within a result group."""

    image_path: str
    prediction: Prediction
    distance: float
    unit_id: str


@dataclass
class SearchJob:
    """State for one semantic-search run."""

    id: str
    parent_job_id: str
    query_record_id: str
    query_image_path: str
    search_path: str
    k: int
    source_type: str = "default"
    status: SearchStatus = "pending"
    error: str | None = None
    num_images_total: int = 0
    num_images_processed: int = 0
    results: list[SearchResult] = field(default_factory=list)


__all__ = ["ScanUnit", "SearchResultPreview", "SearchResult", "SearchJob", "SearchStatus"]
