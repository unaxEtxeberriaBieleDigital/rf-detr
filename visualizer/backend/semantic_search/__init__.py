# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

"""Nearest-neighbour semantic search over an arbitrary folder of images.

Public API re-exported here for convenience:
    * :class:`~visualizer.backend.semantic_search.types.SearchJob`
    * :class:`~visualizer.backend.semantic_search.types.SearchResult`
    * :data:`~visualizer.backend.semantic_search.engine.SEARCH_JOB_STORE`
    * :func:`~visualizer.backend.semantic_search.engine.run_semantic_search`
    * :class:`~visualizer.backend.semantic_search.sources.basesource.BaseSemanticSearchSource`
    * :class:`~visualizer.backend.semantic_search.types.ScanUnit`
"""

from visualizer.backend.semantic_search.sources.basesource import BaseSemanticSearchSource
from visualizer.backend.semantic_search.engine import (
    SEARCH_JOB_STORE,
    SearchJob,
    SearchResult,
    run_semantic_search,
)
from visualizer.backend.semantic_search.types import ScanUnit

__all__ = [
    "SearchJob",
    "SearchResult",
    "SEARCH_JOB_STORE",
    "run_semantic_search",
    "BaseSemanticSearchSource",
    "ScanUnit",
]
