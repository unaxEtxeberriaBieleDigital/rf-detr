# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

"""Built-in :class:`~visualizer.backend.semantic_search.basesource.BaseSemanticSearchSource`
implementations.

Importing this package registers all built-in sources into
``SEMANTIC_SEARCH_SOURCE_REGISTRY`` (see ``visualizer.backend.registry``).
"""

from visualizer.backend.semantic_search.sources.default import DefaultImageSource
from visualizer.backend.semantic_search.sources.tiled import TiledImageSource

__all__ = ["DefaultImageSource", "TiledImageSource"]
