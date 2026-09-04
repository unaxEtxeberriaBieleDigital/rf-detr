# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

"""Pluggable data sources for :mod:`visualizer.backend.semantic_search`.

A "source" decides how a searched folder is turned into inference-ready units (e.g. one
unit per image, or one unit per tile of a large image), how those units are run through the
model to produce per-unit detections, and how a result is turned into a preview the
frontend can display. Everything else (batching units, caching, top-k selection across
groups, running in a background thread) is source-agnostic and lives in ``engine.py``.
"""

from abc import ABC, abstractmethod
from collections.abc import Iterator
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from visualizer.backend.models.basemodel import BaseModel
from visualizer.backend.shared_types.prediction import Prediction
from visualizer.backend.semantic_search.types import ScanUnit, SearchResultPreview

if TYPE_CHECKING:
    from visualizer.backend.semantic_search.types import SearchResult


class BaseSemanticSearchSource(ABC):
    """Defines how a searched folder is scanned and how results are previewed.

    Implementations should only decide *what* gets fed to the model and *how* a matching
    result is rendered back to the user; everything else (batching, caching, ranking, and
    running in a background thread) is handled generically by ``engine.py``.
    """

    @abstractmethod
    def iter_scan_units(self, folder: Path, model: BaseModel | None = None) -> Iterator[ScanUnit]:
        """Enumerate the units of work to run inference on, under *folder*.

        Args:
            folder: The root folder chosen by the user to search.

        Yields:
            One :class:`ScanUnit` per piece of work (e.g. one per image, or one per tile).
        """

    def process_batch(
        self, model: "BaseModel", batch: list[ScanUnit]
    ) -> list[list[tuple[Prediction, list[float]]]]:
        """Run *model* over one batch of scan units and return per-unit detections.

        The default implementation simply forwards ``unit.inference_input`` for every unit
        in *batch* to :meth:`BaseModel.get_batch_embeddings` and pairs the returned
        embeddings with their aligned predictions. Override this to post-process detections
        in a source-specific way -- e.g. a tiled source would translate each detection's
        tile-local bbox back into the coordinate space of its source image here, since only
        the source knows the tile's offset within that image.

        Args:
            model: Model used to extract embeddings/predictions for the batch.
            batch: The scan units to run inference on (already known not to be cached).

        Returns:
            One list of ``(prediction, embedding)`` pairs per unit in *batch*, aligned 1:1
            and in the same order (may be an empty list for a unit with no detections).
        """
        inputs = [unit.inference_input for unit in batch]
        embeddings, predictions = model.get_batch_embeddings(inputs)

        batch_detections: list[list[tuple[Prediction, list[float]]]] = []
        for unit_embeddings, unit_predictions in zip(embeddings, predictions):
            detections: list[tuple[Prediction, list[float]]] = []
            if unit_embeddings is not None and len(unit_embeddings) > 0:
                vecs = unit_embeddings.detach().cpu().numpy().astype(np.float32)
                detections = [(pred, vec.tolist()) for vec, pred in zip(vecs, unit_predictions)]
            batch_detections.append(detections)
        return batch_detections

    @abstractmethod
    def render_result_preview(self, result: "SearchResult") -> SearchResultPreview:
        """Render a preview for *result*, entirely in memory (never touching disk).

        Args:
            result: The search result to render a preview for.

        Returns:
            A :class:`SearchResultPreview` with the preview bytes/media type and the
            detection's bbox translated into that preview's local coordinate space.
        """


__all__ = ["BaseSemanticSearchSource", "ScanUnit", "SearchResultPreview"]
