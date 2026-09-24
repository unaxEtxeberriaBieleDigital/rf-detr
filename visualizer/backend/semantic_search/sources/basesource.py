# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

"""Pluggable data sources for :mod:`visualizer.backend.semantic_search`.

A "source" decides how a searched folder is turned into inference-ready units (e.g. one unit per image, or one unit per
tile of a large image), how those units are run through the model to produce per-unit detections, and how a result is
turned into a preview the frontend can display. Everything else (batching units, caching, top-k selection across groups,
running in a background thread) is source-agnostic and lives in ``engine.py``.
"""

from abc import ABC, abstractmethod
from collections.abc import Iterator
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from visualizer.backend.datasets.basedataset import SUPPORTED_IMAGE_EXTENSIONS
from visualizer.backend.models.basemodel import BaseModel
from visualizer.backend.semantic_search.types import ScanUnit, SearchResultPreview
from visualizer.backend.shared_types.prediction import Prediction

if TYPE_CHECKING:
    from visualizer.backend.semantic_search.types import SearchResult


def iter_image_files(folder: Path) -> Iterator[Path]:
    """Yield every supported image file under *folder*, recursively, in a stable order.

    Shared by the built-in sources so that counting units and enumerating them walk the
    folder in exactly the same way.

    Args:
        folder: Root folder to walk.

    Yields:
        Paths of the image files found, sorted so repeated scans are deterministic.
    """
    for path in sorted(folder.glob("*")):
        if path.is_file() and path.suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS:
            yield path


class BaseSemanticSearchSource(ABC):
    """Defines how a searched folder is scanned and how results are previewed.

    Implementations should only decide *what* gets fed to the model and *how* a matching result is rendered back to the
    user; everything else (batching, caching, ranking, and running in a background thread) is handled generically by
    ``engine.py``.
    """

    def get_num_units(self, folder: Path, model: BaseModel | None = None) -> int:
        """Count the units :meth:`iter_scan_units` would yield for *folder*.

        The engine calls this once up front so it can report progress without having to
        materialise the whole scan in memory. Implementations must stay consistent with
        :meth:`iter_scan_units` (same *folder* and *model* must give the same count) and
        should skip the expensive part of the work -- e.g. read an image's header to get
        its dimensions rather than decoding its pixels.

        Args:
            folder: The root folder chosen by the user to search.
            model: Model the units will be run through, for sources whose unit count
                depends on it (e.g. a tiled source needs the model's input resolution to
                know how many tiles an image is split into).

        Returns:
            The total number of scan units under *folder*.
        """
        return sum(self.get_num_units_for_group(path, model) for path in self.iter_group_paths(folder))

    def iter_scan_units(self, folder: Path, model: BaseModel | None = None) -> Iterator[ScanUnit]:
        """Enumerate the units of work to run inference on, under *folder*.

        Must be lazy: the engine consumes this incrementally, so that only a batch's worth
        of units is ever held in memory at once.

        Args:
            folder: The root folder chosen by the user to search.
            model: Model the units will be run through, for sources that need it to build
                their units (e.g. a tiled source tiles at the model's input resolution).

        Yields:
            One :class:`ScanUnit` per piece of work (e.g. one per image, or one per tile).
        """
        for path in self.iter_group_paths(folder):
            yield from self.iter_scan_units_for_group(path, model)

    def iter_group_paths(self, folder: Path) -> Iterator[Path]:
        """Yield source-image paths without opening or decoding their contents.

        Args:
            folder: Root folder selected for semantic search.

        Yields:
            Paths used as cache groups.
        """
        yield from iter_image_files(folder)

    @abstractmethod
    def get_num_units_for_group(self, path: Path, model: BaseModel | None = None) -> int:
        """Count scan units for one source image.

        This method is only called for an uncached or changed image, so implementations
        may inspect its header when the unit count depends on image dimensions.

        Args:
            path: Source image path.
            model: Model whose input properties may determine the unit count.

        Returns:
            Number of scan units generated for the image.
        """
        raise NotImplementedError

    @abstractmethod
    def iter_scan_units_for_group(self, path: Path, model: BaseModel | None = None) -> Iterator[ScanUnit]:
        """Yield inference units for one uncached or changed source image.

        Args:
            path: Source image path.
            model: Model used to configure source-specific processing.

        Yields:
            Scan units belonging to ``path``.
        """
        raise NotImplementedError

    def cache_signature(self, model: BaseModel | None = None) -> str:
        """Return a stable identity for cache-affecting source configuration.

        Args:
            model: Model whose input properties may affect generated units.

        Returns:
            Stable source configuration signature.
        """
        return f"{type(self).__module__}.{type(self).__qualname__}:v1"

    def process_batch(self, model: "BaseModel", batch: list[ScanUnit]) -> list[list[tuple[Prediction, list[float]]]]:
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
        raise NotImplementedError


__all__ = ["BaseSemanticSearchSource", "ScanUnit", "SearchResultPreview", "iter_image_files"]
