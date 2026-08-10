# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

"""Tiled :class:`BaseSemanticSearchSource`, for images too large to run inference on whole.

Each source image is first downscaled to a third of its original size (in memory, never
written to disk), then chopped into a grid of non-overlapping tiles matching the model's
input resolution; each tile becomes its own :class:`ScanUnit`. Detections are translated
back from tile-local pixel coordinates into the *original* image's coordinate space in
:meth:`TiledImageSource.process_batch`, so the resulting bbox always means something
against the full-resolution source file on disk (e.g. for downstream consumers other than
the preview itself).

Unlike :class:`~visualizer.backend.semantic_search.sources.default.DefaultImageSource`,
``render_result_preview`` here never serves the (possibly huge) source image, nor an
arbitrary crop around the detection: it re-renders *the exact tile that was fed to the
model* when the result's embedding was computed (recomputing the same downscale + crop
from ``result.unit_id``), so what the user sees is precisely what was scored.
"""

import io
from collections.abc import Iterator
from pathlib import Path

import numpy as np
from PIL import Image

from rfdetr.utilities.logger import get_logger
from visualizer.backend.datasets.basedataset import SUPPORTED_IMAGE_EXTENSIONS
from visualizer.backend.models.basemodel import BaseModel
from visualizer.backend.prediction import Prediction
from visualizer.backend.registry import register_semantic_search_source
from visualizer.backend.semantic_search.basesource import (
    BaseSemanticSearchSource,
    ScanUnit,
    SearchResultPreview,
)
from visualizer.backend.semantic_search.engine import SearchResult

logger = get_logger()

# Marks the offsets of a tile within the *resized* image (see module docstring) inside its
# ScanUnit.id, so process_batch can translate a detection's tile-local bbox back into the
# resized image's coordinate space without keeping any extra in-memory state -- the id
# alone is enough to reconstruct it, which also keeps cache hits (loaded on a later run,
# possibly by a fresh TiledImageSource instance) translatable in exactly the same way.
_TILE_MARKER = "::tile_"


@register_semantic_search_source("tiled")
class TiledImageSource(BaseSemanticSearchSource):
    """Scans huge images as a grid of tiles, run through the model at its own resolution.

    ``group_key`` is the source image's own path (one result per source image, no matter
    how many tiles it was split into); ``id`` additionally encodes the tile's pixel bounds
    within the resized image, e.g. ``"img.tif::tile_0_0_560_560"``.
    """

    #: Source images are downscaled to this fraction of their original size before tiling.
    RESIZE_FACTOR = 1.0 / 3.0

    #: Previews are downscaled (preserving aspect ratio) so neither side exceeds this many
    #: pixels, keeping them light enough to embed as base64 data URLs.
    PREVIEW_MAX_SIZE = 800

    def iter_scan_units(self, folder: Path, model: BaseModel | None = None) -> Iterator[ScanUnit]:
        """Yield one :class:`ScanUnit` per tile of every supported image under *folder*."""
        for path in sorted(folder.rglob("*")):
            if path.is_file() and path.suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS:
                yield from self._iter_image_tiles(path, model)

    def _iter_image_tiles(self, path: Path, model: BaseModel | None = None) -> Iterator[ScanUnit]:
        """Downscale *path* by :attr:`RESIZE_FACTOR` and split it into a tile grid."""
        image_path = str(path)
        with Image.open(path) as img:
            img = img.convert("RGB")
            orig_w, orig_h = img.size
            resized_w = max(1, round(orig_w * self.RESIZE_FACTOR))
            resized_h = max(1, round(orig_h * self.RESIZE_FACTOR))
            resized = img.resize((resized_w, resized_h), Image.Resampling.LANCZOS)
            resized_arr = np.asarray(resized)

        if model is None: raise Exception("Se ha intentado procesar un TiledImageSource sin modelo: no se conoce el tile size")
        if not hasattr(model, "input_shape"): raise Exception("Se ha intentado procesar un TiledImageSource con un modelo sin input_shape: no se conoce el tile size")

        logger.debug(
            f"Tiling '{image_path}': {orig_w}x{orig_h} -> {resized_w}x{resized_h}, "
            f"tile_size={model.input_shape}"
        )
        for y0 in range(0, resized_h, model.input_shape):
            y1 = min(y0 + model.input_shape, resized_h)
            for x0 in range(0, resized_w, model.input_shape):
                x1 = min(x0 + model.input_shape, resized_w)
                tile = resized_arr[y0:y1, x0:x1]
                unit_id = f"{image_path}{_TILE_MARKER}{x0}_{y0}_{x1}_{y1}"
                yield ScanUnit(id=unit_id, group_key=image_path, inference_input=tile)

    def process_batch(
        self, model: BaseModel, batch: list[ScanUnit]
    ) -> list[list[tuple[Prediction, list[float]]]]:
        """Run *model* over a batch of tiles, translating bboxes back to source-image space.

        Overrides the default implementation because only this source knows each tile's
        offset within its resized source image (and the resize factor applied), needed to
        turn a tile-local bbox into a bbox that means something against the original file
        on disk.
        """
        inputs = [unit.inference_input for unit in batch]
        embeddings, predictions = model.get_batch_embeddings(inputs)

        batch_detections: list[list[tuple[Prediction, list[float]]]] = []
        for unit, unit_embeddings, unit_predictions in zip(batch, embeddings, predictions):
            tile_x0, tile_y0 = self._parse_tile_offset(unit.id)
            detections: list[tuple[Prediction, list[float]]] = []
            if unit_embeddings is not None and len(unit_embeddings) > 0:
                vecs = unit_embeddings.detach().cpu().numpy().astype(np.float32)
                for vec, pred in zip(vecs, unit_predictions):
                    translated = Prediction(
                        class_id=pred.class_id,
                        confidence=pred.confidence,
                        bbox=self._tile_bbox_to_source(pred.bbox, tile_x0, tile_y0),
                    )
                    detections.append((translated, vec.tolist()))
            batch_detections.append(detections)
        return batch_detections

    def _parse_tile_offset(self, unit_id: str) -> tuple[int, int]:
        """Recover a tile's ``(x0, y0)`` offset within its resized source image from its id."""
        x0, y0, _x1, _y1 = self._parse_tile_bounds(unit_id)
        return x0, y0

    def _tile_bbox_to_source(
        self,
        bbox: tuple[float, float, float, float] | None,
        tile_x0: int,
        tile_y0: int,
    ) -> tuple[float, float, float, float] | None:
        """Translate a tile-local bbox into the coordinate space of the original file."""
        if bbox is None:
            return None
        lx0, ly0, lx1, ly1 = bbox
        # Tile-local -> resized-image coordinates (shift by the tile's own offset)...
        rx0, ry0, rx1, ry1 = tile_x0 + lx0, tile_y0 + ly0, tile_x0 + lx1, tile_y0 + ly1
        # ...then resized-image -> original-image coordinates (undo RESIZE_FACTOR).
        return tuple(v / self.RESIZE_FACTOR for v in (rx0, ry0, rx1, ry1))

    def render_result_preview(self, result: SearchResult) -> SearchResultPreview:
        """Re-render *the exact tile* that was fed to the model for *result*'s embedding.

        Recomputes the same downscale (:attr:`RESIZE_FACTOR`) applied in
        :meth:`_iter_image_tiles` and crops out the tile bounds encoded in
        ``result.unit_id`` -- i.e. exactly the pixels the model actually scored -- instead
        of an arbitrary crop around the (translated) detection bbox. Only that small region
        of the source file is ever loaded into memory; the full (possibly huge) source
        image is never read whole or written/served from disk.
        """
        image_path = Path(result.image_path)
        if not image_path.exists():
            raise FileNotFoundError(f"Image file not found on disk: {image_path}")

        tile_x0, tile_y0, tile_x1, tile_y1 = self._parse_tile_bounds(result.unit_id)

        with Image.open(image_path) as img:
            img = img.convert("RGB")
            orig_w, orig_h = img.size
            resized_w = max(1, round(orig_w * self.RESIZE_FACTOR))
            resized_h = max(1, round(orig_h * self.RESIZE_FACTOR))
            resized = img.resize((resized_w, resized_h), Image.Resampling.LANCZOS)
            tile = resized.crop((tile_x0, tile_y0, tile_x1, tile_y1))
            tile_w, tile_h = tile.size

            scale = 1.0
            if max(tile_w, tile_h) > self.PREVIEW_MAX_SIZE:
                scale = self.PREVIEW_MAX_SIZE / max(tile_w, tile_h)
                tile = tile.resize(
                    (max(1, round(tile_w * scale)), max(1, round(tile_h * scale))),
                    Image.Resampling.LANCZOS,
                )

            buffer = io.BytesIO()
            tile.save(buffer, format="JPEG", quality=90)
            content = buffer.getvalue()

        # result.prediction.bbox is already in original-image coordinates (see
        # process_batch/_tile_bbox_to_source); translate it back into this tile's own local
        # coordinates: original -> resized (apply RESIZE_FACTOR), then resized -> tile-local
        # (shift by the tile's own offset), then apply the same preview downscale as above.
        local_bbox = None
        bbox = result.prediction.bbox
        if bbox is not None:
            rx0, ry0, rx1, ry1 = (v * self.RESIZE_FACTOR for v in bbox)
            local_bbox = (
                (rx0 - tile_x0) * scale,
                (ry0 - tile_y0) * scale,
                (rx1 - tile_x0) * scale,
                (ry1 - tile_y0) * scale,
            )
        return SearchResultPreview(content=content, media_type="image/jpeg", bbox=local_bbox)

    def _parse_tile_bounds(self, unit_id: str) -> tuple[int, int, int, int]:
        """Recover a tile's ``(x0, y0, x1, y1)`` bounds within its resized source image."""
        _, _, marker = unit_id.partition(_TILE_MARKER)
        x0, y0, x1, y1 = (int(v) for v in marker.split("_"))
        return x0, y0, x1, y1


__all__ = ["TiledImageSource"]
