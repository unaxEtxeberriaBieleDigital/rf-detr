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
:meth:`TiledImageSource.process_batch`, since that's the space ``render_result_preview``
re-reads the source file in to produce a preview.

Unlike :class:`~visualizer.backend.semantic_search.sources.default.DefaultImageSource`,
previews here are always a crop around the detection (never the whole image) -- the whole
point of tiling is that the source images are too large to reasonably display in full.
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

    #: Extra context (as a fraction of the detection's own width/height) kept around a
    #: detection when cropping its preview, so the object isn't shown edge-to-edge.
    PREVIEW_MARGIN_RATIO = 0.25

    #: Previews are downscaled (preserving aspect ratio) so neither side exceeds this many
    #: pixels, keeping them light enough to embed as base64 data URLs.
    PREVIEW_MAX_SIZE = 800

    def __init__(self, tile_size: int = 560):
        """Args:
        tile_size: Width/height (in pixels of the *resized* image) of each square tile fed
            to the model. Should match the model's own input resolution (e.g. 512 for
            RFDETRSmall, 560 for RFDETRBase) so tiles aren't internally resized twice.
        """
        self.tile_size = tile_size

    def iter_scan_units(self, folder: Path) -> Iterator[ScanUnit]:
        """Yield one :class:`ScanUnit` per tile of every supported image under *folder*."""
        for path in sorted(folder.rglob("*")):
            if path.is_file() and path.suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS:
                yield from self._iter_image_tiles(path)

    def _iter_image_tiles(self, path: Path) -> Iterator[ScanUnit]:
        """Downscale *path* by :attr:`RESIZE_FACTOR` and split it into a tile grid."""
        image_path = str(path)
        with Image.open(path) as img:
            img = img.convert("RGB")
            orig_w, orig_h = img.size
            resized_w = max(1, round(orig_w * self.RESIZE_FACTOR))
            resized_h = max(1, round(orig_h * self.RESIZE_FACTOR))
            resized = img.resize((resized_w, resized_h), Image.Resampling.LANCZOS)
            resized_arr = np.asarray(resized)

        logger.debug(
            f"Tiling '{image_path}': {orig_w}x{orig_h} -> {resized_w}x{resized_h}, "
            f"tile_size={self.tile_size}"
        )
        for y0 in range(0, resized_h, self.tile_size):
            y1 = min(y0 + self.tile_size, resized_h)
            for x0 in range(0, resized_w, self.tile_size):
                x1 = min(x0 + self.tile_size, resized_w)
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
        on disk (which is what :meth:`render_result_preview` re-reads).
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
        _, _, marker = unit_id.partition(_TILE_MARKER)
        x0, y0, _x1, _y1 = (int(v) for v in marker.split("_"))
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
        """Crop the original source image around *result*'s detection and re-encode it.

        Re-reads only the region of the source file around the bbox (with some margin),
        rescales it to a WebView-friendly size, and JPEG-encodes it in memory -- the full
        (possibly huge) source image is never loaded whole or served/written to disk.
        """
        image_path = Path(result.image_path)
        if not image_path.exists():
            raise FileNotFoundError(f"Image file not found on disk: {image_path}")

        bbox = result.prediction.bbox
        with Image.open(image_path) as img:
            img = img.convert("RGB")
            orig_w, orig_h = img.size

            if bbox is not None:
                x0, y0, x1, y1 = bbox
                margin_x = (x1 - x0) * self.PREVIEW_MARGIN_RATIO
                margin_y = (y1 - y0) * self.PREVIEW_MARGIN_RATIO
                crop_x0 = max(0, int(x0 - margin_x))
                crop_y0 = max(0, int(y0 - margin_y))
                crop_x1 = min(orig_w, int(x1 + margin_x) + 1)
                crop_y1 = min(orig_h, int(y1 + margin_y) + 1)
            else:
                crop_x0, crop_y0, crop_x1, crop_y1 = 0, 0, orig_w, orig_h

            crop = img.crop((crop_x0, crop_y0, crop_x1, crop_y1))
            crop_w, crop_h = crop.size

            scale = 1.0
            if max(crop_w, crop_h) > self.PREVIEW_MAX_SIZE:
                scale = self.PREVIEW_MAX_SIZE / max(crop_w, crop_h)
                crop = crop.resize(
                    (max(1, round(crop_w * scale)), max(1, round(crop_h * scale))),
                    Image.Resampling.LANCZOS,
                )

            buffer = io.BytesIO()
            crop.save(buffer, format="JPEG", quality=90)
            content = buffer.getvalue()

        local_bbox = None
        if bbox is not None:
            local_bbox = (
                (x0 - crop_x0) * scale,
                (y0 - crop_y0) * scale,
                (x1 - crop_x0) * scale,
                (y1 - crop_y0) * scale,
            )
        return SearchResultPreview(content=content, media_type="image/jpeg", bbox=local_bbox)


__all__ = ["TiledImageSource"]
