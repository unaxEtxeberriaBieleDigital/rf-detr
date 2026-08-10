# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

"""Default :class:`BaseSemanticSearchSource`: one scan unit per plain image file.

Mirrors the original (pre-refactor) behaviour of ``semantic_search.py``: every supported
image file found recursively under the search folder is its own unit of work, batches are
run through the model with no source-specific post-processing (using
``BaseSemanticSearchSource.process_batch``'s default implementation), and a result is
previewed by serving the original image file straight off disk.
"""

from collections.abc import Iterator
from pathlib import Path

from rfdetr.utilities.logger import get_logger
from visualizer.backend.datasets.basedataset import SUPPORTED_IMAGE_EXTENSIONS
from visualizer.backend.registry import register_semantic_search_source
from visualizer.backend.semantic_search.basesource import (
    BaseSemanticSearchSource,
    ScanUnit,
    SearchResultPreview,
)

logger = get_logger()

# Maps common image suffixes to their HTTP media type, for serving files as-is.
_MEDIA_TYPES = {
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".png": "image/png",
    ".bmp": "image/bmp",
    ".tif": "image/tiff",
    ".tiff": "image/tiff",
    ".webp": "image/webp",
}


@register_semantic_search_source("default")
class DefaultImageSource(BaseSemanticSearchSource):
    """Treats every image file under the search folder as one scan unit.

    ``group_key`` and ``id`` are both the image's own path, so deduplication behaves as
    "one result per image", exactly as before this source-based refactor.
    """

    def iter_scan_units(self, folder: Path) -> Iterator[ScanUnit]:
        """Yield one :class:`ScanUnit` per supported image file under *folder*."""
        for path in sorted(folder.rglob("*")):
            if path.is_file() and path.suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS:
                image_path = str(path)
                yield ScanUnit(id=image_path, group_key=image_path, inference_input=path)

    def render_result_preview(self, result) -> SearchResultPreview:
        """Return the raw bytes of the original image file referenced by *result*.

        The whole image is served as-is (no crop/rescale), so the detection's bbox is
        already in the right coordinate space and is passed through unchanged.
        """
        image_path = Path(result.image_path)
        if not image_path.exists():
            raise FileNotFoundError(f"Image file not found on disk: {image_path}")
        media_type = _MEDIA_TYPES.get(image_path.suffix.lower(), "application/octet-stream")
        return SearchResultPreview(
            content=image_path.read_bytes(),
            media_type=media_type,
            bbox=result.prediction.bbox,
        )


__all__ = ["DefaultImageSource"]
