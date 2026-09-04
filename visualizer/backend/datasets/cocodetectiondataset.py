# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

import os
import json
from collections.abc import Iterator
from pathlib import Path

from visualizer.backend.datasets.basedataset import *
from visualizer.backend.shared_types.prediction import Prediction
from visualizer.backend.registry import register_dataset
from visualizer.backend.metrics import MetricsCalculator
from visualizer.backend.metrics.coco_detection_metrics import COCODetectionMetricsCalculator


@register_dataset("coco_detection")
class COCODetectionDataset(BaseDataset):
    """COCO-format object detection dataset.

    Assumes that, for every split folder (``train``/``test``/``val``), the images and the ``_annotations.coco.json``
    file describing their ground truth live side by side in that same folder.
    """

    task_type = "detection"

    def __init__(self, dataset_path: str | Path):
        super().__init__(dataset_path, DatasetType.COCO_DETECTION)
        self._annotations_by_split: dict[Split, dict[str, list[Prediction]]] = {
            split: self._build_annotation_index(split_path) for split, split_path in self.splits.items()
        }
        # self._sanity_check()

    def iter_split(self, split: Split) -> Iterator[Path]:
        yield from super().iter_split(split)

    def iter_batches(self, split: Split, batch_size: int) -> Iterator[list[Path]]:
        yield from super().iter_batches(split, batch_size)

    def get_ground_truth(self, image_path: Path) -> list[Prediction]:
        split = self._split_of(image_path)
        split_path = self.splits[split].resolve()
        image_resolved = Path(image_path).resolve()
        lookup_keys = {self._normalize_lookup_key(image_resolved.name)}

        try:
            relative = image_resolved.relative_to(split_path)
            lookup_keys.add(self._normalize_lookup_key(relative.as_posix()))
        except ValueError:
            pass

        for key in lookup_keys:
            if key in self._annotations_by_split[split]:
                return self._annotations_by_split[split][key]
        return []

    def _split_of(self, image_path: Path) -> Split:
        parent = Path(image_path).resolve().parent
        for split, split_path in self.splits.items():
            if split_path.resolve() == parent:
                return split
        raise ValueError(f"Image does not belong to any known split of this dataset: {image_path}")

    def _build_annotation_index(self, split_path: Path) -> dict[str, list[Prediction]]:
        annotations_file = split_path / "_annotations.coco.json"
        if not annotations_file.exists():
            raise FileNotFoundError(
                f"Expected COCO annotations at {annotations_file}: images and their "
                "'_annotations.coco.json' must live in the same folder."
            )

        with open(annotations_file, "r", encoding="utf-8") as f:
            coco = json.load(f)

        file_name_by_image_id: dict[int, str] = {image["id"]: image["file_name"] for image in coco["images"]}
        index: dict[str, list[Prediction]] = {}
        for file_name in file_name_by_image_id.values():
            for key in self._annotation_lookup_keys(file_name):
                index.setdefault(key, [])

        for annotation in coco["annotations"]:
            file_name = file_name_by_image_id[annotation["image_id"]]
            x, y, w, h = annotation["bbox"]
            prediction = Prediction(
                class_id=annotation["category_id"],
                confidence=1.0,
                bbox=(x, y, x + w, y + h),
            )
            for key in self._annotation_lookup_keys(file_name):
                index.setdefault(key, []).append(prediction)

        return index

    def _sanity_check(self):
        for split_path in self.splits.values():
            self._split_sanity_check(split_path)

    def _split_sanity_check(self, path: Path):
        with open(path / "_annotations.coco.json", mode="r") as f:
            anns = json.load(f)
        images_in_dir = [file for file in os.listdir() if file in SUPPORTED_IMAGE_EXTENSIONS]
        num_images_in_dir = len(images_in_dir)
        num_images_in_anns = len(anns["images"])
        if num_images_in_anns != num_images_in_dir: raise Exception(f"Unappropriate dataset: diferent number of images in dir ({num_images_in_dir}) and annoattions ({num_images_in_anns})")

    @staticmethod
    def _normalize_lookup_key(value: str) -> str:
        return value.replace("\\", "/").lstrip("./").lower()

    @classmethod
    def _annotation_lookup_keys(cls, file_name: str) -> set[str]:
        normalized = cls._normalize_lookup_key(file_name)
        keys = {normalized}
        keys.add(cls._normalize_lookup_key(Path(normalized).name))
        parts = Path(normalized).parts
        if len(parts) >= 2:
            keys.add(cls._normalize_lookup_key(Path(*parts[1:]).as_posix()))
        return keys

    def get_metrics_calculator(self) -> MetricsCalculator:
        """Get the metrics calculator for this COCO detection dataset.
        
        Returns:
            COCODetectionMetricsCalculator instance configured with this dataset's categories.
        """
        return COCODetectionMetricsCalculator(self.categories)
 