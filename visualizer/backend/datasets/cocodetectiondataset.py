# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

import json
from collections.abc import Iterator
from pathlib import Path

from visualizer.backend.datasets.basedataset import BaseDataset, DatasetType, Split
from visualizer.backend.prediction import Prediction
from visualizer.backend.registry import register_dataset


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

    def iter_split(self, split: Split) -> Iterator[Path]:
        yield from super().iter_split(split)

    def iter_batches(self, split: Split, batch_size: int) -> Iterator[list[Path]]:
        yield from super().iter_batches(split, batch_size)

    def get_ground_truth(self, image_path: Path) -> list[Prediction]:
        split = self._split_of(image_path)
        return self._annotations_by_split[split].get(Path(image_path).name, [])

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
        index: dict[str, list[Prediction]] = {file_name: [] for file_name in file_name_by_image_id.values()}

        for annotation in coco["annotations"]:
            file_name = file_name_by_image_id[annotation["image_id"]]
            x, y, w, h = annotation["bbox"]
            index[file_name].append(
                Prediction(
                    class_id=annotation["category_id"],
                    confidence=1.0,
                    bbox=(x, y, x + w, y + h),
                )
            )

        return index
