# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

import json
from abc import ABC, abstractmethod
from collections.abc import Iterator
from enum import Enum
from pathlib import Path

from visualizer.backend.prediction import Prediction

SUPPORTED_ANNOTATIONS = """
* COCO: only under the name of _annotations.coco.json
"""

SUPPORTED_IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".tif",
    ".tiff",
    ".webp",
}


class Split(Enum):
    TRAIN = 1
    TEST = 2
    VAL = 3


class DatasetType(Enum):
    CLASSIFICATION = 1
    COCO_DETECTION = 2


class BaseDataset(ABC):
    def __init__(self, dataset_path: str | Path, dataset_type: DatasetType):
        self.type = dataset_type
        dataset_path = Path(dataset_path)

        self._dataset_path_sanity_checks(dataset_path)

        self.path: Path = dataset_path
        self.splits: dict[Split, Path] = self._get_dataset_splits(dataset_path)
        self.categories: dict[int, str] = self._get_categories(dataset_path)

    @abstractmethod
    def iter_split(self, split: Split):
        split_path = self.splits[split]

        for file in split_path.iterdir():
            if file.is_file() and file.suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS:
                yield file

    @abstractmethod
    def iter_batches(self, split: Split, batch_size: int) -> Iterator[list[Path]]:
        batch = []

        for image in self.iter_split(split):
            batch.append(image)

            if len(batch) == batch_size:
                yield batch
                batch = []

        if batch:
            yield batch

    @abstractmethod
    def get_ground_truth(self, image_path: Path) -> list[Prediction]:
        pass

    def _dataset_path_sanity_checks(self, path: Path):
        if not path.exists():
            raise ModuleNotFoundError(f"Could not find dataset at path: {path}")
        if not path.is_dir():
            raise TypeError(f"Expected the path to be a directory: {path}")

    def _get_dataset_splits(self, path: Path) -> dict[Split, Path]:
        splits: dict[Split, Path] = {}

        for child in path.iterdir():
            if not child.is_dir():
                continue

            match child.name.lower():
                case "train":
                    if Split.TRAIN in splits:
                        raise Exception("More than one training set on the dataset")
                    splits[Split.TRAIN] = child

                case "test":
                    if Split.TEST in splits:
                        raise Exception("More than one testing set on the dataset")
                    splits[Split.TEST] = child

                case "val" | "valid":
                    if Split.VAL in splits:
                        raise Exception("More than one validation set on the dataset")
                    splits[Split.VAL] = child

        return splits

    def _get_categories(self, path: Path) -> dict[int, str]:
        categories: dict[int, str] | None = None

        for split_type, split_path in self.splits.items():
            annotations = split_path / "_annotations.coco.json"

            if not annotations.exists():
                continue

            with open(annotations, "r", encoding="utf-8") as f:
                coco = json.load(f)

            current_categories = {cat["id"]: cat["name"] for cat in coco["categories"]}

            if categories is None:
                categories = current_categories
            elif categories != current_categories:
                raise Exception(f"Category definitions differ between dataset splits ({split_type.name})")

        if categories is None:
            raise Exception("The dataset does not have supported annotations")
        return categories
