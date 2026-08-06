# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

import json
from pathlib import Path

from visualizer.backend.datasets.basedataset import Split
from visualizer.backend.datasets.cocodetectiondataset import COCODetectionDataset


def _write_coco_annotations(split_dir: Path, file_name: str) -> None:
    payload = {
        "images": [{"id": 1, "file_name": file_name, "width": 640, "height": 480}],
        "annotations": [{"id": 10, "image_id": 1, "category_id": 2, "bbox": [10, 20, 100, 50]}],
        "categories": [{"id": 2, "name": "defecto"}],
    }
    (split_dir / "_annotations.coco.json").write_text(json.dumps(payload), encoding="utf-8")


def test_get_ground_truth_resolves_split_prefixed_file_name(tmp_path: Path) -> None:
    valid_dir = tmp_path / "valid"
    valid_dir.mkdir()
    (valid_dir / "image_001.jpg").touch()
    _write_coco_annotations(valid_dir, "valid/image_001.jpg")

    dataset = COCODetectionDataset(tmp_path)
    gts = dataset.get_ground_truth(valid_dir / "image_001.jpg")

    assert Split.VAL in dataset.splits
    assert len(gts) == 1
    assert gts[0].class_id == 2
    assert gts[0].bbox == (10, 20, 110, 70)


def test_get_ground_truth_matches_case_insensitive_file_name(tmp_path: Path) -> None:
    valid_dir = tmp_path / "valid"
    valid_dir.mkdir()
    (valid_dir / "Image_ABC.JPG").touch()
    _write_coco_annotations(valid_dir, "image_abc.jpg")

    dataset = COCODetectionDataset(tmp_path)
    gts = dataset.get_ground_truth(valid_dir / "Image_ABC.JPG")

    assert len(gts) == 1
    assert gts[0].class_id == 2
