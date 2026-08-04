# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

from visualizer.backend.datasets.basedataset import BaseDataset
from visualizer.backend.models.basemodel import BaseModel

MODEL_REGISTRY: dict[str, type[BaseModel]] = {}
DATASET_REGISTRY: dict[str, type[BaseDataset]] = {}


def register_model(name: str):
    def deco(cls):
        MODEL_REGISTRY[name] = cls
        return cls

    return deco


def register_dataset(name: str):
    def deco(cls):
        DATASET_REGISTRY[name] = cls
        return cls

    return deco
