# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

from abc import ABC, abstractmethod
from pathlib import Path

import torch

from visualizer.backend.prediction import Prediction


class BaseModel(ABC):
    def __init__(self, model_path: str | Path):
        model_path = Path(model_path)

        self._path_sanity_checks(model_path)
        self.model_path: Path = model_path

    def _path_sanity_checks(self, path: Path):
        if not path.exists():
            raise ModuleNotFoundError(f"Could not find model path: {path}")
        if path.suffix != ".pth":
            raise TypeError(f"Expected a pth file, but got the following file: {path}")

    @abstractmethod
    def get_batch_embeddings(self, batch: list[str | Path]) -> tuple[torch.Tensor, list[list[Prediction]]]:
        """Method to extract the embeddings of a batch.

            Supported dimensions: * [batch_size, hidden_dim]
            * [batch_size, num_queries, hidden_dim]

        Returns:
            A tuple ``(embeddings, predictions)`` where ``predictions[i][j]`` (a ``Prediction``
            with ``bbox``, ``confidence`` and ``class_id``) is the detection tied to
            ``embeddings[i][j]``, i.e. predictions are aligned 1:1 with embeddings, in the
            same order, for every image in the batch.
        """
        pass
