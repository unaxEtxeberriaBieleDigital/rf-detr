# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

from pathlib import Path

import torch

from rfdetr.utilities.logger import get_logger
from rfdetr.variants import RFDETRLarge, RFDETRMedium, RFDETRNano, RFDETRSmall
from visualizer.backend.models.basemodel import BaseModel
from visualizer.backend.prediction import Prediction
from visualizer.backend.registry import register_model

logger = get_logger()


@register_model("rfdetr")
class RFDETR(BaseModel):
    def __init__(self, model_path: str | Path):
        super().__init__(model_path)

        # Explicitly pick CUDA when available instead of relying on RF-DETR's implicit
        # per-variant default, so the visualizer never silently falls back to a slow CPU run
        # and the choice is always visible in the logs.
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        if self.device == "cuda":
            logger.info(f"Visualizer inference will run on GPU: {torch.cuda.get_device_name(0)}")
        else:
            logger.warning(
                "CUDA is not available; visualizer inference will run on CPU. This will be significantly slower."
            )

        self.model_variants = [RFDETRNano, RFDETRSmall, RFDETRMedium, RFDETRLarge]
        variant_found = False
        for variant in self.model_variants:
            try:
                self.model = variant(pretrain_weights=self.model_path, device=self.device)
                logger.info(f"Loaded {variant.__name__} weights from '{self.model_path}' on device '{self.device}'")
                variant_found = True
            except Exception as e:
                logger.debug(f"Model weights do not correspond to {variant}, retrying... \nError message: {e}")

        if (not variant_found): raise TypeError(f"Expected one of the following models, but could not find any: {self.model_variants}")

        self.input_shape = self.model.resolution

    def get_batch_embeddings(self, batch: list[str | Path]) -> tuple[list[torch.Tensor], list[list[Prediction]]]:
        logger.debug(f"Running inference on a batch of {len(batch)} image(s) on device '{self.device}'")
        input_batch: list[str] = [str(path) for path in batch]
        preds = self.model.predict(input_batch, threshold=0.05, return_query_embeddings=True)
        batch_embeddings: list[torch.Tensor] = []
        batch_predictions: list[list[Prediction]] = []
        for pred in preds:
            # `pred.query_embeddings` is gathered with the same `keep`/`query_indices` mask as
            # `pred.xyxy`/`pred.confidence`/`pred.class_id` (see RFDETRDetections docstring and
            # detr.py predict()), so row i of query_embeddings always corresponds to detection i
            # of this same image: the order is preserved across embeddings and predictions.
            batch_embeddings.append(torch.Tensor(pred.query_embeddings))
            batch_predictions.append(
                [
                    Prediction(
                        class_id=int(class_id),
                        confidence=float(confidence),
                        bbox=tuple(float(v) for v in bbox),
                    )
                    for bbox, confidence, class_id in zip(pred.xyxy, pred.confidence, pred.class_id)
                ]
            )
        # Images have a variable number of detections (some may have none at all), so the
        # per-image embedding tensors have different first-dimension sizes and cannot be
        # `torch.stack`-ed into one contiguous batch tensor. Return them as a list instead,
        # one `[num_detections_i, hidden_dim]` tensor per image, aligned 1:1 with `batch`.
        return batch_embeddings, batch_predictions
