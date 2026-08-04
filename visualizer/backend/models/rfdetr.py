# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

from pathlib import Path

import torch

from rfdetr.variants import RFDETRLarge, RFDETRMedium, RFDETRNano, RFDETRSmall
from visualizer.backend.models.basemodel import BaseModel
from visualizer.backend.prediction import Prediction
from visualizer.backend.registry import register_model


@register_model("rfdetr")
class RFDETR(BaseModel):
    def __init__(self, model_path: str | Path):
        super().__init__(model_path)
        self.model_variants = [RFDETRNano, RFDETRSmall, RFDETRMedium, RFDETRLarge]
        for variant in self.model_variants:
            try:
                self.model = variant(pretrain_weights=self.model_path)
                return
            except Exception as e:
                print(f"Model weights do not correspond to {variant}, retrying... \nError message: {e}")
        raise TypeError(f"Expected one of the following models, but could not find any: {self.model_variants}")

    def get_batch_embeddings(self, batch: list[str | Path]) -> tuple[torch.Tensor, list[list[Prediction]]]:
        input_batch: list[str] = [str(path) for path in batch]
        preds = self.model.predict(input_batch, threshold=0.0, return_query_embeddings=True)
        batch_embeddings: list = []
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
        batch_tensor_embeddings = torch.stack(batch_embeddings, dim=0)
        return batch_tensor_embeddings, batch_predictions
