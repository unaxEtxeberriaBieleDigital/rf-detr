# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Utilities for RF-DETR detections, including per-query decoder embeddings."""

from dataclasses import dataclass

import numpy as np
import torch
from supervision import Detections


@dataclass
class RFDETRDetections(Detections):
    """`supervision.Detections` subclass carrying per-detection decoder query embeddings.

    Attributes:
        query_embeddings: Array of shape ``[num_detections, num_decoder_layers * hidden_dim]``
            with the concatenated per-decoder-layer embedding of each kept query, aligned
            1:1 with the other `Detections` fields (`xyxy`, `confidence`, `class_id`, ...).
    """

    query_embeddings: np.ndarray | None = None


def stack_decoder_layers(query_embeddings: torch.Tensor) -> torch.Tensor:
    """Concatenates per-decoder-layer query embeddings into one feature vector per query.

    Args:
        query_embeddings: Decoder hidden states of shape
            ``[num_decoder_layers, batch_size, num_queries, hidden_dim]``.

    Returns:
        Tensor of shape ``[batch_size, num_queries, num_decoder_layers * hidden_dim]``, with the
        hidden states of all decoder layers concatenated along the last dimension for each query.
    """
    num_layers, batch_size, num_queries, hidden_dim = query_embeddings.shape
    # [batch_size, num_queries, num_decoder_layers, hidden_dim]
    query_embeddings = query_embeddings.permute(1, 2, 0, 3)
    # [batch_size, num_queries, num_decoder_layers * hidden_dim]
    return query_embeddings.reshape(batch_size, num_queries, num_layers * hidden_dim)