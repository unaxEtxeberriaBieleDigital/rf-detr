# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Shared test helpers for the inference test suite.

Plain classes and functions (not pytest fixtures) shared across multiple test modules to avoid verbatim duplication.
Import with a relative import::

    from .helpers import _BaseFakeRFDETR, _DummyModel, _DummyRFDETR
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import torch

from rfdetr.detr import RFDETR


class _BaseFakeRFDETR(RFDETR):
    """RFDETR test double that skips weight downloads and returns a minimal model config.

    Subclasses must override ``get_model`` to supply the model context appropriate for
    the scenario under test.

    Examples:
        This class is imported directly by test modules that need a weight-free RFDETR.
    """

    def maybe_download_pretrain_weights(self) -> None:
        """Skip weight download in tests."""
        return None

    def get_model_config(self, **kwargs: object) -> SimpleNamespace:
        """Return a minimal config sufficient for most test scenarios."""
        return SimpleNamespace(num_channels=3)


class _DummyModel:
    """Minimal model stub that returns deterministic postprocessed results.

    Examples:
        >>> m = _DummyModel(labels=[0, 1])
        >>> len(m._labels)
        2
    """

    def __init__(
        self,
        class_names: list[str] | None = None,
        labels: list[int] | None = None,
        include_keypoints: bool = False,
        num_keypoints: int = 17,
    ) -> None:
        """Initialise stub with optional class names, label list, and keypoint flag."""
        self.device = torch.device("cpu")
        self.resolution = 28
        self.model = torch.nn.Identity()
        self.class_names = class_names
        self._labels = labels if labels is not None else [1]
        self._include_keypoints = include_keypoints
        self._num_keypoints = num_keypoints

    def postprocess(self, predictions: Any, target_sizes: torch.Tensor) -> list[dict[str, torch.Tensor]]:
        """Return fixed scores/boxes (and optional keypoints) for every image in the batch."""
        batch = target_sizes.shape[0]
        results = []
        for _ in range(batch):
            result: dict[str, torch.Tensor] = {
                "scores": torch.tensor([0.9] * len(self._labels)),
                "labels": torch.tensor(self._labels),
                "boxes": torch.tensor([[0.0, 0.0, 1.0, 1.0]] * len(self._labels)),
            }
            if self._include_keypoints:
                result["keypoints"] = torch.full((len(self._labels), self._num_keypoints, 3), 0.5, dtype=torch.float32)
                result["keypoint_precision_cholesky"] = torch.full(
                    (len(self._labels), self._num_keypoints, 3), 0.25, dtype=torch.float32
                )
            results.append(result)
        return results


class _QueryEmbeddingForward(torch.nn.Module):
    """``nn.Module`` forward stub returning value-tagged, per-query embeddings.

    A real ``nn.Module`` (rather than a plain function) so ``model.eval()`` in
    ``RFDETR._ensure_eval_mode_for_unoptimized_inference`` keeps working, and its ``forward``
    accepts the ``return_query_embeddings`` keyword the same way the real decoder model does.
    """

    def __init__(self, num_queries: int, hidden_dim: int) -> None:
        """Store the query/embedding dimensions used to fabricate outputs."""
        super().__init__()
        self._num_queries = num_queries
        self._hidden_dim = hidden_dim

    def forward(self, batch_tensor: torch.Tensor, return_query_embeddings: bool = False) -> dict[str, torch.Tensor]:
        """Return a predictions dict with per-query, value-tagged embeddings."""
        batch = batch_tensor.shape[0]
        pred_logits = torch.zeros(batch, self._num_queries, 1)
        pred_boxes = torch.zeros(batch, self._num_queries, 4)
        result: dict[str, torch.Tensor] = {"pred_logits": pred_logits, "pred_boxes": pred_boxes}
        if return_query_embeddings:
            per_query = torch.arange(self._num_queries, dtype=torch.float32).view(1, self._num_queries, 1)
            per_query = per_query.expand(1, self._num_queries, self._hidden_dim)
            # [num_decoder_layers=1, batch, num_queries, hidden_dim]
            result["query_embeddings"] = per_query.unsqueeze(0).expand(1, batch, self._num_queries, self._hidden_dim)
        return result


class _QueryEmbeddingDummyModel:
    """Model stub for verifying that ``query_embeddings`` are gathered by query index, not position.

    ``postprocess`` returns a ``query_indices`` permutation that deliberately does not match the
    boxes/scores order (mirroring real top-k selection, which is sorted by score, not query index),
    so tests can detect if ``RFDETR.predict`` ever regresses to boolean-indexing the query-order
    embedding tensor directly instead of gathering by ``query_indices`` first.

    Examples:
        >>> m = _QueryEmbeddingDummyModel(num_queries=3, hidden_dim=2)
        >>> m.query_indices.tolist()
        [2, 0, 1]
    """

    def __init__(self, num_queries: int = 3, hidden_dim: int = 2) -> None:
        """Initialise stub with a fixed non-identity ``query_indices`` permutation.

        Args:
            num_queries: Number of decoder queries to simulate.
            hidden_dim: Per-layer embedding width; embeddings use a single fake decoder layer.
        """
        self.device = torch.device("cpu")
        self.resolution = 28
        self.class_names = None
        self._num_queries = num_queries
        self._hidden_dim = hidden_dim
        # Non-identity on purpose: query_indices[j] is the originating query for the j-th
        # top-k slot, e.g. slot 0 came from query 2, slot 1 from query 0, slot 2 from query 1.
        self.query_indices = torch.tensor([2, 0, 1][:num_queries])
        # Embedding for query q is filled with value q, so gathered embeddings can be checked
        # by value against the expected originating query index.
        self.model = _QueryEmbeddingForward(num_queries, hidden_dim)

    def postprocess(self, predictions: Any, target_sizes: torch.Tensor) -> list[dict[str, torch.Tensor]]:
        """Return fixed scores/boxes with a non-identity ``query_indices`` per image."""
        batch = target_sizes.shape[0]
        num_detections = self.query_indices.shape[0]
        results = []
        for _ in range(batch):
            results.append(
                {
                    "scores": torch.tensor([0.9] * num_detections),
                    "labels": torch.tensor([1] * num_detections),
                    "boxes": torch.tensor([[0.0, 0.0, 1.0, 1.0]] * num_detections),
                    "query_indices": self.query_indices.clone(),
                }
            )
        return results


class _DummyRFDETR(RFDETR):
    """Weight-free RFDETR that delegates to ``_DummyModel`` for all inference.

    Examples:
        >>> m = _DummyRFDETR()
        >>> isinstance(m.model, _DummyModel)
        True
    """

    def maybe_download_pretrain_weights(self) -> None:
        """Skip weight download in tests."""
        return None

    def get_model_config(self, **kwargs: object) -> SimpleNamespace:
        """Return a minimal namespace with just ``num_channels``."""
        return SimpleNamespace(num_channels=3)

    def get_model(self, config: SimpleNamespace, *, trust_checkpoint: bool = False) -> _DummyModel:
        """Return a fresh ``_DummyModel`` instance."""
        return _DummyModel()
