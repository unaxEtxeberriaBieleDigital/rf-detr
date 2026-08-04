# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

from dataclasses import dataclass

import torch
from torchvision.ops import box_iou

from visualizer.backend.prediction import Prediction


@dataclass
class Match:
    """One resolved (prediction, ground_truth) pairing for a single image.

    Attributes:
        prediction: The model's detection, or ``None`` for a false negative (a ground-truth
            box that no prediction was matched to).
        embedding: The query embedding backing ``prediction``, aligned 1:1 with it, or
            ``None`` when ``prediction`` is ``None``.
        ground_truth: The matched annotation, or ``None`` for a false positive.
        status: ``"tp"`` (correct box + class), ``"misclassified"`` (correct box, wrong
            class), ``"fp"`` (no matching ground truth), or ``"fn"`` (missed ground truth).
    """

    prediction: Prediction | None
    embedding: list[float] | None
    ground_truth: Prediction | None
    status: str


def match_detections(
    predictions: list[Prediction],
    embeddings: torch.Tensor,
    ground_truths: list[Prediction],
    iou_threshold: float = 0.5,
) -> list[Match]:
    """Greedily matches detections to ground truth boxes by IoU (COCO-style).

    Predictions are visited in descending confidence order. Each is paired with its
    highest-IoU still-unmatched ground truth box (if that IoU clears ``iou_threshold``);
    the pair is labeled ``"tp"`` if the classes agree, ``"misclassified"`` otherwise.
    Predictions left without a qualifying ground truth become ``"fp"``. Ground truth boxes
    that end up unmatched become ``"fn"`` entries with no prediction/embedding.

    Args:
        predictions: Detections for one image, in the same order as ``embeddings`` rows.
        embeddings: Per-query embeddings for the same image, shape ``[num_queries, hidden_dim]``,
            aligned 1:1 with ``predictions`` (row ``i`` belongs to ``predictions[i]``).
        ground_truths: Annotated boxes for the same image.
        iou_threshold: Minimum IoU for a prediction/ground-truth pair to be considered a match.

    Returns:
        One `Match` per prediction plus one `Match` per unmatched ground truth.
    """
    matches: list[Match] = []
    matched_gt_indices: set[int] = set()

    gt_boxes = (
        torch.tensor([gt.bbox for gt in ground_truths], dtype=torch.float32)
        if ground_truths
        else torch.zeros((0, 4), dtype=torch.float32)
    )

    prediction_order = sorted(range(len(predictions)), key=lambda i: predictions[i].confidence, reverse=True)

    for i in prediction_order:
        prediction = predictions[i]
        embedding = embeddings[i].tolist()

        if not ground_truths:
            matches.append(Match(prediction, embedding, None, "fp"))
            continue

        prediction_box = torch.tensor([prediction.bbox], dtype=torch.float32)
        ious = box_iou(prediction_box, gt_boxes)[0]
        for matched_idx in matched_gt_indices:
            ious[matched_idx] = -1.0

        best_iou, best_gt_idx = torch.max(ious, dim=0)
        best_gt_idx = int(best_gt_idx.item())

        if best_iou.item() >= iou_threshold:
            matched_gt_indices.add(best_gt_idx)
            ground_truth = ground_truths[best_gt_idx]
            status = "tp" if ground_truth.class_id == prediction.class_id else "misclassified"
            matches.append(Match(prediction, embedding, ground_truth, status))
        else:
            matches.append(Match(prediction, embedding, None, "fp"))

    for gt_idx, ground_truth in enumerate(ground_truths):
        if gt_idx not in matched_gt_indices:
            matches.append(Match(None, None, ground_truth, "fn"))

    return matches
