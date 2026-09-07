# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

"""Tests for persisted per-record IoU and the IoU-driven COCO mAP/mAR metrics."""

import sqlite3
from pathlib import Path

import pytest
import torch

from visualizer.backend.dataset_inference_store import DB_FILENAME, DatasetInferenceStore
from visualizer.backend.evaluation.types import Match
from visualizer.backend.evaluator import match_detections
from visualizer.backend.inference.types import EmbeddingRecord
from visualizer.backend.metrics.coco_detection_metrics import COCODetectionMetricsCalculator
from visualizer.backend.shared_types.prediction import Prediction


def _make_store(tmp_path: Path) -> DatasetInferenceStore:
    store = DatasetInferenceStore(tmp_path)
    store.create_tables()
    return store


class TestStoreIoUPersistence:
    """The ``records`` table persists a per-record IoU as a REAL column."""

    def test_insert_records_persists_iou(self, tmp_path: Path) -> None:
        store = _make_store(tmp_path)

        store.insert_records(
            [
                EmbeddingRecord(
                    id="r1",
                    image_path=str(tmp_path / "a.jpg"),
                    split="VAL",
                    embedding=None,
                    prediction=Prediction(class_id=1, confidence=0.9, bbox=(0, 0, 10, 10)),
                    ground_truth=Prediction(class_id=1, confidence=1.0, bbox=(0, 0, 10, 10)),
                    status="tp",
                    iou=0.83,
                )
            ]
        )

        rows = store.get_evaluation_rows()

        assert rows[0]["iou"] == pytest.approx(0.83)

    def test_persist_batch_persists_iou(self, tmp_path: Path) -> None:
        store = _make_store(tmp_path)

        store.persist_batch(
            [
                EmbeddingRecord(
                    id="r1",
                    image_path=str(tmp_path / "a.jpg"),
                    split="VAL",
                    embedding=None,
                    prediction=Prediction(class_id=1, confidence=0.9, bbox=(0, 0, 10, 10)),
                    ground_truth=Prediction(class_id=1, confidence=1.0, bbox=(0, 0, 10, 10)),
                    status="tp",
                    iou=0.77,
                )
            ],
            [(str(tmp_path / "a.jpg"), "VAL")],
            num_images_processed=1,
        )

        rows = store.get_evaluation_rows()

        assert rows[0]["iou"] == pytest.approx(0.77)

    def test_false_negative_has_null_iou(self, tmp_path: Path) -> None:
        store = _make_store(tmp_path)

        store.insert_records(
            [
                EmbeddingRecord(
                    id="r1",
                    image_path=str(tmp_path / "a.jpg"),
                    split="VAL",
                    embedding=None,
                    prediction=None,
                    ground_truth=Prediction(class_id=1, confidence=1.0, bbox=(0, 0, 10, 10)),
                    status="fn",
                )
            ]
        )

        rows = store.get_evaluation_rows()

        assert rows[0]["iou"] is None

    def test_create_tables_migrates_legacy_records_table(self, tmp_path: Path) -> None:
        db_path = tmp_path / DB_FILENAME
        with sqlite3.connect(db_path) as conn:
            conn.executescript(
                """
                CREATE TABLE records (
                    id              TEXT PRIMARY KEY,
                    image_path      TEXT NOT NULL,
                    split           TEXT NOT NULL,
                    status          TEXT NOT NULL,
                    pred_class_id   INTEGER,
                    pred_confidence REAL,
                    pred_x1         REAL,
                    pred_y1         REAL,
                    pred_x2         REAL,
                    pred_y2         REAL,
                    gt_class_id     INTEGER,
                    gt_confidence   REAL,
                    gt_x1           REAL,
                    gt_y1           REAL,
                    gt_x2           REAL,
                    gt_y2           REAL,
                    raw_embedding   TEXT,
                    pca_embedding   TEXT,
                    pca_components  INTEGER
                );
                INSERT INTO records (id, image_path, split, status)
                VALUES ('legacy', 'a.jpg', 'VAL', 'tp');
                """
            )

        DatasetInferenceStore(tmp_path).create_tables()

        with sqlite3.connect(db_path) as conn:
            columns = {row[1] for row in conn.execute("PRAGMA table_info(records)")}
        assert "iou" in columns


class TestEvaluatorIoU:
    """``match_detections`` reports the IoU of every resolved pairing."""

    def test_matched_pair_reports_iou(self) -> None:
        predictions = [Prediction(class_id=1, confidence=0.9, bbox=(0.0, 0.0, 10.0, 10.0))]
        ground_truths = [Prediction(class_id=1, confidence=1.0, bbox=(0.0, 0.0, 10.0, 5.0))]

        matches = match_detections(predictions, torch.zeros((1, 2)), ground_truths)

        assert matches[0].status == "tp"
        assert matches[0].iou == pytest.approx(0.5)

    def test_false_positive_has_no_iou(self) -> None:
        predictions = [Prediction(class_id=1, confidence=0.9, bbox=(0.0, 0.0, 10.0, 10.0))]
        ground_truths = [Prediction(class_id=1, confidence=1.0, bbox=(90.0, 90.0, 100.0, 100.0))]

        matches = match_detections(predictions, torch.zeros((1, 2)), ground_truths)

        assert matches[0].status == "fp"
        assert matches[0].iou is None

    def test_false_negative_has_no_iou(self) -> None:
        ground_truths = [Prediction(class_id=1, confidence=1.0, bbox=(0.0, 0.0, 10.0, 10.0))]

        matches = match_detections([], torch.zeros((0, 2)), ground_truths)

        assert matches[0].status == "fn"
        assert matches[0].iou is None


class TestCocoMeanAveragePrecision:
    """MAP/mAR are derived from stored IoU rather than TP/FP/FN ratios."""

    @staticmethod
    def _perfect_match(iou: float) -> Match:
        return Match(
            prediction=Prediction(class_id=1, confidence=0.9, bbox=(0, 0, 10, 10)),
            embedding=None,
            ground_truth=Prediction(class_id=1, confidence=1.0, bbox=(0, 0, 10, 10)),
            status="tp",
            iou=iou,
        )

    def test_perfect_iou_yields_full_map(self) -> None:
        calculator = COCODetectionMetricsCalculator({1: "cat"})

        metrics = calculator.calculate([self._perfect_match(1.0)])

        assert metrics["mAP50"] == pytest.approx(1.0)

    def test_perfect_iou_yields_full_map_across_iou_range(self) -> None:
        calculator = COCODetectionMetricsCalculator({1: "cat"})

        metrics = calculator.calculate([self._perfect_match(1.0)])

        assert metrics["mAP50:90"] == pytest.approx(1.0)

    def test_loose_box_counts_at_iou_50(self) -> None:
        calculator = COCODetectionMetricsCalculator({1: "cat"})

        metrics = calculator.calculate([self._perfect_match(0.6)])

        assert metrics["mAP50"] == pytest.approx(1.0)

    def test_loose_box_is_penalised_at_strict_iou(self) -> None:
        calculator = COCODetectionMetricsCalculator({1: "cat"})

        metrics = calculator.calculate([self._perfect_match(0.6)])

        # Only IoU thresholds 0.50, 0.55 and 0.60 are satisfied out of the 10 COCO thresholds.
        assert metrics["mAP50:90"] == pytest.approx(0.3)

    def test_strict_box_scores_higher_than_loose_box(self) -> None:
        calculator = COCODetectionMetricsCalculator({1: "cat"})

        loose = calculator.calculate([self._perfect_match(0.6)])
        strict = calculator.calculate([self._perfect_match(0.95)])

        assert strict["mAP50:90"] > loose["mAP50:90"]

    def test_recall_accounts_for_missed_ground_truth(self) -> None:
        calculator = COCODetectionMetricsCalculator({1: "cat"})
        missed = Match(
            prediction=None,
            embedding=None,
            ground_truth=Prediction(class_id=1, confidence=1.0, bbox=(0, 0, 10, 10)),
            status="fn",
        )

        metrics = calculator.calculate([self._perfect_match(1.0), missed])

        assert metrics["mAR50"] == pytest.approx(0.5)

    def test_average_precision_penalises_false_positives(self) -> None:
        calculator = COCODetectionMetricsCalculator({1: "cat"})
        false_positive = Match(
            prediction=Prediction(class_id=1, confidence=0.95, bbox=(0, 0, 10, 10)),
            embedding=None,
            ground_truth=None,
            status="fp",
        )

        metrics = calculator.calculate([self._perfect_match(1.0), false_positive])

        assert metrics["mAP50"] < 1.0

    def test_misclassified_detection_is_not_a_true_positive(self) -> None:
        calculator = COCODetectionMetricsCalculator({1: "cat", 2: "dog"})
        misclassified = Match(
            prediction=Prediction(class_id=2, confidence=0.9, bbox=(0, 0, 10, 10)),
            embedding=None,
            ground_truth=Prediction(class_id=1, confidence=1.0, bbox=(0, 0, 10, 10)),
            status="misclassified",
            iou=0.9,
        )

        metrics = calculator.calculate([misclassified])

        assert metrics["mAP50"] == pytest.approx(0.0)

    def test_legacy_records_without_iou_still_score_at_iou_50(self) -> None:
        calculator = COCODetectionMetricsCalculator({1: "cat"})
        legacy = Match(
            prediction=Prediction(class_id=1, confidence=0.9, bbox=(0, 0, 10, 10)),
            embedding=None,
            ground_truth=Prediction(class_id=1, confidence=1.0, bbox=(0, 0, 10, 10)),
            status="tp",
        )

        metrics = calculator.calculate([legacy])

        assert metrics["mAP50"] == pytest.approx(1.0)
