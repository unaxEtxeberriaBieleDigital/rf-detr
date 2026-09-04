# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

import json
import sqlite3
from collections.abc import Iterator
from pathlib import Path

import pytest
import torch
from fastapi import HTTPException

from visualizer.backend.app import JobRequest, check_dataset, create_job
from visualizer.backend.datasets.basedataset import Split
from visualizer.backend.datasets.cocodetectiondataset import COCODetectionDataset
from visualizer.backend.jobs import JOB_STORE, DatasetInferenceJobStatus, run_job
from visualizer.backend.models.basemodel import BaseModel
from visualizer.backend.shared_types.prediction import Prediction
from visualizer.backend.registry import MODEL_REGISTRY
from visualizer.backend.dataset_inference_store import DB_FILENAME, DatasetInferenceStore
from visualizer.backend import jobs as jobs_module


class FakeModel(BaseModel):
    calls: list[list[str]] = []

    def get_batch_embeddings(
        self, batch: list[str | Path]
    ) -> tuple[list[torch.Tensor], list[list[Prediction]]]:
        names = [Path(item).name for item in batch]
        FakeModel.calls.append(names)

        embeddings: list[torch.Tensor] = []
        predictions: list[list[Prediction]] = []
        for item in batch:
            image_name = Path(item).name
            if image_name.startswith("empty"):
                embeddings.append(torch.zeros((0, 2), dtype=torch.float32))
                predictions.append([])
                continue

            embeddings.append(torch.tensor([[0.1, 0.2]], dtype=torch.float32))
            predictions.append(
                [Prediction(class_id=1, confidence=0.9, bbox=(0.0, 0.0, 1.0, 1.0))]
            )
        return embeddings, predictions


@pytest.fixture(autouse=True)
def _reset_visualizer_state(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    JOB_STORE.clear()
    jobs_module._ACTIVE_JOB_IDS_BY_PATH.clear()
    FakeModel.calls.clear()
    monkeypatch.setitem(MODEL_REGISTRY, "fake_model", FakeModel)
    yield
    JOB_STORE.clear()
    jobs_module._ACTIVE_JOB_IDS_BY_PATH.clear()
    MODEL_REGISTRY.pop("fake_model", None)
    FakeModel.calls.clear()


def _write_dataset(dataset_path: Path, image_names: list[str]) -> Path:
    valid_dir = dataset_path / "valid"
    valid_dir.mkdir(parents=True)

    for image_name in image_names:
        (valid_dir / image_name).touch()

    payload = {
        "images": [
            {"id": index + 1, "file_name": image_name, "width": 32, "height": 32}
            for index, image_name in enumerate(image_names)
        ],
        "annotations": [],
        "categories": [{"id": 1, "name": "object"}],
    }
    (valid_dir / "_annotations.coco.json").write_text(json.dumps(payload), encoding="utf-8")
    return valid_dir


def _write_legacy_db(dataset_path: Path, *, status: str, total: int, processed: int) -> None:
    db_path = dataset_path / DB_FILENAME
    with sqlite3.connect(db_path) as conn:
        conn.executescript(
            """
            CREATE TABLE job_meta (
                key   TEXT PRIMARY KEY,
                value TEXT
            );

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
            """
        )
        conn.executemany(
            "INSERT INTO job_meta (key, value) VALUES (?, ?)",
            [
                ("status", json.dumps(status)),
                ("num_images_total", json.dumps(total)),
                ("num_images_processed", json.dumps(processed)),
            ],
        )


def _default_request(dataset_path: Path, model_path: Path, **overrides: object) -> JobRequest:
    payload: dict[str, object] = {
        "dataset_path": str(dataset_path),
        "dataset_type": "coco_detection",
        "model_path": str(model_path),
        "model_type": "fake_model",
        "splits": None,
        "batch_size": 8,
        "iou_threshold": 0.5,
        "resume": False,
    }
    payload.update(overrides)
    return JobRequest(**payload)


def test_store_persist_batch_tracks_zero_record_images_and_reset_cleans_progress(
    tmp_path: Path,
) -> None:
    store = DatasetInferenceStore(tmp_path)
    store.create_tables()

    zero_record_image = str(tmp_path / "valid" / "empty.jpg")
    store.persist_batch([], [(zero_record_image, "VAL")], num_images_processed=1)

    assert store.record_count() == 0
    assert store.processed_image_count() == 1
    assert store.get_processed_image_paths() == {zero_record_image}

    store.reset_for_rerun()

    assert store.record_count() == 0
    assert store.processed_image_count() == 0
    assert store.get_processed_image_paths() == set()


def test_run_job_resume_skips_processed_images_and_preserves_counters(tmp_path: Path) -> None:
    valid_dir = _write_dataset(tmp_path, ["empty_a.jpg", "keep_b.jpg", "keep_c.jpg"])
    model_path = tmp_path / "model.pth"
    model_path.touch()

    dataset = COCODetectionDataset(tmp_path)
    model = FakeModel(model_path)
    store = DatasetInferenceStore(tmp_path)
    store.create_tables()
    store.enable_progress_tracking()
    store.set_run_config(
        {
            "dataset_type": "coco_detection",
            "model_path": str(model_path),
            "model_type": "fake_model",
            "splits": ["VAL"],
            "batch_size": 2,
            "iou_threshold": 0.5,
        }
    )
    store.persist_batch([], [(str(valid_dir / "empty_a.jpg"), "VAL")], num_images_processed=1)
    store.set_meta("status", "error")
    store.set_meta("num_images_total", 3)

    job = DatasetInferenceJobStatus(id="resume-job", store=store, categories={1: "object"})
    run_job(job, dataset, model, [Split.VAL], batch_size=2, iou_threshold=0.5, resume=True)

    flattened_calls = [image_name for batch in FakeModel.calls for image_name in batch]
    assert job.status == "done"
    assert job.num_images_total == 3
    assert job.num_images_processed == 3
    assert store.processed_image_count(split_names=["VAL"]) == 3
    assert store.record_count() == 2
    assert sorted(flattened_calls) == ["keep_b.jpg", "keep_c.jpg"]
    assert "empty_a.jpg" not in flattened_calls


def test_check_dataset_reports_resume_fields_for_new_and_legacy_dbs(tmp_path: Path) -> None:
    resumable_dir = tmp_path / "resumable"
    resumable_dir.mkdir()
    resumable_store = DatasetInferenceStore(resumable_dir)
    resumable_store.create_tables()
    resumable_store.enable_progress_tracking()
    resumable_store.set_run_config(
        {
            "dataset_type": "coco_detection",
            "model_path": "model.pth",
            "model_type": "fake_model",
            "splits": ["VAL"],
            "batch_size": 8,
            "iou_threshold": 0.5,
        }
    )
    resumable_store.set_meta("status", "error")
    resumable_store.set_meta("num_images_total", 5)
    resumable_store.set_meta("num_images_processed", 2)

    resumable_response = check_dataset(str(resumable_dir))

    assert resumable_response.has_db is True
    assert resumable_response.can_resume is True
    assert resumable_response.num_images_total == 5
    assert resumable_response.num_images_processed == 2
    assert resumable_response.num_images_remaining == 3

    legacy_dir = tmp_path / "legacy"
    legacy_dir.mkdir()
    _write_legacy_db(legacy_dir, status="error", total=4, processed=1)

    legacy_response = check_dataset(str(legacy_dir))

    assert legacy_response.has_db is True
    assert legacy_response.can_resume is False
    assert legacy_response.num_images_total == 4
    assert legacy_response.num_images_processed == 1
    assert legacy_response.num_images_remaining == 3


def test_create_job_resume_rejects_db_without_progress_support(tmp_path: Path) -> None:
    _write_dataset(tmp_path, ["one.jpg"])
    model_path = tmp_path / "model.pth"
    model_path.touch()

    store = DatasetInferenceStore(tmp_path)
    store.create_tables()
    store.set_meta("status", "error")
    store.set_meta("num_images_total", 1)
    store.set_meta("num_images_processed", 0)

    request = _default_request(tmp_path, model_path, resume=True)

    with pytest.raises(HTTPException) as exc_info:
        create_job(request)

    assert exc_info.value.status_code == 409
    assert "does not support persisted progress" in str(exc_info.value.detail)


def test_create_job_resume_rejects_mismatched_persisted_run_config(tmp_path: Path) -> None:
    _write_dataset(tmp_path, ["one.jpg"])
    model_path = tmp_path / "model.pth"
    model_path.touch()

    store = DatasetInferenceStore(tmp_path)
    store.create_tables()
    store.enable_progress_tracking()
    store.set_run_config(
        {
            "dataset_type": "coco_detection",
            "model_path": str(model_path),
            "model_type": "fake_model",
            "splits": ["VAL"],
            "batch_size": 4,
            "iou_threshold": 0.4,
        }
    )
    store.set_meta("status", "error")
    store.set_meta("num_images_total", 1)
    store.set_meta("num_images_processed", 0)

    request = _default_request(tmp_path, model_path, resume=True, batch_size=8, iou_threshold=0.5)

    with pytest.raises(HTTPException) as exc_info:
        create_job(request)

    assert exc_info.value.status_code == 409
    assert "Resume configuration does not match" in str(exc_info.value.detail)


def test_create_job_resume_reattaches_to_existing_active_job(tmp_path: Path) -> None:
    _write_dataset(tmp_path, ["one.jpg"])
    model_path = tmp_path / "model.pth"
    model_path.touch()

    store = DatasetInferenceStore(tmp_path)
    store.create_tables()
    store.enable_progress_tracking()
    store.set_run_config(
        {
            "dataset_type": "coco_detection",
            "model_path": str(model_path),
            "model_type": "fake_model",
            "splits": ["VAL"],
            "batch_size": 8,
            "iou_threshold": 0.5,
        }
    )
    store.set_meta("status", "running")
    store.set_meta("num_images_total", 3)
    store.set_meta("num_images_processed", 1)

    active_job = DatasetInferenceJobStatus(
        id="active-job",
        store=store,
        status="running",
        categories={1: "object"},
        num_images_total=3,
        num_images_processed=1,
    )
    JOB_STORE[active_job.id] = active_job
    assert jobs_module.try_register_active_job(tmp_path, active_job.id) is None

    response = create_job(_default_request(tmp_path, model_path, resume=True))

    assert response.id == active_job.id
    assert response.status == "running"
    assert response.can_resume is True
    assert response.num_images_remaining == 2
    assert FakeModel.calls == []
