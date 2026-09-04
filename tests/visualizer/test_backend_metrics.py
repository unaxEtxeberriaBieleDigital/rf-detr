# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

from collections.abc import Iterator
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from visualizer.backend.app import app
from visualizer.backend.inference.types import EmbeddingRecord
from visualizer.backend.jobs import JOB_STORE, DatasetInferenceJobStatus
from visualizer.backend.shared_types.prediction import Prediction
from visualizer.backend.dataset_inference_store import DatasetInferenceStore


@pytest.fixture(autouse=True)
def _reset_visualizer_jobs() -> Iterator[None]:
    JOB_STORE.clear()
    yield
    JOB_STORE.clear()


@pytest.fixture
def client() -> Iterator[TestClient]:
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture
def finished_job(tmp_path: Path) -> DatasetInferenceJobStatus:
    dataset_path = tmp_path / "dataset"
    dataset_path.mkdir()
    store = DatasetInferenceStore(dataset_path)
    store.create_tables()
    store.set_meta("dataset_type", "coco_detection")
    store.set_meta("categories", {1: "cat", 2: "dog"})
    store.insert_records(
        [
            EmbeddingRecord(
                id="tp-class-1",
                image_path=str(dataset_path / "img1.jpg"),
                split="VAL",
                embedding=[0.1, 0.2],
                prediction=Prediction(class_id=1, confidence=0.9),
                ground_truth=Prediction(class_id=1, confidence=1.0),
                status="tp",
            ),
            EmbeddingRecord(
                id="fp-class-1",
                image_path=str(dataset_path / "img2.jpg"),
                split="VAL",
                embedding=[0.3, 0.4],
                prediction=Prediction(class_id=1, confidence=0.2),
                ground_truth=None,
                status="fp",
            ),
            EmbeddingRecord(
                id="tp-class-2",
                image_path=str(dataset_path / "img3.jpg"),
                split="VAL",
                embedding=[0.5, 0.6],
                prediction=Prediction(class_id=2, confidence=0.3),
                ground_truth=Prediction(class_id=2, confidence=1.0),
                status="tp",
            ),
            EmbeddingRecord(
                id="fp-class-2",
                image_path=str(dataset_path / "img4.jpg"),
                split="VAL",
                embedding=[0.7, 0.8],
                prediction=Prediction(class_id=2, confidence=0.8),
                ground_truth=None,
                status="fp",
            ),
            EmbeddingRecord(
                id="fn-any-class",
                image_path=str(dataset_path / "img5.jpg"),
                split="VAL",
                embedding=None,
                prediction=None,
                ground_truth=Prediction(class_id=1, confidence=1.0),
                status="fn",
            ),
        ]
    )

    job = DatasetInferenceJobStatus(
        id="finished-job",
        store=store,
        status="done",
        categories={1: "cat", 2: "dog"},
    )
    JOB_STORE[job.id] = job
    return job


def test_evaluation_applies_class_thresholds_without_reusing_unthresholded_cache(
    client: TestClient,
    finished_job: DatasetInferenceJobStatus,
) -> None:
    first_response = client.get(f"/api/v1/dataset_inference_jobs/{finished_job.id}/evaluation")

    assert first_response.status_code == 200
    first_payload = first_response.json()
    assert first_payload["cached"] is False
    assert first_payload["applied_class_thresholds"] is None
    assert first_payload["metrics"]["precision"] == pytest.approx(0.5)
    assert first_payload["metrics"]["recall"] == pytest.approx(2.0 / 3.0)

    cached_response = client.get(f"/api/v1/dataset_inference_jobs/{finished_job.id}/evaluation")

    assert cached_response.status_code == 200
    cached_payload = cached_response.json()
    assert cached_payload["cached"] is True
    assert cached_payload["metrics"] == first_payload["metrics"]

    thresholded_response = client.get(
        f"/api/v1/dataset_inference_jobs/{finished_job.id}/evaluation",
        params={"class_thresholds": '{"2": 0.5}'},
    )

    assert thresholded_response.status_code == 200
    thresholded_payload = thresholded_response.json()
    assert thresholded_payload["cached"] is False
    assert thresholded_payload["applied_class_thresholds"] == {"2": 0.5}
    assert thresholded_payload["metrics"]["precision"] == pytest.approx(1.0 / 3.0)
    assert thresholded_payload["metrics"]["recall"] == pytest.approx(1.0 / 3.0)
    assert thresholded_payload["metrics"] != first_payload["metrics"]

    cached_after_thresholded_response = client.get(f"/api/v1/dataset_inference_jobs/{finished_job.id}/evaluation")

    assert cached_after_thresholded_response.status_code == 200
    cached_after_thresholded_payload = cached_after_thresholded_response.json()
    assert cached_after_thresholded_payload["cached"] is True
    assert cached_after_thresholded_payload["metrics"] == first_payload["metrics"]


def test_evaluation_applies_record_ids_without_reusing_unfiltered_cache(
    client: TestClient,
    finished_job: DatasetInferenceJobStatus,
) -> None:
    first_response = client.get(f"/api/v1/dataset_inference_jobs/{finished_job.id}/evaluation")

    assert first_response.status_code == 200
    first_payload = first_response.json()
    assert first_payload["cached"] is False
    assert first_payload["applied_record_ids"] is None
    assert first_payload["applied_record_count"] is None

    cached_response = client.get(f"/api/v1/dataset_inference_jobs/{finished_job.id}/evaluation")

    assert cached_response.status_code == 200
    cached_payload = cached_response.json()
    assert cached_payload["cached"] is True
    assert cached_payload["metrics"] == first_payload["metrics"]

    filtered_params = {"record_ids": '["tp-class-1", "fn-any-class"]'}
    filtered_response = client.get(
        f"/api/v1/dataset_inference_jobs/{finished_job.id}/evaluation",
        params=filtered_params,
    )

    assert filtered_response.status_code == 200
    filtered_payload = filtered_response.json()
    assert filtered_payload["cached"] is False
    assert filtered_payload["calculated_at"] is None
    assert filtered_payload["applied_record_ids"] == ["tp-class-1", "fn-any-class"]
    assert filtered_payload["applied_record_count"] == 2
    assert filtered_payload["applied_class_thresholds"] is None
    assert filtered_payload["metrics"]["precision"] == pytest.approx(1.0)
    assert filtered_payload["metrics"]["recall"] == pytest.approx(0.5)
    assert filtered_payload["metrics"] != first_payload["metrics"]

    repeated_filtered_response = client.get(
        f"/api/v1/dataset_inference_jobs/{finished_job.id}/evaluation",
        params=filtered_params,
    )

    assert repeated_filtered_response.status_code == 200
    repeated_filtered_payload = repeated_filtered_response.json()
    assert repeated_filtered_payload["cached"] is False
    assert repeated_filtered_payload["calculated_at"] is None
    assert repeated_filtered_payload["metrics"] == filtered_payload["metrics"]

    cached_after_filtered_response = client.get(f"/api/v1/dataset_inference_jobs/{finished_job.id}/evaluation")

    assert cached_after_filtered_response.status_code == 200
    cached_after_filtered_payload = cached_after_filtered_response.json()
    assert cached_after_filtered_payload["cached"] is True
    assert cached_after_filtered_payload["metrics"] == first_payload["metrics"]


def test_evaluation_combines_record_ids_and_class_thresholds(
    client: TestClient,
    finished_job: DatasetInferenceJobStatus,
) -> None:
    response = client.get(
        f"/api/v1/dataset_inference_jobs/{finished_job.id}/evaluation",
        params={
            "record_ids": '["tp-class-1", "fp-class-1", "fn-any-class"]',
            "class_thresholds": '{"1": 0.5}',
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["cached"] is False
    assert payload["calculated_at"] is None
    assert payload["applied_record_ids"] == ["tp-class-1", "fp-class-1", "fn-any-class"]
    assert payload["applied_record_count"] == 3
    assert payload["applied_class_thresholds"] == {"1": 0.5}
    assert payload["metrics"]["precision"] == pytest.approx(1.0)
    assert payload["metrics"]["recall"] == pytest.approx(0.5)


def test_evaluation_accepts_large_filter_payload_as_json_body(
    client: TestClient,
    finished_job: DatasetInferenceJobStatus,
) -> None:
    response = client.post(
        f"/api/v1/dataset_inference_jobs/{finished_job.id}/evaluation",
        json={
            "record_ids": ["tp-class-1", "fn-any-class"],
            "class_thresholds": {"1": 0.5},
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["applied_record_count"] == 2
    assert payload["metrics"]["precision"] == pytest.approx(1.0)


def test_optimal_threshold_supports_predicted_class_scoping(client: TestClient, finished_job: DatasetInferenceJobStatus) -> None:
    class_one_response = client.get(
        f"/api/v1/dataset_inference_jobs/{finished_job.id}/optimal-threshold",
        params={"metric": "f1", "num_thresholds": 11, "class_id": 1},
    )

    assert class_one_response.status_code == 200
    class_one_payload = class_one_response.json()
    assert class_one_payload["class_id"] == 1
    assert class_one_payload["threshold"] == pytest.approx(0.3)
    assert class_one_payload["metric_value"] == pytest.approx(2.0 / 3.0)

    class_two_response = client.get(
        f"/api/v1/dataset_inference_jobs/{finished_job.id}/optimal-threshold",
        params={"metric": "f1", "num_thresholds": 11, "class_id": 2},
    )

    assert class_two_response.status_code == 200
    class_two_payload = class_two_response.json()
    assert class_two_payload["class_id"] == 2
    assert class_two_payload["threshold"] == pytest.approx(0.0)
    assert class_two_payload["metric_value"] == pytest.approx(2.0 / 3.0)


@pytest.mark.parametrize(
    ("params", "expected_detail"),
    [
        (
            {"class_thresholds": '{"999": 0.5}'},
            "Invalid class_thresholds: unknown class_id '999'.",
        ),
        (
            {"record_ids": '{"tp-class-1": true}'},
            "Invalid record_ids: expected a JSON array of record id strings.",
        ),
        (
            {"record_ids": '["tp-class-1", 1]'},
            "Invalid record_ids: every entry must be a string.",
        ),
        (
            {"record_ids": '["missing-record"]'},
            "Invalid record_ids: unknown record ids ['missing-record'].",
        ),
        (
            {"metric": "f1", "num_thresholds": 11, "class_id": 999},
            "Unknown class_id '999' for this job.",
        ),
    ],
)
def test_threshold_endpoints_validate_class_inputs(
    client: TestClient,
    finished_job: DatasetInferenceJobStatus,
    params: dict[str, int | str],
    expected_detail: str,
) -> None:
    route = "/evaluation" if {"class_thresholds", "record_ids"} & set(params) else "/optimal-threshold"

    response = client.get(f"/api/v1/dataset_inference_jobs/{finished_job.id}{route}", params=params)

    assert response.status_code == 422
    assert response.json()["detail"] == expected_detail
