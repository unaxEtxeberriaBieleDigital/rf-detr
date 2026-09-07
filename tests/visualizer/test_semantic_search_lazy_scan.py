# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

"""Tests for unit counting and lazy (streaming) scanning in semantic search."""

from collections.abc import Iterator
from pathlib import Path

import pytest
from PIL import Image

from visualizer.backend.semantic_search.engine import _BATCH_SIZE, run_semantic_search
from visualizer.backend.semantic_search.sources.basesource import BaseSemanticSearchSource
from visualizer.backend.semantic_search.sources.default import DefaultImageSource
from visualizer.backend.semantic_search.sources.tiled import TiledImageSource
from visualizer.backend.semantic_search.types import ScanUnit, SearchJob
from visualizer.backend.shared_types.prediction import Prediction


class _StubModel:
    """Minimal stand-in for ``BaseModel``, carrying only what the engine/sources read."""

    def __init__(self, model_path: str = "stub-model.pth", input_shape: int = 40) -> None:
        self.model_path = model_path
        self.input_shape = input_shape


class _RecordingSource(BaseSemanticSearchSource):
    """Source that records how many units had been produced when each batch ran.

    ``inference_batch_marks`` stores, per ``process_batch`` call, the number of units the generator had yielded so far.
    An engine that materialises the whole scan up front would record the total unit count on its very first batch.
    """

    def __init__(self, num_units: int, detections_by_unit: dict[str, list] | None = None) -> None:
        self.num_units = num_units
        self.detections_by_unit = detections_by_unit or {}
        self.units_yielded = 0
        self.inference_batch_marks: list[int] = []

    def get_num_units(self, folder: Path, model=None) -> int:
        return self.num_units

    def iter_scan_units(self, folder: Path, model=None) -> Iterator[ScanUnit]:
        for index in range(self.num_units):
            self.units_yielded += 1
            unit_id = f"unit-{index}"
            yield ScanUnit(id=unit_id, group_key=f"group-{index}", inference_input=unit_id)

    def process_batch(self, model, batch: list[ScanUnit]) -> list[list]:
        self.inference_batch_marks.append(self.units_yielded)
        return [self.detections_by_unit.get(unit.id, []) for unit in batch]

    def render_result_preview(self, result):
        raise NotImplementedError


def _make_search_job(search_path: Path, k: int = 5) -> SearchJob:
    return SearchJob(
        id="search-1",
        parent_job_id="job-1",
        query_record_id="record-1",
        query_image_path=str(search_path / "query.jpg"),
        search_path=str(search_path),
        k=k,
    )


def _write_image(path: Path, size: tuple[int, int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", size, color=(10, 20, 30)).save(path)


class TestDefaultSourceUnitCount:
    """``DefaultImageSource.get_num_units`` counts one unit per supported image."""

    def test_counts_supported_images(self, tmp_path: Path) -> None:
        _write_image(tmp_path / "a.jpg", (32, 32))
        _write_image(tmp_path / "nested" / "b.png", (32, 32))

        assert DefaultImageSource().get_num_units(tmp_path) == 2

    def test_ignores_non_image_files(self, tmp_path: Path) -> None:
        _write_image(tmp_path / "a.jpg", (32, 32))
        (tmp_path / "notes.txt").write_text("not an image")

        assert DefaultImageSource().get_num_units(tmp_path) == 1

    def test_matches_iter_scan_units_length(self, tmp_path: Path) -> None:
        _write_image(tmp_path / "a.jpg", (32, 32))
        _write_image(tmp_path / "b.jpg", (32, 32))
        source = DefaultImageSource()

        assert source.get_num_units(tmp_path) == len(list(source.iter_scan_units(tmp_path)))


class TestTiledSourceUnitCount:
    """``TiledImageSource.get_num_units`` counts the tiles the scan would produce."""

    def test_counts_tiles_of_single_image(self, tmp_path: Path) -> None:
        # 300px -> resized to 100px; a 40px tile grid covers it in 3 steps per axis.
        _write_image(tmp_path / "big.jpg", (300, 300))

        assert TiledImageSource().get_num_units(tmp_path, _StubModel(input_shape=40)) == 9

    def test_matches_iter_scan_units_length(self, tmp_path: Path) -> None:
        _write_image(tmp_path / "big.jpg", (300, 300))
        _write_image(tmp_path / "small.jpg", (60, 90))
        source = TiledImageSource()
        model = _StubModel(input_shape=40)

        counted = source.get_num_units(tmp_path, model)

        assert counted == len(list(source.iter_scan_units(tmp_path, model)))

    def test_requires_a_model_to_know_the_tile_size(self, tmp_path: Path) -> None:
        _write_image(tmp_path / "big.jpg", (300, 300))

        with pytest.raises(ValueError):
            TiledImageSource().get_num_units(tmp_path, None)


class TestEngineLazyScanning:
    """The engine streams units instead of materialising the whole scan up front."""

    def test_runs_first_batch_before_consuming_every_unit(self, tmp_path: Path) -> None:
        num_units = _BATCH_SIZE * 3
        source = _RecordingSource(num_units=num_units)
        job = _make_search_job(tmp_path)

        run_semantic_search(job, _StubModel(), "stub", [1.0, 0.0], source)

        assert source.inference_batch_marks[0] == _BATCH_SIZE

    def test_consumes_units_incrementally_across_batches(self, tmp_path: Path) -> None:
        source = _RecordingSource(num_units=_BATCH_SIZE * 3)
        job = _make_search_job(tmp_path)

        run_semantic_search(job, _StubModel(), "stub", [1.0, 0.0], source)

        assert source.inference_batch_marks == [
            _BATCH_SIZE,
            _BATCH_SIZE * 2,
            _BATCH_SIZE * 3,
        ]

    def test_reports_total_units_from_get_num_units(self, tmp_path: Path) -> None:
        source = _RecordingSource(num_units=5)
        job = _make_search_job(tmp_path)

        run_semantic_search(job, _StubModel(), "stub", [1.0, 0.0], source)

        assert job.num_images_total == 5

    def test_processes_a_trailing_partial_batch(self, tmp_path: Path) -> None:
        source = _RecordingSource(num_units=_BATCH_SIZE + 3)
        job = _make_search_job(tmp_path)

        run_semantic_search(job, _StubModel(), "stub", [1.0, 0.0], source)

        assert job.num_images_processed == _BATCH_SIZE + 3

    def test_completes_successfully(self, tmp_path: Path) -> None:
        source = _RecordingSource(num_units=3)
        job = _make_search_job(tmp_path)

        run_semantic_search(job, _StubModel(), "stub", [1.0, 0.0], source)

        assert job.status == "done"

    def test_ranks_results_by_cosine_distance(self, tmp_path: Path) -> None:
        near = (Prediction(class_id=1, confidence=0.9, bbox=(0, 0, 5, 5)), [1.0, 0.0])
        far = (Prediction(class_id=1, confidence=0.9, bbox=(0, 0, 5, 5)), [0.0, 1.0])
        source = _RecordingSource(
            num_units=2,
            detections_by_unit={"unit-0": [far], "unit-1": [near]},
        )
        job = _make_search_job(tmp_path)

        run_semantic_search(job, _StubModel(), "stub", [1.0, 0.0], source)

        assert [r.unit_id for r in job.results] == ["unit-1", "unit-0"]
