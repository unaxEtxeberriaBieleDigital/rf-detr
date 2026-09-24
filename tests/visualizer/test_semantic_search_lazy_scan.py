# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

"""Tests for unit counting and lazy (streaming) scanning in semantic search."""

from collections.abc import Iterator
from pathlib import Path
from unittest.mock import patch

import pytest
from PIL import Image

from visualizer.backend.semantic_search.cache import SearchCache
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

    def iter_group_paths(self, folder: Path) -> Iterator[Path]:
        yield folder

    def get_num_units_for_group(self, path: Path, model=None) -> int:
        return self.num_units

    def iter_scan_units_for_group(self, path: Path, model=None) -> Iterator[ScanUnit]:
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


def _cache_group(
    folder: Path,
    model: _StubModel,
    source: BaseSemanticSearchSource,
    image_path: Path,
    units: list[tuple[str, list[tuple[Prediction, list[float]]]]],
) -> None:
    """Populate one complete cache group for an unchanged source image."""
    cache = SearchCache(
        folder,
        model_path=str(model.model_path),
        model_type="stub",
        source_signature=source.cache_signature(model),
    )
    stat = image_path.stat()
    group_key = str(image_path)
    cache.invalidate_group(group_key)
    for unit_id, detections in units:
        cache.store_unit(unit_id, group_key, detections)
    cache.mark_group_complete(group_key, stat.st_size, stat.st_mtime_ns, len(units))


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


class TestEngineCacheFirstScanning:
    """Complete cache groups are ranked without materialising their source images."""

    def test_completed_inference_is_reused_by_next_search(self, tmp_path: Path) -> None:
        image_path = tmp_path / "new.jpg"
        _write_image(image_path, (32, 32))
        model = _StubModel()
        source = DefaultImageSource()
        detection = (Prediction(class_id=1, confidence=0.9, bbox=(0, 0, 5, 5)), [1.0, 0.0])
        first_job = _make_search_job(tmp_path)

        with patch.object(source, "process_batch", return_value=[[detection]]) as inference:
            run_semantic_search(first_job, model, "stub", [1.0, 0.0], source)

        second_job = _make_search_job(tmp_path)
        with patch.object(
            source,
            "process_batch",
            side_effect=AssertionError("completed inference should be cached"),
        ):
            run_semantic_search(second_job, model, "stub", [1.0, 0.0], source)

        inference.assert_called_once()
        assert second_job.status == "done"
        assert [result.unit_id for result in second_job.results] == [str(image_path)]

    @pytest.mark.parametrize(
        ("source", "unit_id"),
        [
            pytest.param(DefaultImageSource(), "cached.jpg", id="default"),
            pytest.param(TiledImageSource(), "cached.jpg::tile_0_0_40_40", id="tiled"),
        ],
    )
    def test_complete_cache_hit_does_not_open_image_or_run_inference(
        self,
        tmp_path: Path,
        source: BaseSemanticSearchSource,
        unit_id: str,
    ) -> None:
        image_path = tmp_path / "cached.jpg"
        _write_image(image_path, (120, 120))
        model = _StubModel()
        resolved_unit_id = str(tmp_path / unit_id)
        detection = (Prediction(class_id=1, confidence=0.9, bbox=(0, 0, 5, 5)), [1.0, 0.0])
        _cache_group(tmp_path, model, source, image_path, [(resolved_unit_id, [detection])])
        job = _make_search_job(tmp_path)

        with (
            patch(
                "visualizer.backend.semantic_search.sources.tiled.Image.open",
                side_effect=AssertionError("cached images must not be opened"),
            ),
            patch.object(
                source,
                "process_batch",
                side_effect=AssertionError("cached images must not run inference"),
            ),
        ):
            run_semantic_search(job, model, "stub", [1.0, 0.0], source)

        assert job.status == "done"
        assert job.num_images_processed == 1
        assert [result.unit_id for result in job.results] == [resolved_unit_id]

    def test_only_modified_image_is_reprocessed(self, tmp_path: Path) -> None:
        unchanged_path = tmp_path / "unchanged.jpg"
        changed_path = tmp_path / "changed.jpg"
        _write_image(unchanged_path, (32, 32))
        _write_image(changed_path, (32, 32))
        model = _StubModel()
        source = DefaultImageSource()
        cached_detection = (Prediction(class_id=1, confidence=0.9, bbox=(0, 0, 5, 5)), [1.0, 0.0])
        _cache_group(
            tmp_path,
            model,
            source,
            unchanged_path,
            [(str(unchanged_path), [cached_detection])],
        )
        _cache_group(
            tmp_path,
            model,
            source,
            changed_path,
            [(str(changed_path), [cached_detection])],
        )
        changed_path.write_bytes(changed_path.read_bytes() + b"changed")
        processed_paths: list[str] = []

        def process_batch(model, batch: list[ScanUnit]) -> list[list]:
            processed_paths.extend(str(unit.inference_input) for unit in batch)
            return [[cached_detection] for _ in batch]

        job = _make_search_job(tmp_path)
        with patch.object(source, "process_batch", side_effect=process_batch):
            run_semantic_search(job, model, "stub", [1.0, 0.0], source)

        assert processed_paths == [str(changed_path)]

    def test_deleted_cached_image_is_excluded(self, tmp_path: Path) -> None:
        existing_path = tmp_path / "existing.jpg"
        deleted_path = tmp_path / "deleted.jpg"
        _write_image(existing_path, (32, 32))
        _write_image(deleted_path, (32, 32))
        model = _StubModel()
        source = DefaultImageSource()
        detection = (Prediction(class_id=1, confidence=0.9, bbox=(0, 0, 5, 5)), [1.0, 0.0])
        _cache_group(tmp_path, model, source, existing_path, [(str(existing_path), [detection])])
        _cache_group(tmp_path, model, source, deleted_path, [(str(deleted_path), [detection])])
        deleted_path.unlink()
        job = _make_search_job(tmp_path)

        with patch.object(
            source,
            "process_batch",
            side_effect=AssertionError("remaining image should be served from cache"),
        ):
            run_semantic_search(job, model, "stub", [1.0, 0.0], source)

        assert job.num_images_total == 1
        assert [result.image_path for result in job.results] == [str(existing_path)]

    def test_incomplete_group_is_not_reused(self, tmp_path: Path) -> None:
        image_path = tmp_path / "incomplete.jpg"
        _write_image(image_path, (32, 32))
        model = _StubModel()
        source = DefaultImageSource()
        detection = (Prediction(class_id=1, confidence=0.9, bbox=(0, 0, 5, 5)), [1.0, 0.0])
        cache = SearchCache(
            tmp_path,
            model_path=str(model.model_path),
            model_type="stub",
            source_signature=source.cache_signature(model),
        )
        cache.store_unit(str(image_path), str(image_path), [detection])
        processed_paths: list[str] = []

        def process_batch(model, batch: list[ScanUnit]) -> list[list]:
            processed_paths.extend(str(unit.inference_input) for unit in batch)
            return [[detection] for _ in batch]

        job = _make_search_job(tmp_path)
        with patch.object(source, "process_batch", side_effect=process_batch):
            run_semantic_search(job, model, "stub", [1.0, 0.0], source)

        assert processed_paths == [str(image_path)]

    def test_tiled_configuration_change_invalidates_group(self, tmp_path: Path) -> None:
        image_path = tmp_path / "tiled.jpg"
        _write_image(image_path, (120, 120))
        source = TiledImageSource()
        cached_model = _StubModel(input_shape=40)
        detection = (Prediction(class_id=1, confidence=0.9, bbox=(0, 0, 5, 5)), [1.0, 0.0])
        _cache_group(
            tmp_path,
            cached_model,
            source,
            image_path,
            [(f"{image_path}::tile_0_0_40_40", [detection])],
        )
        changed_model = _StubModel(input_shape=60)
        processed_units: list[str] = []

        def process_batch(model, batch: list[ScanUnit]) -> list[list]:
            processed_units.extend(unit.id for unit in batch)
            return [[detection] for _ in batch]

        job = _make_search_job(tmp_path)
        with patch.object(source, "process_batch", side_effect=process_batch):
            run_semantic_search(job, changed_model, "stub", [1.0, 0.0], source)

        assert processed_units
