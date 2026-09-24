# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

"""SQLite-backed cache of per-image inference results for semantic search.

Running a model over a large arbitrary folder can be expensive, and users are expected to run several semantic searches
(different query detections, different ``k``) against the *same* folder. To avoid re-running inference every time, this
module persists every detection's embedding (plus its bbox/confidence/class) to a small SQLite database placed at the
root of the searched folder (``rfdetr_semantic_search_cache.db``). A later search over the same folder with the same
model can then skip inference entirely for any image already present in the cache and only needs to recompute the cosine
distance to the new query embedding.

The cache is namespaced by model and source preprocessing configuration, and records images with zero detections too (so
they aren't mistaken for "not yet scanned").
"""

import json
import sqlite3
import threading
from dataclasses import dataclass
from pathlib import Path

from visualizer.backend.shared_types.prediction import Prediction

CACHE_FILENAME = "rfdetr_semantic_search_cache.db"
_CREATE_TABLES_SQL = """
CREATE TABLE IF NOT EXISTS cache_groups (
    cache_key TEXT NOT NULL,
    group_key TEXT NOT NULL,
    file_size INTEGER NOT NULL,
    file_mtime_ns INTEGER NOT NULL,
    unit_count INTEGER NOT NULL,
    PRIMARY KEY (cache_key, group_key)
);

CREATE TABLE IF NOT EXISTS cache_units (
    cache_key TEXT NOT NULL,
    unit_id TEXT NOT NULL,
    group_key TEXT NOT NULL,
    PRIMARY KEY (cache_key, unit_id)
);

CREATE TABLE IF NOT EXISTS cache_detections (
    cache_key TEXT NOT NULL,
    unit_id TEXT NOT NULL,
    detection_index INTEGER NOT NULL,
    class_id INTEGER NOT NULL,
    confidence REAL NOT NULL,
    bbox_x1 REAL,
    bbox_y1 REAL,
    bbox_x2 REAL,
    bbox_y2 REAL,
    embedding TEXT NOT NULL,
    PRIMARY KEY (cache_key, unit_id, detection_index)
);

CREATE INDEX IF NOT EXISTS idx_cache_units_group
    ON cache_units (cache_key, group_key);

CREATE INDEX IF NOT EXISTS idx_cache_detections_unit
    ON cache_detections (cache_key, unit_id);
"""


@dataclass(frozen=True)
class CachedGroupManifest:
    """Metadata proving that one source image has a complete cached result."""

    file_size: int
    file_mtime_ns: int
    unit_count: int


class SearchCache:
    """Wraps one ``rfdetr_semantic_search_cache.db`` file at the root of a searched folder.

    Args:
        folder: The folder that was (or will be) searched. The cache DB lives at
            ``folder / rfdetr_semantic_search_cache.db``.
        model_path: Path to the model checkpoint used for inference. Cached rows are keyed
            by this value so different models never share cached embeddings.
        model_type: The model type/registry key, stored alongside ``model_path`` purely as
            informational metadata (not used for cache invalidation).
        source_signature: Stable identity of the source preprocessing configuration.
    """

    def __init__(self, folder: Path, model_path: str, model_type: str, source_signature: str) -> None:
        self.folder = folder
        self.db_path = folder / CACHE_FILENAME
        self.model_path = str(model_path)
        self.model_type = model_type
        self.source_signature = source_signature
        self.cache_key = json.dumps(
            {
                "model_path": self.model_path,
                "model_type": self.model_type,
                "source_signature": self.source_signature,
            },
            sort_keys=True,
            separators=(",", ":"),
        )
        self._write_lock = threading.Lock()
        self._create_tables()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        return conn

    def _create_tables(self) -> None:
        with self._write_lock, self._connect() as conn:
            conn.executescript(_CREATE_TABLES_SQL)

    def list_group_manifests(self) -> dict[str, CachedGroupManifest]:
        """Return complete group manifests for the current cache namespace."""
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT group_key, file_size, file_mtime_ns, unit_count FROM cache_groups WHERE cache_key = ?",
                (self.cache_key,),
            ).fetchall()
        return {
            row["group_key"]: CachedGroupManifest(
                file_size=row["file_size"],
                file_mtime_ns=row["file_mtime_ns"],
                unit_count=row["unit_count"],
            )
            for row in rows
        }

    def get_cached_group(self, group_key: str) -> list[tuple[str, list[tuple[Prediction, list[float]]]]]:
        """Load all cached units and detections for one complete group."""
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT
                    u.unit_id,
                    d.detection_index,
                    d.class_id,
                    d.confidence,
                    d.bbox_x1,
                    d.bbox_y1,
                    d.bbox_x2,
                    d.bbox_y2,
                    d.embedding
                FROM cache_units AS u
                LEFT JOIN cache_detections AS d
                    ON d.cache_key = u.cache_key AND d.unit_id = u.unit_id
                WHERE u.cache_key = ? AND u.group_key = ?
                ORDER BY u.unit_id, d.detection_index
                """,
                (self.cache_key, group_key),
            ).fetchall()

        units: list[tuple[str, list[tuple[Prediction, list[float]]]]] = []
        current_unit_id: str | None = None
        current_detections: list[tuple[Prediction, list[float]]] = []
        for row in rows:
            unit_id = row["unit_id"]
            if unit_id != current_unit_id:
                if current_unit_id is not None:
                    units.append((current_unit_id, current_detections))
                current_unit_id = unit_id
                current_detections = []
            if row["detection_index"] is not None:
                bbox = None
                if row["bbox_x1"] is not None:
                    bbox = (row["bbox_x1"], row["bbox_y1"], row["bbox_x2"], row["bbox_y2"])
                prediction = Prediction(
                    class_id=row["class_id"],
                    confidence=row["confidence"],
                    bbox=bbox,
                )
                current_detections.append((prediction, json.loads(row["embedding"])))
        if current_unit_id is not None:
            units.append((current_unit_id, current_detections))
        return units

    def invalidate_group(self, group_key: str) -> None:
        """Remove cached units and completeness metadata for one source image."""
        with self._write_lock, self._connect() as conn:
            unit_rows = conn.execute(
                "SELECT unit_id FROM cache_units WHERE cache_key = ? AND group_key = ?",
                (self.cache_key, group_key),
            ).fetchall()
            unit_ids = [row["unit_id"] for row in unit_rows]
            if unit_ids:
                conn.executemany(
                    "DELETE FROM cache_detections WHERE cache_key = ? AND unit_id = ?",
                    [(self.cache_key, unit_id) for unit_id in unit_ids],
                )
            conn.execute(
                "DELETE FROM cache_units WHERE cache_key = ? AND group_key = ?",
                (self.cache_key, group_key),
            )
            conn.execute(
                "DELETE FROM cache_groups WHERE cache_key = ? AND group_key = ?",
                (self.cache_key, group_key),
            )

    def store_unit(
        self,
        unit_id: str,
        group_key: str,
        detections: list[tuple[Prediction, list[float]]],
    ) -> None:
        """Persist one processed unit without marking its group complete."""
        with self._write_lock, self._connect() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO cache_units (cache_key, unit_id, group_key) VALUES (?, ?, ?)",
                (self.cache_key, unit_id, group_key),
            )
            conn.execute(
                "DELETE FROM cache_detections WHERE cache_key = ? AND unit_id = ?",
                (self.cache_key, unit_id),
            )
            conn.executemany(
                "INSERT INTO cache_detections "
                "(cache_key, unit_id, detection_index, class_id, confidence, "
                "bbox_x1, bbox_y1, bbox_x2, bbox_y2, embedding) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                [
                    (
                        self.cache_key,
                        unit_id,
                        index,
                        prediction.class_id,
                        prediction.confidence,
                        *(prediction.bbox if prediction.bbox is not None else (None, None, None, None)),
                        json.dumps(embedding),
                    )
                    for index, (prediction, embedding) in enumerate(detections)
                ],
            )

    def mark_group_complete(
        self,
        group_key: str,
        file_size: int,
        file_mtime_ns: int,
        unit_count: int,
    ) -> None:
        """Mark a group reusable after every expected unit has been stored."""
        with self._write_lock, self._connect() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO cache_groups "
                "(cache_key, group_key, file_size, file_mtime_ns, unit_count) "
                "VALUES (?, ?, ?, ?, ?)",
                (self.cache_key, group_key, file_size, file_mtime_ns, unit_count),
            )


__all__ = ["SearchCache", "CachedGroupManifest", "CACHE_FILENAME"]
