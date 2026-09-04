# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

"""SQLite-backed cache of per-image inference results for semantic search.

Running a model over a large arbitrary folder can be expensive, and users are expected to
run several semantic searches (different query detections, different ``k``) against the
*same* folder. To avoid re-running inference every time, this module persists every
detection's embedding (plus its bbox/confidence/class) to a small SQLite database placed
at the root of the searched folder (``rfdetr_semantic_search_cache.db``). A later search
over the same folder with the same model can then skip inference entirely for any image
already present in the cache and only needs to recompute the cosine distance to the new
query embedding.

The cache is keyed by ``model_path`` so results from different models never mix, and it
records images with zero detections too (so they aren't mistaken for "not yet scanned").
"""

import json
import sqlite3
import threading
from pathlib import Path

from visualizer.backend.shared_types.prediction import Prediction

CACHE_FILENAME = "rfdetr_semantic_search_cache.db"


class SearchCache:
    """Wraps one ``rfdetr_semantic_search_cache.db`` file at the root of a searched folder.

    Args:
        folder: The folder that was (or will be) searched. The cache DB lives at
            ``folder / rfdetr_semantic_search_cache.db``.
        model_path: Path to the model checkpoint used for inference. Cached rows are keyed
            by this value so different models never share cached embeddings.
        model_type: The model type/registry key, stored alongside ``model_path`` purely as
            informational metadata (not used for cache invalidation).
    """

    def __init__(self, folder: Path, model_path: str, model_type: str) -> None:
        self.folder = folder
        self.db_path = folder / CACHE_FILENAME
        self.model_path = str(model_path)
        self.model_type = model_type
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
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS scanned_images (
                    model_path TEXT NOT NULL,
                    image_path TEXT NOT NULL,
                    model_type TEXT,
                    PRIMARY KEY (model_path, image_path)
                );

                CREATE TABLE IF NOT EXISTS detections (
                    model_path TEXT NOT NULL,
                    image_path TEXT NOT NULL,
                    detection_index INTEGER NOT NULL,
                    class_id   INTEGER NOT NULL,
                    confidence REAL NOT NULL,
                    bbox_x1 REAL,
                    bbox_y1 REAL,
                    bbox_x2 REAL,
                    bbox_y2 REAL,
                    embedding TEXT NOT NULL,
                    PRIMARY KEY (model_path, image_path, detection_index)
                );

                CREATE INDEX IF NOT EXISTS idx_detections_image
                    ON detections (model_path, image_path);
                """
            )

    def is_scanned(self, image_path: str) -> bool:
        """Return whether *image_path* was already run through ``self.model_path``."""
        with self._connect() as conn:
            row = conn.execute(
                "SELECT 1 FROM scanned_images WHERE model_path = ? AND image_path = ?",
                (self.model_path, image_path),
            ).fetchone()
        return row is not None

    def get_cached(self, image_path: str) -> list[tuple[Prediction, list[float]]]:
        """Return the cached ``(prediction, embedding)`` pairs for *image_path*.

        Only meaningful when :meth:`is_scanned` is ``True`` for the same path; returns an
        empty list both when the image hasn't been scanned yet and when it was scanned but
        had zero detections.
        """
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT class_id, confidence, bbox_x1, bbox_y1, bbox_x2, bbox_y2, embedding "
                "FROM detections WHERE model_path = ? AND image_path = ? "
                "ORDER BY detection_index",
                (self.model_path, image_path),
            ).fetchall()
        results = []
        for row in rows:
            bbox = None
            if row["bbox_x1"] is not None:
                bbox = (row["bbox_x1"], row["bbox_y1"], row["bbox_x2"], row["bbox_y2"])
            prediction = Prediction(class_id=row["class_id"], confidence=row["confidence"], bbox=bbox)
            embedding = json.loads(row["embedding"])
            results.append((prediction, embedding))
        return results

    def store(self, image_path: str, detections: list[tuple[Prediction, list[float]]]) -> None:
        """Persist *detections* (and mark *image_path* as scanned) for ``self.model_path``.

        Args:
            image_path: Path of the image that was just run through the model.
            detections: ``(prediction, embedding)`` pairs, one per detection found in the
                image (may be empty when the image has no detections).
        """
        with self._write_lock, self._connect() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO scanned_images (model_path, image_path, model_type) "
                "VALUES (?, ?, ?)",
                (self.model_path, image_path, self.model_type),
            )
            conn.execute(
                "DELETE FROM detections WHERE model_path = ? AND image_path = ?",
                (self.model_path, image_path),
            )
            conn.executemany(
                "INSERT INTO detections "
                "(model_path, image_path, detection_index, class_id, confidence, "
                " bbox_x1, bbox_y1, bbox_x2, bbox_y2, embedding) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                [
                    (
                        self.model_path,
                        image_path,
                        i,
                        pred.class_id,
                        pred.confidence,
                        *(pred.bbox if pred.bbox is not None else (None, None, None, None)),
                        json.dumps(embedding),
                    )
                    for i, (pred, embedding) in enumerate(detections)
                ],
            )


__all__ = ["SearchCache", "CACHE_FILENAME"]
