# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

"""SQLite-backed persistent store for a single visualizer job.

One :class:`JobStore` maps to one ``rfdetr_visualizer.db`` file placed at the
root of the dataset directory.  Raw 512-D embeddings are kept permanently so
that dimensionality reduction can be (re-)computed at any time without
re-running inference.

Typical lifecycle
-----------------
1. ``JobStore(dataset_path)``      – opens (or creates) the DB
2. ``store.create_tables()``       – idempotent schema setup (called by jobs.py)
3. ``store.insert_records(batch)`` – called once per inference batch
4. ``store.compute_reduction(2)``  – on-demand, updates reduced-coordinates column
5. API endpoints call ``store.get_records(…)`` etc. for every request
"""

import json
import sqlite3
import threading
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.decomposition import IncrementalPCA

from rfdetr.utilities.logger import get_logger
from visualizer.backend.embeddingrecord import EmbeddingRecord

logger = get_logger()

DB_FILENAME = "rfdetr_visualizer.db"
PROGRESS_SCHEMA_VERSION = 1

# How many records are loaded from DB at once during IncrementalPCA fitting.
_PCA_BATCH_SIZE = 10_000


class JobStore:
    """Wraps a single SQLite database file for one visualizer job.

    Thread-safety: SQLite connections are not shareable across threads by
    default.  We open a *new* connection per public method call using
    ``check_same_thread=False`` and protect the write path with a lock so that
    the background inference thread and the FastAPI request threads don't race.

    Args:
        dataset_path: Root directory of the dataset.  The DB file is created
            at ``dataset_path / rfdetr_visualizer.db``.
    """

    def __init__(self, dataset_path: str | Path) -> None:
        self.dataset_path = Path(dataset_path)
        self.db_path: Path = self.dataset_path / DB_FILENAME
        self._write_lock = threading.Lock()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        conn.execute("PRAGMA foreign_keys=ON")
        return conn

    # ------------------------------------------------------------------
    # Schema
    # ------------------------------------------------------------------

    def create_tables(self) -> None:
        """Create tables and indices if they don't already exist (idempotent)."""
        with self._connect() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS job_meta (
                    key   TEXT PRIMARY KEY,
                    value TEXT
                );

                CREATE TABLE IF NOT EXISTS records (
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

                CREATE TABLE IF NOT EXISTS processed_images (
                    image_path    TEXT PRIMARY KEY,
                    split         TEXT NOT NULL,
                    processed_at  TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
                );

                CREATE TABLE IF NOT EXISTS evaluation_cache (
                    job_id         TEXT NOT NULL,
                    dataset_type   TEXT NOT NULL,
                    metric_name    TEXT NOT NULL,
                    metric_value   TEXT NOT NULL,
                    calculated_at  TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    PRIMARY KEY (job_id, dataset_type, metric_name)
                );

                CREATE INDEX IF NOT EXISTS idx_records_image_path
                    ON records (image_path);
                CREATE INDEX IF NOT EXISTS idx_records_split
                    ON records (split);
                CREATE INDEX IF NOT EXISTS idx_records_status
                    ON records (status);
                CREATE INDEX IF NOT EXISTS idx_records_split_status
                    ON records (split, status);
                CREATE INDEX IF NOT EXISTS idx_processed_images_split
                    ON processed_images (split);
                CREATE INDEX IF NOT EXISTS idx_evaluation_cache_lookup
                    ON evaluation_cache (job_id, dataset_type);
                """
            )

    # ------------------------------------------------------------------
    # Metadata
    # ------------------------------------------------------------------

    def set_meta(self, key: str, value: Any) -> None:
        """Upsert a metadata key-value pair (value is JSON-serialised).

        Args:
            key: Metadata key string.
            value: Any JSON-serialisable value.
        """
        with self._write_lock, self._connect() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO job_meta (key, value) VALUES (?, ?)",
                (key, json.dumps(value)),
            )

    def get_meta(self, key: str, default: Any = None) -> Any:
        """Return the deserialised value for *key*, or *default* if absent.

        Args:
            key: Metadata key string.
            default: Value to return when the key does not exist.

        Returns:
            Deserialised JSON value, or *default*.
        """
        with self._connect() as conn:
            row = conn.execute(
                "SELECT value FROM job_meta WHERE key = ?", (key,)
            ).fetchone()
        return json.loads(row["value"]) if row else default

    def set_run_config(self, config: dict[str, Any]) -> None:
        """Persist the normalized run configuration used for resume validation.

        Args:
            config: JSON-serializable run configuration.
        """
        self.set_meta("run_config", config)

    def get_run_config(self) -> dict[str, Any] | None:
        """Return the persisted run configuration, if available."""
        config = self.get_meta("run_config")
        return config if isinstance(config, dict) else None

    def enable_progress_tracking(self) -> None:
        """Mark this DB as supporting resumable per-image progress."""
        self.set_meta("progress_schema_version", PROGRESS_SCHEMA_VERSION)

    def has_progress_tracking(self) -> bool:
        """Return True when this DB supports resumable per-image progress."""
        return (
            self.get_meta("progress_schema_version") == PROGRESS_SCHEMA_VERSION
            and self.get_run_config() is not None
            and self._table_exists("processed_images")
        )

    def cache_metrics(
        self,
        job_id: str,
        dataset_type: str,
        metrics: dict[str, Any],
    ) -> None:
        """Cache computed evaluation metrics for a job/dataset pair.

        Args:
            job_id: In-memory job identifier used by API.
            dataset_type: Dataset registry key (e.g., ``coco_detection``).
            metrics: Mapping metric_name -> JSON-serializable metric value.
        """
        rows = [
            (job_id, dataset_type, metric_name, json.dumps(metric_value))
            for metric_name, metric_value in metrics.items()
        ]
        with self._write_lock, self._connect() as conn:
            conn.execute(
                "DELETE FROM evaluation_cache WHERE job_id = ? AND dataset_type = ?",
                (job_id, dataset_type),
            )
            if rows:
                conn.executemany(
                    """
                    INSERT INTO evaluation_cache (job_id, dataset_type, metric_name, metric_value)
                    VALUES (?, ?, ?, ?)
                    """,
                    rows,
                )

    def get_cached_metrics(
        self,
        job_id: str,
        dataset_type: str,
    ) -> tuple[dict[str, Any], str | None] | None:
        """Get cached metrics for a job/dataset pair.

        Args:
            job_id: In-memory job identifier used by API.
            dataset_type: Dataset registry key (e.g., ``coco_detection``).

        Returns:
            Tuple of ``(metrics, calculated_at)`` when cache exists; otherwise ``None``.
        """
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT metric_name, metric_value, calculated_at
                FROM evaluation_cache
                WHERE job_id = ? AND dataset_type = ?
                ORDER BY metric_name
                """,
                (job_id, dataset_type),
            ).fetchall()

        if not rows:
            return None

        metrics: dict[str, Any] = {}
        calculated_at = rows[0]["calculated_at"]
        for row in rows:
            metrics[row["metric_name"]] = json.loads(row["metric_value"])
        return metrics, calculated_at

    def invalidate_metrics_cache(
        self,
        job_id: str,
        dataset_type: str | None = None,
    ) -> None:
        """Delete cached evaluation metrics.

        Args:
            job_id: In-memory job identifier used by API.
            dataset_type: Optional dataset type; when omitted removes all cache rows for the job.
        """
        with self._write_lock, self._connect() as conn:
            if dataset_type is None:
                conn.execute("DELETE FROM evaluation_cache WHERE job_id = ?", (job_id,))
            else:
                conn.execute(
                    "DELETE FROM evaluation_cache WHERE job_id = ? AND dataset_type = ?",
                    (job_id, dataset_type),
                )

    def reset_for_rerun(self) -> None:
        """Clear previous inference artifacts to safely rerun on same dataset/model.

        This removes record rows, dimensionality-reduction outputs, and evaluation cache,
        while keeping stable metadata keys such as dataset path/type and categories.
        """
        with self._write_lock, self._connect() as conn:
            conn.execute("DELETE FROM records")
            conn.execute("DELETE FROM processed_images")
            conn.execute("DELETE FROM evaluation_cache")

    # ------------------------------------------------------------------
    # Write
    # ------------------------------------------------------------------

    def insert_records(self, records: list[EmbeddingRecord]) -> None:
        """Bulk-insert a batch of :class:`EmbeddingRecord` objects.

        Existing rows with the same ``id`` are replaced (supports re-runs on
        the same DB without leaving stale data).

        Args:
            records: Batch of records produced by a single image batch.
        """
        rows = []
        for r in records:
            pred = r.prediction
            gt = r.ground_truth
            rows.append(
                (
                    r.id,
                    r.image_path,
                    r.split,
                    r.status,
                    pred.class_id if pred else None,
                    pred.confidence if pred else None,
                    pred.bbox[0] if pred and pred.bbox else None,
                    pred.bbox[1] if pred and pred.bbox else None,
                    pred.bbox[2] if pred and pred.bbox else None,
                    pred.bbox[3] if pred and pred.bbox else None,
                    gt.class_id if gt else None,
                    gt.confidence if gt else None,
                    gt.bbox[0] if gt and gt.bbox else None,
                    gt.bbox[1] if gt and gt.bbox else None,
                    gt.bbox[2] if gt and gt.bbox else None,
                    gt.bbox[3] if gt and gt.bbox else None,
                    json.dumps(r.embedding) if r.embedding else None,
                )
            )
        with self._write_lock, self._connect() as conn:
            conn.executemany(
                """
                INSERT OR REPLACE INTO records
                (id, image_path, split, status,
                 pred_class_id, pred_confidence,
                 pred_x1, pred_y1, pred_x2, pred_y2,
                 gt_class_id, gt_confidence,
                 gt_x1, gt_y1, gt_x2, gt_y2,
                 raw_embedding)
                VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                """,
                rows,
            )

    def persist_batch(
        self,
        records: list[EmbeddingRecord],
        processed_images: list[tuple[str, str]],
        num_images_processed: int,
    ) -> None:
        """Atomically persist one processed image batch.

        Record rows, per-image progress rows, and the persisted processed-image counter
        are committed in one SQLite transaction so resume never sees a half-written batch.

        Args:
            records: Batch record rows to insert.
            processed_images: ``(image_path, split)`` tuples for processed images.
            num_images_processed: Persisted cumulative processed-image count after this batch.
        """
        record_rows = []
        for record in records:
            prediction = record.prediction
            ground_truth = record.ground_truth
            record_rows.append(
                (
                    record.id,
                    record.image_path,
                    record.split,
                    record.status,
                    prediction.class_id if prediction else None,
                    prediction.confidence if prediction else None,
                    prediction.bbox[0] if prediction and prediction.bbox else None,
                    prediction.bbox[1] if prediction and prediction.bbox else None,
                    prediction.bbox[2] if prediction and prediction.bbox else None,
                    prediction.bbox[3] if prediction and prediction.bbox else None,
                    ground_truth.class_id if ground_truth else None,
                    ground_truth.confidence if ground_truth else None,
                    ground_truth.bbox[0] if ground_truth and ground_truth.bbox else None,
                    ground_truth.bbox[1] if ground_truth and ground_truth.bbox else None,
                    ground_truth.bbox[2] if ground_truth and ground_truth.bbox else None,
                    ground_truth.bbox[3] if ground_truth and ground_truth.bbox else None,
                    json.dumps(record.embedding) if record.embedding else None,
                )
            )

        with self._write_lock, self._connect() as conn:
            if record_rows:
                conn.executemany(
                    """
                    INSERT OR REPLACE INTO records
                    (id, image_path, split, status,
                     pred_class_id, pred_confidence,
                     pred_x1, pred_y1, pred_x2, pred_y2,
                     gt_class_id, gt_confidence,
                     gt_x1, gt_y1, gt_x2, gt_y2,
                     raw_embedding)
                    VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                    """,
                    record_rows,
                )

            if processed_images:
                conn.executemany(
                    """
                    INSERT OR REPLACE INTO processed_images (image_path, split)
                    VALUES (?, ?)
                    """,
                    processed_images,
                )

            conn.execute(
                "INSERT OR REPLACE INTO job_meta (key, value) VALUES (?, ?)",
                ("num_images_processed", json.dumps(num_images_processed)),
            )

    # ------------------------------------------------------------------
    # Dimensionality reduction (on-demand) — PCA, t-SNE, UMAP
    # ------------------------------------------------------------------

    def compute_reduction(
        self,
        components: int,
        algorithm: str = "pca",
        record_ids: list[str] | None = None,
        perplexity: float = 30.0,
        n_neighbors: int = 15,
        min_dist: float = 0.1,
    ) -> int:
        """Dispatch to the requested dimensionality-reduction algorithm.

        **Scalability notes:**

        * PCA  – Incremental; memory-safe for any dataset size.
        * t-SNE – All embeddings loaded into RAM at once.  Practical up to
          ~100 k records (~200 MB for float32 @512D).  Barnes-Hut
          approximation keeps time complexity at *O(n log n)*, but for very
          large subsets it can still take minutes.  Prefer filtering to a
          meaningful subset before running.
        * UMAP  – Also batch but typically 5-10x faster than t-SNE and
          better-suited for large datasets (up to ~500 k in practice on
          CPU).  Requires the optional ``umap-learn`` package.

        Args:
            components: Number of output dimensions (2 or 3).
            algorithm: ``"pca"``, ``"tsne"``, or ``"umap"``.
            record_ids: Optional subset of record IDs; ``None`` = all records.
            perplexity: t-SNE perplexity (ignored for PCA/UMAP).
            n_neighbors: UMAP n_neighbors (ignored for PCA/t-SNE).
            min_dist: UMAP min_dist (ignored for PCA/t-SNE).

        Returns:
            Number of records updated.
        """
        if algorithm == "pca":
            return self._compute_pca(components, record_ids=record_ids)
        if algorithm == "tsne":
            return self._compute_tsne(components, record_ids=record_ids, perplexity=perplexity)
        if algorithm == "umap":
            return self._compute_umap(
                components, record_ids=record_ids, n_neighbors=n_neighbors, min_dist=min_dist
            )
        raise ValueError(f"Unknown algorithm {algorithm!r}. Choose 'pca', 'tsne', or 'umap'.")

    def _load_all_embeddings(
        self, record_ids: list[str] | None = None
    ) -> tuple[list[str], np.ndarray]:
        """Load all raw embeddings (or a filtered subset) into a numpy array.

        Batch algorithms (t-SNE, UMAP) need all data in memory.  For large
        subsets this may use significant RAM (512 floats × n rows × 4 bytes).

        To avoid SQLite's 999-variable limit on ``IN (…)`` clauses, we always
        load everything and filter in Python when *record_ids* is provided.

        Args:
            record_ids: Optional list of IDs to restrict to.

        Returns:
            Tuple of (list of record ids, float32 matrix of shape (n, 512)).
        """
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT id, raw_embedding FROM records WHERE raw_embedding IS NOT NULL"
            ).fetchall()
        if record_ids is not None:
            id_set = set(record_ids)
            rows = [r for r in rows if r["id"] in id_set]
        ids = [r["id"] for r in rows]
        vecs = np.array([json.loads(r["raw_embedding"]) for r in rows], dtype=np.float32)
        return ids, vecs

    def _save_reduction_coords(
        self, ids: list[str], coords: np.ndarray, n_components: int
    ) -> int:
        """Persist reduced coordinates to the ``pca_embedding`` column.

        Args:
            ids: Record IDs in the same order as *coords* rows.
            coords: Reduced coordinate matrix, shape (n, n_components).
            n_components: Number of dimensions stored.

        Returns:
            Number of rows updated.
        """
        rows = [
            (json.dumps(coord.tolist()), n_components, rid)
            for rid, coord in zip(ids, coords)
        ]
        with self._write_lock, self._connect() as conn:
            conn.executemany(
                "UPDATE records SET pca_embedding=?, pca_components=? WHERE id=?",
                rows,
            )
        return len(rows)

    def _compute_tsne(
        self,
        components: int,
        record_ids: list[str] | None = None,
        perplexity: float = 30.0,
    ) -> int:
        """Run scikit-learn t-SNE on the (filtered) raw embeddings.

        Uses Barnes-Hut approximation (default in sklearn) for O(n log n)
        scaling.  Loads all embeddings into RAM; see :meth:`compute_reduction`
        for scalability notes.

        Args:
            components: Output dimensions (2 or 3).
            record_ids: Optional subset of record IDs.
            perplexity: t-SNE perplexity; clamped to ``(n-1)/3`` automatically.

        Returns:
            Number of records updated.
        """
        from sklearn.manifold import TSNE  # soft import – sklearn is always present

        label = f"subset of {len(record_ids)}" if record_ids is not None else "all"
        logger.info(
            f"[store] starting t-SNE (components={components}, "
            f"perplexity={perplexity}, records={label}) ..."
        )
        ids, vecs = self._load_all_embeddings(record_ids)
        if len(ids) == 0:
            logger.warning("[store] no embeddings found – t-SNE skipped")
            return 0

        n_components = min(components, len(ids))
        effective_perplexity = min(perplexity, max(1.0, (len(ids) - 1) / 3.0))
        tsne = TSNE(
            n_components=n_components,
            perplexity=effective_perplexity,
            random_state=42,
            n_jobs=-1,
        )
        coords = tsne.fit_transform(vecs)
        updated = self._save_reduction_coords(ids, coords, n_components)
        logger.info(f"[store] t-SNE done – {updated} records updated")
        return updated

    def _compute_umap(
        self,
        components: int,
        record_ids: list[str] | None = None,
        n_neighbors: int = 15,
        min_dist: float = 0.1,
    ) -> int:
        """Run UMAP on the (filtered) raw embeddings.

        Requires the optional ``umap-learn`` package
        (``pip install umap-learn``).  Loads all embeddings into RAM; see
        :meth:`compute_reduction` for scalability notes.

        Args:
            components: Output dimensions (2 or 3).
            record_ids: Optional subset of record IDs.
            n_neighbors: UMAP n_neighbors hyperparameter.
            min_dist: UMAP min_dist hyperparameter.

        Returns:
            Number of records updated.
        """
        try:
            import umap as umap_lib  # type: ignore[import-untyped]
        except ImportError as exc:
            raise RuntimeError(
                "umap-learn is required for UMAP. "
                "Install it with: pip install umap-learn"
            ) from exc

        label = f"subset of {len(record_ids)}" if record_ids is not None else "all"
        logger.info(
            f"[store] starting UMAP (components={components}, "
            f"n_neighbors={n_neighbors}, min_dist={min_dist}, records={label}) ..."
        )
        ids, vecs = self._load_all_embeddings(record_ids)
        if len(ids) == 0:
            logger.warning("[store] no embeddings found – UMAP skipped")
            return 0

        n_components = min(components, len(ids))
        effective_n_neighbors = min(n_neighbors, max(1, len(ids) - 1))
        reducer = umap_lib.UMAP(
            n_components=n_components,
            n_neighbors=effective_n_neighbors,
            min_dist=min_dist,
            random_state=42,
        )
        coords = reducer.fit_transform(vecs)
        updated = self._save_reduction_coords(ids, coords, n_components)
        logger.info(f"[store] UMAP done – {updated} records updated")
        return updated

    # ------------------------------------------------------------------
    # IncrementalPCA (memory-safe, supports any dataset size)
    # ------------------------------------------------------------------

    def _compute_pca(self, components: int, record_ids: list[str] | None = None) -> int:
        """Fit IncrementalPCA over a set of raw embeddings and persist coords.

        When *record_ids* is provided, only those records are used for both
        fitting and transformation, allowing PCA to be computed on a filtered
        subset (e.g. a single class or split).  All other records keep whatever
        ``pca_embedding`` they had before.

        Raw embeddings are read from the DB in batches of :data:`_PCA_BATCH_SIZE`
        so memory usage stays bounded regardless of dataset size.

        Args:
            components: Number of PCA dimensions (2 or 3 are typical).
            record_ids: Optional list of record IDs to restrict PCA to.  Pass
                ``None`` (default) to use all records with a raw embedding.

        Returns:
            Number of records updated with new PCA coordinates.
        """
        id_filter = ""
        id_params: list = []
        if record_ids is not None:
            if not record_ids:
                logger.warning("[store] record_ids list is empty – PCA skipped")
                return 0
            placeholders = ",".join("?" * len(record_ids))
            id_filter = f" AND id IN ({placeholders})"  # noqa: S608
            id_params = list(record_ids)

        label = f"subset of {len(record_ids)}" if record_ids is not None else "all"
        logger.info(f"[store] starting IncrementalPCA (components={components}, records={label}) ...")

        with self._connect() as conn:
            total = conn.execute(
                f"SELECT COUNT(*) FROM records WHERE raw_embedding IS NOT NULL{id_filter}",  # noqa: S608
                id_params,
            ).fetchone()[0]

        if total == 0:
            logger.warning("[store] no raw embeddings found for the given subset – PCA skipped")
            return 0

        n_components = min(components, total)
        ipca = IncrementalPCA(n_components=n_components)

        # --- Pass 1: partial_fit ------------------------------------------
        offset = 0
        while offset < total:
            ids, vecs = self._load_raw_batch(offset, _PCA_BATCH_SIZE, id_filter, id_params)
            if not ids:
                break
            ipca.partial_fit(np.array(vecs, dtype=np.float32))
            offset += len(ids)
            logger.info(f"[store] PCA fit pass: {min(offset, total)}/{total}")

        # --- Pass 2: transform + update -----------------------------------
        offset = 0
        updated = 0
        while offset < total:
            ids, vecs = self._load_raw_batch(offset, _PCA_BATCH_SIZE, id_filter, id_params)
            if not ids:
                break
            coords = ipca.transform(np.array(vecs, dtype=np.float32))
            rows = [
                (json.dumps(coord.tolist()), n_components, rid)
                for rid, coord in zip(ids, coords)
            ]
            with self._write_lock, self._connect() as conn:
                conn.executemany(
                    "UPDATE records SET pca_embedding=?, pca_components=? WHERE id=?",
                    rows,
                )
            updated += len(rows)
            offset += len(ids)
            logger.info(f"[store] PCA transform: {min(offset, total)}/{total}")

        logger.info(f"[store] IncrementalPCA done – {updated} records updated")
        return updated

    def _load_raw_batch(
        self,
        offset: int,
        limit: int,
        id_filter: str = "",
        id_params: list | None = None,
    ) -> tuple[list[str], list[list[float]]]:
        """Load a page of raw embeddings from the DB.

        Args:
            offset: Row offset into the raw-embedding subset.
            limit: Maximum rows to return.
            id_filter: Optional SQL AND clause to restrict to a record id subset.
            id_params: Positional params for *id_filter*.

        Returns:
            Tuple of (list of record ids, list of embedding vectors).
        """
        params: list = list(id_params or [])
        params += [limit, offset]
        with self._connect() as conn:
            rows = conn.execute(
                f"SELECT id, raw_embedding FROM records "  # noqa: S608
                f"WHERE raw_embedding IS NOT NULL{id_filter} "
                "LIMIT ? OFFSET ?",
                params,
            ).fetchall()
        ids = [r["id"] for r in rows]
        vecs = [json.loads(r["raw_embedding"]) for r in rows]
        return ids, vecs

    # ------------------------------------------------------------------
    # Queries (used by API endpoints)
    # ------------------------------------------------------------------

    def record_count(self) -> int:
        """Return total number of records stored.

        Returns:
            Integer row count.
        """
        with self._connect() as conn:
            return conn.execute("SELECT COUNT(*) FROM records").fetchone()[0]

    def has_dimensionality_reduction(self) -> bool:
        """Return True if at least one record has reduced coordinates.

        Returns:
            Boolean indicating reduction availability.
        """
        with self._connect() as conn:
            n = conn.execute(
                "SELECT COUNT(*) FROM records WHERE pca_embedding IS NOT NULL"
            ).fetchone()[0]
        return n > 0

    def dimensionality_reduction_components(self) -> int | None:
        """Return the number of reduced dimensions currently stored, or None.

        Returns:
            Integer dimension count, or None if no reduction has been computed.
        """
        with self._connect() as conn:
            row = conn.execute(
                "SELECT pca_components FROM records "
                "WHERE pca_components IS NOT NULL LIMIT 1"
            ).fetchone()
        return row[0] if row else None

    def processed_image_count(self, split_names: list[str] | None = None) -> int:
        """Return how many images have been marked processed.

        Args:
            split_names: Optional split-name filter.

        Returns:
            Number of processed images tracked in the DB.
        """
        if not self._table_exists("processed_images"):
            return 0

        sql = "SELECT COUNT(*) FROM processed_images"
        params: list[Any] = []
        if split_names:
            placeholders = ",".join("?" * len(split_names))
            sql += f" WHERE split IN ({placeholders})"  # noqa: S608
            params.extend(split_names)

        with self._connect() as conn:
            return int(conn.execute(sql, params).fetchone()[0])

    def get_processed_image_paths(self, split_names: list[str] | None = None) -> set[str]:
        """Return the set of already-processed image paths.

        Args:
            split_names: Optional split-name filter.

        Returns:
            Set of persisted image-path strings.
        """
        if not self._table_exists("processed_images"):
            return set()

        sql = "SELECT image_path FROM processed_images"
        params: list[Any] = []
        if split_names:
            placeholders = ",".join("?" * len(split_names))
            sql += f" WHERE split IN ({placeholders})"  # noqa: S608
            params.extend(split_names)

        with self._connect() as conn:
            rows = conn.execute(sql, params).fetchall()
        return {str(row["image_path"]) for row in rows}

    def get_records(
        self,
        split: str | None = None,
        status: str | None = None,
        class_id: int | None = None,
        limit: int = 200,
        offset: int = 0,
    ) -> list[dict]:
        """Return a page of records with optional filters.

        Args:
            split: Filter by split name (case-insensitive).
            status: Filter by detection status string.
            class_id: Filter by class id (prediction or ground truth).
            limit: Maximum records to return.
            offset: Row offset for pagination.

        Returns:
            List of record dicts matching :class:`EmbeddingRecordDTO` shape.
        """
        where, params = _build_where(split=split, status=status, class_id=class_id)
        sql = f"SELECT * FROM records{where} LIMIT ? OFFSET ?"  # noqa: S608
        params += [limit, offset]
        with self._connect() as conn:
            rows = conn.execute(sql, params).fetchall()
        return [_row_to_dto(r) for r in rows]

    def get_image_paths(
        self,
        split: str | None = None,
        limit: int = 60,
        offset: int = 0,
    ) -> tuple[list[str], int]:
        """Return a paginated list of distinct image paths.

        Args:
            split: Optional split filter (case-insensitive).
            limit: Maximum paths to return.
            offset: Row offset for pagination.

        Returns:
            Tuple of (list of image path strings, total distinct path count).
        """
        where, params = _build_where(split=split)
        count_sql = f"SELECT COUNT(DISTINCT image_path) FROM records{where}"  # noqa: S608
        page_sql = (
            f"SELECT DISTINCT image_path FROM records{where} "  # noqa: S608
            "ORDER BY image_path LIMIT ? OFFSET ?"
        )
        with self._connect() as conn:
            total = conn.execute(count_sql, params).fetchone()[0]
            rows = conn.execute(page_sql, params + [limit, offset]).fetchall()
        paths = [r[0] for r in rows]
        return paths, total

    def get_records_by_image_paths(
        self,
        image_paths: list[str],
        split: str | None = None,
    ) -> list[dict]:
        """Return all records whose ``image_path`` is in *image_paths*.

        Args:
            image_paths: List of absolute image path strings.
            split: Optional split filter (case-insensitive).

        Returns:
            List of record dicts matching :class:`EmbeddingRecordDTO` shape.
        """
        if not image_paths:
            return []
        placeholders = ",".join("?" * len(image_paths))
        sql = f"SELECT * FROM records WHERE image_path IN ({placeholders})"  # noqa: S608
        params: list[Any] = list(image_paths)
        if split is not None:
            sql += " AND LOWER(split) = LOWER(?)"
            params.append(split)
        with self._connect() as conn:
            rows = conn.execute(sql, params).fetchall()
        return [_row_to_dto(r) for r in rows]

    def get_evaluation_rows(self) -> list[dict[str, Any]]:
        """Return compact rows required to compute evaluation metrics.

        Returns:
            List of dictionaries with status, predicted class/confidence and ground-truth class.
        """
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT
                    status,
                    pred_class_id,
                    pred_confidence,
                    pred_x1, pred_y1, pred_x2, pred_y2,
                    gt_class_id,
                    gt_confidence,
                    gt_x1, gt_y1, gt_x2, gt_y2
                FROM records
                """
            ).fetchall()
        return [dict(row) for row in rows]

    def get_image_path_for_record(self, record_id: str) -> str | None:
        """Return the ``image_path`` for a given record id (primary-key lookup).

        Args:
            record_id: Primary key of the record.

        Returns:
            Absolute image path string, or None if the record doesn't exist.
        """
        with self._connect() as conn:
            row = conn.execute(
                "SELECT image_path FROM records WHERE id = ?", (record_id,)
            ).fetchone()
        return row["image_path"] if row else None

    def get_raw_embedding(self, record_id: str) -> list[float] | None:
        """Return the full-dimensionality raw embedding for a given record id.

        Unlike the ``embedding`` field in :func:`_row_to_dto` (which prefers
        reduced coordinates once computed), this always returns the original
        full-dimensionality vector used for similarity search.

        Args:
            record_id: Primary key of the record.

        Returns:
            The raw embedding as a list of floats, or None if the record doesn't
            exist or has no stored embedding (e.g. false negatives).
        """
        with self._connect() as conn:
            row = conn.execute(
                "SELECT raw_embedding FROM records WHERE id = ?", (record_id,)
            ).fetchone()
        if row is None or row["raw_embedding"] is None:
            return None
        return json.loads(row["raw_embedding"])

    @staticmethod
    def db_exists(dataset_path: str | Path) -> bool:
        """Return True if a DB file already exists at *dataset_path*.

        Args:
            dataset_path: Root directory of a dataset.

        Returns:
            Boolean indicating whether the DB file exists.
        """
        return (Path(dataset_path) / DB_FILENAME).exists()

    def _table_exists(self, table_name: str) -> bool:
        """Return True when *table_name* exists in this SQLite DB."""
        with self._connect() as conn:
            row = conn.execute(
                "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = ?",
                (table_name,),
            ).fetchone()
        return row is not None


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _build_where(
    split: str | None = None,
    status: str | None = None,
    class_id: int | None = None,
) -> tuple[str, list[Any]]:
    """Build a SQL WHERE clause from optional filter arguments.

    Args:
        split: Optional split name filter.
        status: Optional status filter.
        class_id: Optional class id filter (matches prediction OR ground truth).

    Returns:
        Tuple of (WHERE clause string starting with ' WHERE ' or '', params list).
    """
    clauses: list[str] = []
    params: list[Any] = []
    if split is not None:
        clauses.append("LOWER(split) = LOWER(?)")
        params.append(split)
    if status is not None:
        clauses.append("status = ?")
        params.append(status)
    if class_id is not None:
        clauses.append("(pred_class_id = ? OR gt_class_id = ?)")
        params += [class_id, class_id]
    where = (" WHERE " + " AND ".join(clauses)) if clauses else ""
    return where, params


def _row_to_dto(row: sqlite3.Row) -> dict:
    """Convert a DB row to an :class:`EmbeddingRecordDTO`-shaped dict.

    Args:
        row: A sqlite3.Row from the ``records`` table.

    Returns:
        Dict with fields matching the frontend EmbeddingRecordDTO type.
    """
    prediction = None
    if row["pred_class_id"] is not None:
        prediction = {
            "class_id": row["pred_class_id"],
            "confidence": row["pred_confidence"],
            "bbox": (
                [row["pred_x1"], row["pred_y1"], row["pred_x2"], row["pred_y2"]]
                if row["pred_x1"] is not None
                else None
            ),
        }

    ground_truth = None
    if row["gt_class_id"] is not None:
        ground_truth = {
            "class_id": row["gt_class_id"],
            "confidence": row["gt_confidence"],
            "bbox": (
                [row["gt_x1"], row["gt_y1"], row["gt_x2"], row["gt_y2"]]
                if row["gt_x1"] is not None
                else None
            ),
        }

    # Return reduced coords if available, otherwise the raw embedding.
    embedding: list[float] | None = None
    if row["pca_embedding"] is not None:
        embedding = json.loads(row["pca_embedding"])
    elif row["raw_embedding"] is not None:
        embedding = json.loads(row["raw_embedding"])

    return {
        "id": row["id"],
        "image_path": row["image_path"],
        "split": row["split"],
        "status": row["status"],
        "embedding": embedding,
        "prediction": prediction,
        "ground_truth": ground_truth,
    }
