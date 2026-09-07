# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

"""Tests for the single-active-search semantic-search registry."""

from collections.abc import Iterator

import pytest

from visualizer.backend.semantic_search.engine import (
    SEARCH_JOB_STORE,
    SEARCH_JOB_STORE_LOCK,
    try_register_active_search,
)
from visualizer.backend.semantic_search.types import SearchJob, SearchStatus


def _search_job(search_id: str, status: SearchStatus = "pending") -> SearchJob:
    """Create a minimal search job for registry tests."""
    return SearchJob(
        id=search_id,
        parent_job_id="job-1",
        query_record_id="record-1",
        query_image_path="query.jpg",
        search_path="search-folder",
        k=20,
        status=status,
    )


@pytest.fixture(autouse=True)
def _isolate_search_job_store() -> Iterator[None]:
    """Restore the process-global semantic-search registry after each test."""
    with SEARCH_JOB_STORE_LOCK:
        original = SEARCH_JOB_STORE.copy()
        SEARCH_JOB_STORE.clear()
    yield
    with SEARCH_JOB_STORE_LOCK:
        SEARCH_JOB_STORE.clear()
        SEARCH_JOB_STORE.update(original)


class TestSingleActiveSemanticSearch:
    """Only one pending or running semantic search may be registered at a time."""

    @pytest.mark.parametrize("status", ["pending", "running"])
    def test_rejects_another_search_while_one_is_active(self, status: SearchStatus) -> None:
        assert try_register_active_search(_search_job("active", status))

        assert not try_register_active_search(_search_job("new"))
        assert set(SEARCH_JOB_STORE) == {"active"}

    @pytest.mark.parametrize("status", ["done", "error", "cancelled"])
    def test_allows_a_new_search_after_a_terminal_search(self, status: SearchStatus) -> None:
        SEARCH_JOB_STORE["completed"] = _search_job("completed", status)

        assert try_register_active_search(_search_job("new"))
        assert set(SEARCH_JOB_STORE) == {"completed", "new"}
