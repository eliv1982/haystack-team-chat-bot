"""Tests for IndexingService against a mocked pipeline result.

The service with the real indexing pipeline is covered by test_indexing_failure.py.
"""

from __future__ import annotations

from dataclasses import replace
from unittest.mock import MagicMock

import pytest
from haystack import Pipeline

from indexing_service import IndexingService, IndexingServiceError
from models import ChatMessage


@pytest.fixture
def pipeline() -> MagicMock:
    return MagicMock(spec=Pipeline)


def test_indexing_service_rejects_empty_input(pipeline: MagicMock) -> None:
    service = IndexingService(pipeline=pipeline)

    with pytest.raises(IndexingServiceError, match="must not be empty"):
        service.index_messages([])

    pipeline.run.assert_not_called()


def test_indexing_service_runs_the_pipeline_once_for_a_batch(
    pipeline: MagicMock, sample_message: ChatMessage
) -> None:
    pipeline.run.return_value = {"writer": {"documents_written": 2}}
    messages = [sample_message, replace(sample_message, message_id=43, text="Second message")]

    written = IndexingService(pipeline=pipeline).index_messages(messages)

    assert written == 2
    pipeline.run.assert_called_once()
    run_args, run_kwargs = pipeline.run.call_args
    assert len(run_args[0]["document_embedder"]["documents"]) == 2
    assert run_kwargs["include_outputs_from"] == {"writer"}


def test_indexing_service_reports_zero_documents_written(
    pipeline: MagicMock, sample_message: ChatMessage
) -> None:
    pipeline.run.return_value = {"writer": {"documents_written": 0}}

    assert IndexingService(pipeline=pipeline).index_messages([sample_message]) == 0


@pytest.mark.parametrize(
    "pipeline_result",
    [
        {},
        {"writer": {}},
        {"writer": {"documents_written": -1}},
        {"writer": {"documents_written": True}},
        {"writer": {"documents_written": "2"}},
        {"writer": "bad"},
    ],
)
def test_indexing_service_rejects_malformed_pipeline_results(
    pipeline: MagicMock,
    sample_message: ChatMessage,
    pipeline_result: dict[str, object],
) -> None:
    pipeline.run.return_value = pipeline_result

    with pytest.raises(IndexingServiceError):
        IndexingService(pipeline=pipeline).index_messages([sample_message])


def test_indexing_service_does_not_swallow_pipeline_errors(
    pipeline: MagicMock, sample_message: ChatMessage
) -> None:
    pipeline.run.side_effect = RuntimeError("pipeline failed")

    with pytest.raises(RuntimeError, match="pipeline failed"):
        IndexingService(pipeline=pipeline).index_messages([sample_message])
