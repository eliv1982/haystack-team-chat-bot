"""Tests for retrieval service behavior and validation."""

from __future__ import annotations

import math
from unittest.mock import MagicMock

import pytest
from haystack import Document, Pipeline

from models import RetrievalRequest
from retrieval_service import (
    RetrievalInvariantError,
    RetrievalService,
    RetrievalServiceError,
)


def _request(
    *,
    chat_id: int = -1001234567890,
    session_id: str = "chat:-1001234567890",
    query: str = "What did Alice say?",
) -> RetrievalRequest:
    return RetrievalRequest(query=query, chat_id=chat_id, session_id=session_id)


def _metadata(
    *,
    chat_id: int = -1001234567890,
    session_id: str = "chat:-1001234567890",
    **overrides: object,
) -> dict[str, object]:
    metadata: dict[str, object] = {
        "source": "telegram",
        "schema_version": 1,
        "chat_id": str(chat_id),
        "message_id": "42",
        "user_id": "7",
        "session_id": session_id,
        "author_name": "Alice",
        "sent_at": "2024-01-15T12:30:00+00:00",
    }
    metadata.update(overrides)
    return metadata


def _document(
    *,
    document_id: str = "doc-1",
    content: str = "Hello from Alice",
    score: float | int = 0.91,
    chat_id: int = -1001234567890,
    session_id: str = "chat:-1001234567890",
    include_embedding: bool = False,
    metadata_overrides: dict[str, object] | None = None,
    drop_metadata_fields: tuple[str, ...] = (),
) -> Document:
    metadata = _metadata(chat_id=chat_id, session_id=session_id)
    if metadata_overrides is not None:
        metadata.update(metadata_overrides)
    for field_name in drop_metadata_fields:
        metadata.pop(field_name, None)
    document = Document(
        id=document_id,
        content=content,
        meta=metadata,
        score=score,
        embedding=[0.1, 0.2] if include_embedding else None,
    )
    if document_id == "":
        object.__setattr__(document, "id", "")
    return document


def _pipeline_result(documents: list[Document]) -> dict[str, object]:
    return {"retriever": {"documents": documents}}


@pytest.fixture
def pipeline() -> MagicMock:
    return MagicMock(spec=Pipeline)


@pytest.fixture
def service(pipeline: MagicMock) -> RetrievalService:
    return RetrievalService(pipeline=pipeline, top_k=3)


@pytest.mark.parametrize("top_k", [1, 3, 50])
def test_constructor_accepts_positive_top_k(pipeline: MagicMock, top_k: int) -> None:
    service = RetrievalService(pipeline=pipeline, top_k=top_k)

    assert service._top_k == top_k


@pytest.mark.parametrize("top_k", [0, -1])
def test_constructor_rejects_non_positive_top_k(pipeline: MagicMock, top_k: int) -> None:
    with pytest.raises(RetrievalServiceError, match="top_k"):
        RetrievalService(pipeline=pipeline, top_k=top_k)


@pytest.mark.parametrize("top_k", [True, "3", 3.5])
def test_constructor_rejects_invalid_top_k_type(pipeline: MagicMock, top_k: object) -> None:
    with pytest.raises(RetrievalServiceError, match="top_k"):
        RetrievalService(pipeline=pipeline, top_k=top_k)  # type: ignore[arg-type]


def test_retrieve_calls_pipeline_once_with_expected_inputs(service: RetrievalService, pipeline: MagicMock) -> None:
    request = _request()
    pipeline.run.return_value = _pipeline_result([])

    service.retrieve(request)

    pipeline.run.assert_called_once()
    run_args, run_kwargs = pipeline.run.call_args
    assert run_args[0] == {
        "text_embedder": {"text": request.query},
        "retriever": {
            "filters": {
                "operator": "AND",
                "conditions": [
                    {
                        "field": "meta.chat_id",
                        "operator": "==",
                        "value": str(request.chat_id),
                    },
                    {
                        "field": "meta.session_id",
                        "operator": "==",
                        "value": request.session_id,
                    },
                ],
            },
            "top_k": 3,
        },
    }
    assert run_kwargs["include_outputs_from"] == {"retriever"}


def test_retrieve_does_not_mutate_request(service: RetrievalService, pipeline: MagicMock) -> None:
    request = _request()
    original = (request.query, request.chat_id, request.session_id)
    pipeline.run.return_value = _pipeline_result([])

    service.retrieve(request)

    assert (request.query, request.chat_id, request.session_id) == original


def test_retrieve_returns_single_document(service: RetrievalService, pipeline: MagicMock) -> None:
    document = _document()
    pipeline.run.return_value = _pipeline_result([document])

    result = service.retrieve(_request())

    assert result == (document,)


def test_retrieve_returns_multiple_documents_in_order(service: RetrievalService, pipeline: MagicMock) -> None:
    first = _document(document_id="doc-1", score=0.95)
    second = _document(document_id="doc-2", score=0.80)
    pipeline.run.return_value = _pipeline_result([first, second])

    result = service.retrieve(_request())

    assert result == (first, second)


def test_retrieve_allows_empty_result(service: RetrievalService, pipeline: MagicMock) -> None:
    pipeline.run.return_value = _pipeline_result([])

    assert service.retrieve(_request()) == ()


def test_retrieve_returns_immutable_tuple(service: RetrievalService, pipeline: MagicMock) -> None:
    document = _document()
    pipeline.run.return_value = _pipeline_result([document])

    result = service.retrieve(_request())

    assert isinstance(result, tuple)


@pytest.mark.parametrize("score", [1, 0.75])
def test_retrieve_accepts_int_and_float_scores(
    service: RetrievalService,
    pipeline: MagicMock,
    score: float | int,
) -> None:
    pipeline.run.return_value = _pipeline_result([_document(score=score)])

    result = service.retrieve(_request())

    assert result[0].score == score


def test_retrieve_allows_missing_embedding(service: RetrievalService, pipeline: MagicMock) -> None:
    pipeline.run.return_value = _pipeline_result([_document(include_embedding=False)])

    result = service.retrieve(_request())

    assert result[0].embedding is None


@pytest.mark.parametrize(
    "pipeline_result",
    [
        "bad",
        [],
        {"writer": {"documents": []}},
        {"retriever": "bad"},
        {"retriever": {}},
        {"retriever": {"documents": "bad"}},
        {"retriever": {"documents": [object()]}},
        {"retriever": {"documents": [_document(document_id="")]}},
        {"retriever": {"documents": [_document(content="")]}},
        {"retriever": {"documents": [_document(score=None)]}},  # type: ignore[arg-type]
        {"retriever": {"documents": [_document(score=True)]}},  # type: ignore[arg-type]
        {"retriever": {"documents": [_document(score="0.5")]}},  # type: ignore[arg-type]
        {"retriever": {"documents": [_document(score=math.nan)]}},
        {"retriever": {"documents": [_document(score=math.inf)]}},
        {"retriever": {"documents": [_document(score=-math.inf)]}},
        {"retriever": {"documents": [_document(), _document(), _document(), _document()]}},
        {
            "retriever": {
                "documents": [
                    _document(document_id="dup"),
                    _document(document_id="dup", score=0.5),
                ]
            }
        },
        {"retriever": {"documents": [_document(metadata_overrides={"chat_id": "other"})]}},
        {"retriever": {"documents": [_document(metadata_overrides={"session_id": "other"})]}},
        {"retriever": {"documents": [_document(metadata_overrides={"source": "email"})]}},
        {"retriever": {"documents": [_document(metadata_overrides={"message_id": None})]}},
        {"retriever": {"documents": [_document(metadata_overrides={"user_id": None})]}},
        {"retriever": {"documents": [_document(metadata_overrides={"author_name": None})]}},
        {"retriever": {"documents": [_document(metadata_overrides={"sent_at": None})]}},
        {"retriever": {"documents": [_document(drop_metadata_fields=("message_id",))]}},
        {"retriever": {"documents": [_document(drop_metadata_fields=("user_id",))]}},
        {"retriever": {"documents": [_document(drop_metadata_fields=("author_name",))]}},
        {"retriever": {"documents": [_document(drop_metadata_fields=("sent_at",))]}},
    ],
)
def test_retrieve_rejects_malformed_pipeline_results(
    service: RetrievalService,
    pipeline: MagicMock,
    pipeline_result: object,
) -> None:
    pipeline.run.return_value = pipeline_result

    with pytest.raises((RetrievalServiceError, RetrievalInvariantError)):
        service.retrieve(_request())


def test_retrieve_rejects_non_mapping_metadata(service: RetrievalService, pipeline: MagicMock) -> None:
    document = _document()
    object.__setattr__(document, "meta", ["bad"])  # type: ignore[arg-type]
    pipeline.run.return_value = _pipeline_result([document])

    with pytest.raises(RetrievalServiceError, match="metadata"):
        service.retrieve(_request())


def test_retrieve_does_not_swallow_pipeline_errors(service: RetrievalService, pipeline: MagicMock) -> None:
    pipeline.run.side_effect = RuntimeError("pipeline failed")

    with pytest.raises(RuntimeError, match="pipeline failed"):
        service.retrieve(_request())

    assert pipeline.run.call_count == 1


def test_retrieve_does_not_retry_after_pipeline_error(service: RetrievalService, pipeline: MagicMock) -> None:
    pipeline.run.side_effect = RuntimeError("pipeline failed")

    with pytest.raises(RuntimeError):
        service.retrieve(_request())

    pipeline.run.assert_called_once()


def test_retrieve_has_no_document_store_fallback(service: RetrievalService, pipeline: MagicMock) -> None:
    pipeline.run.side_effect = RuntimeError("pipeline failed")

    with pytest.raises(RuntimeError):
        service.retrieve(_request())

    assert not hasattr(service, "_document_store")
