"""Service for running the query/retrieval Haystack pipeline."""

from __future__ import annotations

import math
from collections.abc import Mapping

from haystack import Document, Pipeline

from models import RetrievalRequest
from retrieval_filters import build_chat_session_filter

_REQUIRED_METADATA_FIELDS = ("message_id", "user_id", "author_name", "sent_at")


class RetrievalServiceError(Exception):
    """Raised when retrieval input or pipeline output is invalid."""


class RetrievalInvariantError(RetrievalServiceError):
    """Raised when a retrieved document violates chat/session invariants."""


class RetrievalService:
    """Runs the query pipeline and validates retrieved documents."""

    def __init__(self, pipeline: Pipeline, *, top_k: int) -> None:
        self._pipeline = pipeline
        self._top_k = _validate_top_k(top_k)

    def retrieve(self, request: RetrievalRequest) -> tuple[Document, ...]:
        """Retrieve validated documents for the scoped request."""
        filters = build_chat_session_filter(request)
        result = self._pipeline.run(
            {
                "text_embedder": {
                    "text": request.query,
                },
                "retriever": {
                    "filters": filters,
                    "top_k": self._top_k,
                },
            },
            include_outputs_from={"retriever"},
        )
        return _extract_validated_documents(result, request, self._top_k)


def _validate_top_k(top_k: object) -> int:
    if isinstance(top_k, bool) or not isinstance(top_k, int):
        raise RetrievalServiceError("top_k must be a positive integer")
    if top_k <= 0:
        raise RetrievalServiceError("top_k must be a positive integer")
    return top_k


def _extract_validated_documents(
    result: object,
    request: RetrievalRequest,
    top_k: int,
) -> tuple[Document, ...]:
    if not isinstance(result, Mapping):
        raise RetrievalServiceError("Pipeline result must be a mapping")

    try:
        retriever_result = result["retriever"]
    except KeyError as exc:
        raise RetrievalServiceError("Pipeline result is missing retriever output") from exc

    if not isinstance(retriever_result, Mapping):
        raise RetrievalServiceError("Pipeline retriever output must be a mapping")

    try:
        documents = retriever_result["documents"]
    except KeyError as exc:
        raise RetrievalServiceError("Pipeline result is missing retriever.documents") from exc

    if not isinstance(documents, list):
        raise RetrievalServiceError("retriever.documents must be a list")

    if len(documents) > top_k:
        raise RetrievalServiceError("retriever returned more documents than top_k")

    validated: list[Document] = []
    seen_ids: set[str] = set()
    for document in documents:
        _validate_document(document, request, seen_ids)
        validated.append(document)

    return tuple(validated)


def _validate_document(
    document: object,
    request: RetrievalRequest,
    seen_ids: set[str],
) -> None:
    _validate_identity_and_content(document, seen_ids)
    _validate_score(document)
    _validate_scope_and_metadata(
        document,
        chat_id=request.chat_id,
        session_id=request.session_id,
    )


def validate_scoped_document(
    document: object,
    *,
    chat_id: int,
    session_id: str,
    seen_ids: set[str],
) -> None:
    """Fail closed unless a loaded document belongs to the given chat and session.

    Shared by every path that loads session documents. Unlike ``_validate_document``
    it does not require a similarity score, because documents loaded by metadata
    filter alone have none.
    """
    _validate_identity_and_content(document, seen_ids)
    _validate_scope_and_metadata(document, chat_id=chat_id, session_id=session_id)


def _validate_identity_and_content(document: object, seen_ids: set[str]) -> None:
    if not isinstance(document, Document):
        raise RetrievalServiceError("retriever.documents must contain Haystack Document instances")

    if not document.id:
        raise RetrievalServiceError("retrieved document id must not be empty")

    if document.id in seen_ids:
        raise RetrievalServiceError("retriever returned duplicate document ids")
    seen_ids.add(document.id)

    if not isinstance(document.content, str) or not document.content:
        raise RetrievalServiceError("retrieved document content must be a non-empty string")


def _validate_score(document: Document) -> None:
    score = document.score
    if score is None:
        raise RetrievalServiceError("retrieved document score must be present")
    if isinstance(score, bool) or not isinstance(score, (int, float)):
        raise RetrievalServiceError("retrieved document score must be a number")
    if not math.isfinite(float(score)):
        raise RetrievalServiceError("retrieved document score must be finite")


def _validate_scope_and_metadata(
    document: Document,
    *,
    chat_id: int,
    session_id: str,
) -> None:
    metadata = document.meta
    if not isinstance(metadata, Mapping):
        raise RetrievalServiceError("retrieved document metadata must be a mapping")

    expected_chat_id = str(chat_id)
    if metadata.get("chat_id") != expected_chat_id:
        raise RetrievalInvariantError("retrieved document chat_id does not match request")

    if metadata.get("session_id") != session_id:
        raise RetrievalInvariantError("retrieved document session_id does not match request")

    if metadata.get("source") != "telegram":
        raise RetrievalInvariantError("retrieved document source must be telegram")

    for field_name in _REQUIRED_METADATA_FIELDS:
        if field_name not in metadata:
            raise RetrievalInvariantError(f"retrieved document metadata is missing {field_name}")
        if metadata[field_name] is None:
            raise RetrievalInvariantError(f"retrieved document metadata is missing {field_name}")
