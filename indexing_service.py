"""Service for running the indexing Haystack pipeline."""

from __future__ import annotations

from collections.abc import Sequence

from haystack import Pipeline

from documents import chat_message_to_document
from models import ChatMessage


class IndexingServiceError(Exception):
    """Raised when indexing input or pipeline output is invalid."""


class IndexingService:
    """Runs the indexing pipeline for batches of chat messages."""

    def __init__(self, pipeline: Pipeline) -> None:
        self._pipeline = pipeline

    def index_messages(self, messages: Sequence[ChatMessage]) -> int:
        """Index chat messages and return the number of documents written."""
        if not messages:
            raise IndexingServiceError("messages must not be empty")

        documents = [chat_message_to_document(message) for message in messages]
        result = self._pipeline.run(
            {"document_embedder": {"documents": documents}},
            include_outputs_from={"writer"},
        )
        return _extract_documents_written(result)


def _extract_documents_written(result: dict[str, object]) -> int:
    try:
        writer_result = result["writer"]
        if not isinstance(writer_result, dict):
            raise IndexingServiceError("Pipeline writer output must be a mapping")
        documents_written = writer_result["documents_written"]
    except KeyError as exc:
        raise IndexingServiceError("Pipeline result is missing writer.documents_written") from exc

    if isinstance(documents_written, bool) or not isinstance(documents_written, int):
        raise IndexingServiceError("documents_written must be a non-negative integer")
    if documents_written < 0:
        raise IndexingServiceError("documents_written must be a non-negative integer")
    return documents_written
