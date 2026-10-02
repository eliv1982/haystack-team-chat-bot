"""Service for running the indexing Haystack pipeline."""

from __future__ import annotations

from collections.abc import Sequence

from haystack import Pipeline
from haystack.core.errors import PipelineRuntimeError

from documents import chat_message_to_document
from models import ChatMessage
from provider_errors import OPENAI_REQUEST_FAILURES, PINECONE_REQUEST_FAILURES, is_provider_failure

# The indexing pipeline embeds with OpenAI and writes to Pinecone.
_PROVIDER_REQUEST_FAILURES = (*OPENAI_REQUEST_FAILURES, *PINECONE_REQUEST_FAILURES)


class IndexingServiceError(Exception):
    """Raised when indexing input or pipeline output is invalid, or a provider failed."""


class IndexingProviderError(IndexingServiceError):
    """Raised when a request to OpenAI or Pinecone failed while indexing.

    Means exactly that: an expected outage, rate limit or rejected request of the
    provider. A bug inside a pipeline component is never reported as this error.
    """


class IndexingService:
    """Runs the indexing pipeline for batches of chat messages."""

    def __init__(self, pipeline: Pipeline) -> None:
        self._pipeline = pipeline

    def index_messages(self, messages: Sequence[ChatMessage]) -> int:
        """Index chat messages and return the number of documents written."""
        if not messages:
            raise IndexingServiceError("messages must not be empty")

        documents = [chat_message_to_document(message) for message in messages]
        try:
            result = self._pipeline.run(
                {"document_embedder": {"documents": documents}},
                include_outputs_from={"writer"},
            )
        except PipelineRuntimeError as exc:
            if is_provider_failure(exc, _PROVIDER_REQUEST_FAILURES):
                raise IndexingProviderError("indexing provider request failed") from exc
            raise
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
