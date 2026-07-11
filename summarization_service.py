"""Service for retrieving context and generating team chat summaries."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from haystack import Document, Pipeline
from haystack.dataclasses.chat_message import ChatMessage as HaystackChatMessage
from haystack.dataclasses.chat_message import ChatRole

from models import RetrievalRequest, SummarizationRequest, SummarizationResult
from retrieval_service import RetrievalService


class SummarizationServiceError(Exception):
    """Raised when summarization input or pipeline output is invalid."""


class NoSummarizationContextError(SummarizationServiceError):
    """Raised when retrieval returned no documents to summarize."""


class SummarizationResultError(SummarizationServiceError):
    """Raised when retrieved context or LLM output is invalid for summarization."""


class SummarizationService:
    """Runs retrieval and the summarization pipeline for a scoped request."""

    def __init__(
        self,
        retrieval_service: RetrievalService,
        summarization_pipeline: Pipeline,
    ) -> None:
        self._retrieval_service = retrieval_service
        self._summarization_pipeline = summarization_pipeline

    def summarize(self, request: SummarizationRequest) -> SummarizationResult:
        """Retrieve validated context and generate a summary."""
        retrieval_request = RetrievalRequest(
            query=request.context_query,
            chat_id=request.chat_id,
            session_id=request.session_id,
        )
        documents = self._retrieval_service.retrieve(retrieval_request)
        if not documents:
            raise NoSummarizationContextError("retrieval returned no documents to summarize")

        _validate_documents_for_summarization(documents, request)

        result = self._summarization_pipeline.run(
            {
                "prompt_builder": {
                    "documents": list(documents),
                    "instruction": request.instruction,
                }
            },
            include_outputs_from={"llm"},
        )
        reply_text = _extract_reply_text(result)
        return SummarizationResult(
            text=reply_text,
            source_document_ids=tuple(document.id for document in documents),
        )


def _validate_documents_for_summarization(
    documents: Sequence[Document],
    request: SummarizationRequest,
) -> None:
    if not documents:
        raise SummarizationResultError("documents must not be empty")

    seen_ids: set[str] = set()
    expected_chat_id = str(request.chat_id)
    for document in documents:
        if not isinstance(document, Document):
            raise SummarizationResultError("documents must contain Haystack Document instances")
        if not document.id:
            raise SummarizationResultError("document id must not be empty")
        if document.id in seen_ids:
            raise SummarizationResultError("documents contain duplicate ids")
        seen_ids.add(document.id)

        metadata = document.meta
        if not isinstance(metadata, Mapping):
            raise SummarizationResultError("document metadata must be a mapping")
        if metadata.get("chat_id") != expected_chat_id:
            raise SummarizationResultError("document chat_id does not match request")
        if metadata.get("session_id") != request.session_id:
            raise SummarizationResultError("document session_id does not match request")
        if metadata.get("source") != "telegram":
            raise SummarizationResultError("document source must be telegram")


def _extract_reply_text(result: object) -> str:
    if not isinstance(result, Mapping):
        raise SummarizationResultError("Pipeline result must be a mapping")

    try:
        llm_result = result["llm"]
    except KeyError as exc:
        raise SummarizationResultError("Pipeline result is missing llm output") from exc

    if not isinstance(llm_result, Mapping):
        raise SummarizationResultError("Pipeline llm output must be a mapping")

    try:
        replies = llm_result["replies"]
    except KeyError as exc:
        raise SummarizationResultError("Pipeline result is missing llm.replies") from exc

    if not isinstance(replies, list):
        raise SummarizationResultError("llm.replies must be a list")
    if len(replies) != 1:
        raise SummarizationResultError("llm.replies must contain exactly one reply")

    reply = replies[0]
    if not isinstance(reply, HaystackChatMessage):
        raise SummarizationResultError("llm reply must be a ChatMessage")
    if not reply.is_from(ChatRole.ASSISTANT):
        raise SummarizationResultError("llm reply must have assistant role")

    if reply.tool_calls:
        raise SummarizationResultError("llm reply must not contain tool calls without text")

    text = reply.text
    if text is None or not text.strip():
        raise SummarizationResultError("llm reply text must be a non-empty string")

    return text.strip()
