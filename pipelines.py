"""Haystack pipeline placeholders for indexing, retrieval, and summarization."""

from __future__ import annotations

from typing import Protocol

from config import Settings
from models import ChatMessage


class IndexingPipeline(Protocol):
    """Pipeline for indexing chat messages into the vector store."""

    def index_messages(self, messages: list[ChatMessage]) -> None:
        """Index messages into Pinecone. Implementation deferred to a later stage."""
        ...


class RetrievalPipeline(Protocol):
    """Pipeline for querying relevant context from the vector store."""

    def retrieve(self, query: str) -> list[ChatMessage]:
        """Retrieve relevant messages. Implementation deferred to a later stage."""
        ...


class SummarizationPipeline(Protocol):
    """Pipeline for generating summaries from retrieved context."""

    def summarize(self, context: str) -> str:
        """Generate a summary. Implementation deferred to a later stage."""
        ...


def build_indexing_pipeline(settings: Settings) -> IndexingPipeline:
    """Build the indexing Haystack pipeline. Not implemented in Stage 1."""
    raise NotImplementedError("Indexing pipeline is not implemented yet.")


def build_retrieval_pipeline(settings: Settings) -> RetrievalPipeline:
    """Build the query/retrieval Haystack pipeline. Not implemented in Stage 1."""
    raise NotImplementedError("Retrieval pipeline is not implemented yet.")


def build_summarization_pipeline(settings: Settings) -> SummarizationPipeline:
    """Build the summarization Haystack pipeline. Not implemented in Stage 1."""
    raise NotImplementedError("Summarization pipeline is not implemented yet.")
