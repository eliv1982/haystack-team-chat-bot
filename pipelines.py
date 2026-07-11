"""Haystack pipeline builders for indexing, retrieval, and summarization."""

from __future__ import annotations

from typing import Protocol

from haystack import Pipeline
from haystack.components.embedders import OpenAIDocumentEmbedder
from haystack.components.writers import DocumentWriter
from haystack.document_stores.types import DocumentStore, DuplicatePolicy
from haystack.utils import Secret

from config import Settings
from models import ChatMessage


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


def _build_document_embedder_kwargs(settings: Settings) -> dict[str, object]:
    kwargs: dict[str, object] = {
        "api_key": Secret.from_env_var("OPENAI_API_KEY"),
        "model": settings.embedding_model,
        "progress_bar": False,
    }
    if settings.api_base_url is not None:
        kwargs["api_base_url"] = settings.api_base_url
    return kwargs


def create_indexing_pipeline(settings: Settings, document_store: DocumentStore) -> Pipeline:
    """Build the indexing Haystack pipeline."""
    document_embedder = OpenAIDocumentEmbedder(**_build_document_embedder_kwargs(settings))
    writer = DocumentWriter(
        document_store=document_store,
        policy=DuplicatePolicy.OVERWRITE,
    )

    pipeline = Pipeline()
    pipeline.add_component("document_embedder", document_embedder)
    pipeline.add_component("writer", writer)
    pipeline.connect("document_embedder.documents", "writer.documents")
    return pipeline


def build_retrieval_pipeline(settings: Settings) -> RetrievalPipeline:
    """Build the query/retrieval Haystack pipeline. Not implemented in Stage 2A."""
    raise NotImplementedError("Retrieval pipeline is not implemented yet.")


def build_summarization_pipeline(settings: Settings) -> SummarizationPipeline:
    """Build the summarization Haystack pipeline. Not implemented in Stage 2A."""
    raise NotImplementedError("Summarization pipeline is not implemented yet.")
