"""Haystack pipeline builders for indexing, retrieval, and summarization."""

from __future__ import annotations

from typing import Protocol

from haystack import Pipeline
from haystack.components.builders import ChatPromptBuilder
from haystack.components.embedders import OpenAIDocumentEmbedder, OpenAITextEmbedder
from haystack.components.generators.chat import OpenAIChatGenerator
from haystack.components.writers import DocumentWriter
from haystack.document_stores.types import DocumentStore, DuplicatePolicy
from haystack.utils import Secret
from haystack_integrations.components.retrievers.pinecone import PineconeEmbeddingRetriever

from config import Settings
from models import ChatMessage
from summarization_prompt import SUMMARIZATION_PROMPT_TEMPLATE


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


def _build_openai_embedder_kwargs(settings: Settings) -> dict[str, object]:
    kwargs: dict[str, object] = {
        "api_key": Secret.from_env_var("OPENAI_API_KEY"),
        "model": settings.embedding_model,
    }
    if settings.api_base_url is not None:
        kwargs["api_base_url"] = settings.api_base_url
    return kwargs


def _build_document_embedder_kwargs(settings: Settings) -> dict[str, object]:
    # raise_on_failure=True is required: by default Haystack logs a failed embedding
    # request and passes the document on without a vector, after which the Pinecone
    # store writes a dummy vector and reports the document as written.
    return {
        **_build_openai_embedder_kwargs(settings),
        "progress_bar": False,
        "raise_on_failure": True,
    }


def _build_openai_chat_generator_kwargs(settings: Settings) -> dict[str, object]:
    kwargs: dict[str, object] = {
        "api_key": Secret.from_env_var("OPENAI_API_KEY"),
        "model": settings.openai_model,
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


def create_query_pipeline(settings: Settings, document_store: DocumentStore) -> Pipeline:
    """Build the query/retrieval Haystack pipeline."""
    text_embedder = OpenAITextEmbedder(**_build_openai_embedder_kwargs(settings))
    retriever = PineconeEmbeddingRetriever(
        document_store=document_store,
        top_k=settings.retrieval_top_k,
    )

    pipeline = Pipeline()
    pipeline.add_component("text_embedder", text_embedder)
    pipeline.add_component("retriever", retriever)
    pipeline.connect("text_embedder.embedding", "retriever.query_embedding")
    return pipeline


def create_summarization_pipeline(settings: Settings) -> Pipeline:
    """Build the summarization Haystack pipeline."""
    prompt_builder = ChatPromptBuilder(
        template=list(SUMMARIZATION_PROMPT_TEMPLATE),
        required_variables=["documents", "instruction"],
    )
    llm = OpenAIChatGenerator(**_build_openai_chat_generator_kwargs(settings))

    pipeline = Pipeline()
    pipeline.add_component("prompt_builder", prompt_builder)
    pipeline.add_component("llm", llm)
    pipeline.connect("prompt_builder.prompt", "llm.messages")
    return pipeline


def build_retrieval_pipeline(settings: Settings) -> RetrievalPipeline:
    """Build the query/retrieval Haystack pipeline. Not implemented in Stage 2A."""
    raise NotImplementedError("Retrieval pipeline is not implemented yet.")


def build_summarization_pipeline(settings: Settings) -> SummarizationPipeline:
    """Build the summarization Haystack pipeline. Not implemented in Stage 2A."""
    raise NotImplementedError("Summarization pipeline is not implemented yet.")
