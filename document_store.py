"""Factories for Haystack document stores."""

from __future__ import annotations

from haystack.utils import Secret
from haystack_integrations.document_stores.pinecone import PineconeDocumentStore

from config import Settings


def create_pinecone_document_store(settings: Settings) -> PineconeDocumentStore:
    """Create a Pinecone document store configured from application settings."""
    return PineconeDocumentStore(
        api_key=Secret.from_env_var("PINECONE_API_KEY"),
        index=settings.pinecone_index_name,
        namespace=settings.pinecone_namespace,
        dimension=settings.pinecone_dimension,
        metric=settings.pinecone_metric,
        show_progress=False,
    )
