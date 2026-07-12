"""Production runtime composition and Telegram polling boundary."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

import telebot

from config import Settings, load_settings
from document_store import create_pinecone_document_store
from indexing_service import IndexingService
from pinecone_preflight import PineconePreflightError, validate_existing_pinecone_index
from pipelines import (
    create_indexing_pipeline,
    create_query_pipeline,
    create_summarization_pipeline,
)
from retrieval_service import RetrievalService
from session_store import InMemorySessionStore
from summarization_service import SummarizationService
from telegram_application import TelegramApplicationService
from telegram_bot import create_configured_telegram_bot
from telegram_commands import configure_telegram_command_menu
from telegram_summary_application import TelegramSummaryApplicationService

if TYPE_CHECKING:
    from haystack_integrations.document_stores.pinecone import PineconeDocumentStore

logger = logging.getLogger(__name__)

POLLING_KWARGS = {
    "skip_pending": True,
    "allowed_updates": ["message"],
}


def configure_logging() -> None:
    """Configure standard library logging for production startup."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


@dataclass(frozen=True, slots=True)
class RuntimeComponents:
    """Immutable container for assembled runtime dependencies."""

    document_store: PineconeDocumentStore
    session_store: InMemorySessionStore
    indexing_service: IndexingService
    retrieval_service: RetrievalService
    summarization_service: SummarizationService
    telegram_application_service: TelegramApplicationService
    telegram_summary_application_service: TelegramSummaryApplicationService
    bot: telebot.TeleBot


def assemble_runtime(
    *,
    settings: Settings,
    document_store: PineconeDocumentStore,
) -> RuntimeComponents:
    """Wire pipelines, services, session store, and configured TeleBot."""
    logger.info("Runtime assembly started")

    indexing_pipeline = create_indexing_pipeline(settings, document_store)
    query_pipeline = create_query_pipeline(settings, document_store)
    summarization_pipeline = create_summarization_pipeline(settings)

    indexing_service = IndexingService(indexing_pipeline)
    retrieval_service = RetrievalService(
        query_pipeline,
        top_k=settings.retrieval_top_k,
    )
    summarization_service = SummarizationService(
        retrieval_service,
        summarization_pipeline,
    )

    session_store = InMemorySessionStore()

    telegram_application_service = TelegramApplicationService(
        session_store=session_store,
        indexing_service=indexing_service,
    )
    telegram_summary_application_service = TelegramSummaryApplicationService(
        session_store=session_store,
        summarization_service=summarization_service,
    )

    bot = create_configured_telegram_bot(
        settings,
        telegram_application_service,
        telegram_summary_application_service,
    )

    logger.info("Runtime assembly completed")

    return RuntimeComponents(
        document_store=document_store,
        session_store=session_store,
        indexing_service=indexing_service,
        retrieval_service=retrieval_service,
        summarization_service=summarization_service,
        telegram_application_service=telegram_application_service,
        telegram_summary_application_service=telegram_summary_application_service,
        bot=bot,
    )


def build_runtime() -> RuntimeComponents:
    """Load settings, validate Pinecone, and assemble the full runtime."""
    settings = load_settings()
    preflight = validate_existing_pinecone_index(settings)
    if not preflight.ready:
        raise PineconePreflightError(
            f"Pinecone index not ready: status={preflight.status}"
        )

    logger.info(
        "Pinecone preflight completed: index=%s namespace=%s dimension=%s metric=%s",
        preflight.name,
        settings.pinecone_namespace,
        preflight.dimension,
        preflight.metric,
    )
    logger.info("OpenAI model: %s", settings.openai_model)

    document_store = create_pinecone_document_store(settings)
    return assemble_runtime(settings=settings, document_store=document_store)


def run_polling(bot: telebot.TeleBot) -> None:
    """Start Telegram long polling and perform graceful cleanup on exit."""
    logger.info("Telegram polling starting")
    configure_telegram_command_menu(bot)
    try:
        bot.infinity_polling(**POLLING_KWARGS)
    except KeyboardInterrupt:
        logger.info("Telegram polling stopped")
    except Exception as exc:
        logger.error("Telegram polling failed: %s", type(exc).__name__)
        raise
    else:
        logger.info("Telegram polling stopped")
    finally:
        bot.stop_polling()
