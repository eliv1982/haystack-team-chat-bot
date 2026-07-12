"""Tests for production runtime composition and polling boundary."""

from __future__ import annotations

import ast
import logging
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

import runtime
from config import Settings
from pinecone_preflight import PineconeIndexInfo, PineconeIndexNotFoundError


@pytest.fixture
def settings() -> Settings:
    return Settings(
        telegram_bot_token="secret-telegram-token",
        openai_api_key="test-openai-key",
        api_base_url=None,
        openai_model="test-chat-model",
        embedding_model="test-embedding-model",
        pinecone_api_key="test-pinecone-key",
        pinecone_index_name="test-index",
        pinecone_namespace="haystack-team-chat-homework",
        pinecone_dimension=1536,
        pinecone_metric="cosine",
        retrieval_top_k=50,
    )


@pytest.fixture
def document_store() -> MagicMock:
    return MagicMock(name="document_store")


@pytest.fixture
def indexing_pipeline() -> MagicMock:
    return MagicMock(name="indexing_pipeline")


@pytest.fixture
def query_pipeline() -> MagicMock:
    return MagicMock(name="query_pipeline")


@pytest.fixture
def summarization_pipeline() -> MagicMock:
    return MagicMock(name="summarization_pipeline")


@pytest.fixture
def indexing_service() -> MagicMock:
    return MagicMock(name="indexing_service")


@pytest.fixture
def retrieval_service() -> MagicMock:
    return MagicMock(name="retrieval_service")


@pytest.fixture
def summarization_service() -> MagicMock:
    return MagicMock(name="summarization_service")


@pytest.fixture
def session_store() -> MagicMock:
    return MagicMock(name="session_store")


@pytest.fixture
def telegram_application_service() -> MagicMock:
    return MagicMock(name="telegram_application_service")


@pytest.fixture
def telegram_summary_application_service() -> MagicMock:
    return MagicMock(name="telegram_summary_application_service")


@pytest.fixture
def bot() -> MagicMock:
    return MagicMock(name="bot")


@pytest.fixture
def assemble_patches(
    settings: Settings,
    document_store: MagicMock,
    indexing_pipeline: MagicMock,
    query_pipeline: MagicMock,
    summarization_pipeline: MagicMock,
    indexing_service: MagicMock,
    retrieval_service: MagicMock,
    summarization_service: MagicMock,
    session_store: MagicMock,
    telegram_application_service: MagicMock,
    telegram_summary_application_service: MagicMock,
    bot: MagicMock,
):
    with (
        patch("runtime.create_indexing_pipeline", return_value=indexing_pipeline) as mock_indexing,
        patch("runtime.create_query_pipeline", return_value=query_pipeline) as mock_query,
        patch(
            "runtime.create_summarization_pipeline",
            return_value=summarization_pipeline,
        ) as mock_summarization,
        patch("runtime.IndexingService", return_value=indexing_service) as mock_indexing_service,
        patch("runtime.RetrievalService", return_value=retrieval_service) as mock_retrieval_service,
        patch(
            "runtime.SummarizationService",
            return_value=summarization_service,
        ) as mock_summarization_service,
        patch("runtime.InMemorySessionStore", return_value=session_store) as mock_session_store,
        patch(
            "runtime.TelegramApplicationService",
            return_value=telegram_application_service,
        ) as mock_telegram_app,
        patch(
            "runtime.TelegramSummaryApplicationService",
            return_value=telegram_summary_application_service,
        ) as mock_telegram_summary,
        patch("runtime.create_configured_telegram_bot", return_value=bot) as mock_create_bot,
    ):
        yield {
            "settings": settings,
            "document_store": document_store,
            "indexing_pipeline": indexing_pipeline,
            "query_pipeline": query_pipeline,
            "summarization_pipeline": summarization_pipeline,
            "indexing_service": indexing_service,
            "retrieval_service": retrieval_service,
            "summarization_service": summarization_service,
            "session_store": session_store,
            "telegram_application_service": telegram_application_service,
            "telegram_summary_application_service": telegram_summary_application_service,
            "bot": bot,
            "mock_indexing": mock_indexing,
            "mock_query": mock_query,
            "mock_summarization": mock_summarization,
            "mock_indexing_service": mock_indexing_service,
            "mock_retrieval_service": mock_retrieval_service,
            "mock_summarization_service": mock_summarization_service,
            "mock_session_store": mock_session_store,
            "mock_telegram_app": mock_telegram_app,
            "mock_telegram_summary": mock_telegram_summary,
            "mock_create_bot": mock_create_bot,
        }


def test_import_runtime_does_not_load_settings_or_preflight() -> None:
    runtime_source = Path(__file__).resolve().parent.parent / "runtime.py"
    runtime_tree = ast.parse(runtime_source.read_text(encoding="utf-8"))
    forbidden_calls = {
        "load_settings",
        "validate_existing_pinecone_index",
        "create_pinecone_document_store",
        "create_indexing_pipeline",
        "create_query_pipeline",
        "create_summarization_pipeline",
        "create_configured_telegram_bot",
        "assemble_runtime",
        "build_runtime",
        "run_polling",
    }

    for node in runtime_tree.body:
        if not isinstance(node, ast.Expr) or not isinstance(node.value, ast.Call):
            continue
        call = node.value
        if isinstance(call.func, ast.Name) and call.func.id in forbidden_calls:
            pytest.fail(f"runtime.py must not call {call.func.id} at import time")


def test_import_runtime_does_not_create_global_dependencies() -> None:
    import runtime

    assert not hasattr(runtime, "bot")
    assert not hasattr(runtime, "store")
    assert not hasattr(runtime, "session_store")
    assert not hasattr(runtime, "runtime")


def test_assemble_runtime_wires_dependencies_once(assemble_patches: dict) -> None:
    settings = assemble_patches["settings"]
    document_store = assemble_patches["document_store"]

    result = runtime.assemble_runtime(settings=settings, document_store=document_store)

    assemble_patches["mock_indexing"].assert_called_once_with(settings, document_store)
    assemble_patches["mock_query"].assert_called_once_with(settings, document_store)
    assemble_patches["mock_summarization"].assert_called_once_with(settings)

    assemble_patches["mock_indexing_service"].assert_called_once_with(
        assemble_patches["indexing_pipeline"]
    )
    assemble_patches["mock_retrieval_service"].assert_called_once_with(
        assemble_patches["query_pipeline"],
        top_k=settings.retrieval_top_k,
    )
    assemble_patches["mock_summarization_service"].assert_called_once_with(
        assemble_patches["retrieval_service"],
        assemble_patches["summarization_pipeline"],
    )

    assemble_patches["mock_session_store"].assert_called_once_with()
    assemble_patches["mock_telegram_app"].assert_called_once_with(
        session_store=assemble_patches["session_store"],
        indexing_service=assemble_patches["indexing_service"],
    )
    assemble_patches["mock_telegram_summary"].assert_called_once_with(
        session_store=assemble_patches["session_store"],
        summarization_service=assemble_patches["summarization_service"],
    )
    assert (
        assemble_patches["mock_telegram_app"].call_args.kwargs["session_store"]
        is assemble_patches["mock_telegram_summary"].call_args.kwargs["session_store"]
    )

    assemble_patches["mock_create_bot"].assert_called_once_with(
        settings,
        assemble_patches["telegram_application_service"],
        assemble_patches["telegram_summary_application_service"],
    )

    assert isinstance(result, runtime.RuntimeComponents)
    assert result.document_store is document_store
    assert result.session_store is assemble_patches["session_store"]
    assert result.indexing_service is assemble_patches["indexing_service"]
    assert result.retrieval_service is assemble_patches["retrieval_service"]
    assert result.summarization_service is assemble_patches["summarization_service"]
    assert result.telegram_application_service is assemble_patches["telegram_application_service"]
    assert (
        result.telegram_summary_application_service
        is assemble_patches["telegram_summary_application_service"]
    )
    assert result.bot is assemble_patches["bot"]

    assemble_patches["bot"].infinity_polling.assert_not_called()
    assemble_patches["bot"].get_me.assert_not_called()
    assemble_patches["bot"].send_message.assert_not_called()


def test_assemble_runtime_does_not_mutate_inputs(
    assemble_patches: dict,
    settings: Settings,
    document_store: MagicMock,
) -> None:
    original_settings = settings
    original_store = document_store

    runtime.assemble_runtime(settings=settings, document_store=document_store)

    assert settings is original_settings
    assert document_store is original_store


def test_build_runtime_order_and_wiring(settings: Settings) -> None:
    preflight = PineconeIndexInfo(
        name="test-index",
        dimension=1536,
        metric="cosine",
        ready=True,
        status="Ready",
    )
    document_store = MagicMock(name="document_store")
    runtime_components = MagicMock(name="runtime_components")
    events: list[str] = []

    def _load_settings() -> Settings:
        events.append("load_settings")
        return settings

    def _validate(existing_settings: Settings) -> PineconeIndexInfo:
        events.append("validate_preflight")
        assert existing_settings is settings
        return preflight

    def _create_store(existing_settings: Settings) -> MagicMock:
        events.append("create_document_store")
        assert existing_settings is settings
        return document_store

    def _assemble(*, settings: Settings, document_store: MagicMock) -> MagicMock:
        events.append("assemble_runtime")
        assert settings is settings
        assert document_store is document_store
        return runtime_components

    with (
        patch("runtime.load_settings", side_effect=_load_settings) as mock_load,
        patch("runtime.validate_existing_pinecone_index", side_effect=_validate) as mock_validate,
        patch("runtime.create_pinecone_document_store", side_effect=_create_store) as mock_create,
        patch("runtime.assemble_runtime", side_effect=_assemble) as mock_assemble,
    ):
        result = runtime.build_runtime()

    assert result is runtime_components
    assert events == [
        "load_settings",
        "validate_preflight",
        "create_document_store",
        "assemble_runtime",
    ]
    mock_load.assert_called_once_with()
    mock_validate.assert_called_once_with(settings)
    mock_create.assert_called_once_with(settings)
    mock_assemble.assert_called_once_with(settings=settings, document_store=document_store)


def test_build_runtime_preflight_failure_skips_store_and_assembly(settings: Settings) -> None:
    with (
        patch("runtime.load_settings", return_value=settings),
        patch(
            "runtime.validate_existing_pinecone_index",
            side_effect=PineconeIndexNotFoundError("Pinecone index not found: test-index"),
        ),
        patch("runtime.create_pinecone_document_store") as mock_create,
        patch("runtime.assemble_runtime") as mock_assemble,
    ):
        with pytest.raises(PineconeIndexNotFoundError, match="Pinecone index not found"):
            runtime.build_runtime()

    mock_create.assert_not_called()
    mock_assemble.assert_not_called()


def test_build_runtime_store_failure_skips_assembly(settings: Settings) -> None:
    preflight = PineconeIndexInfo(
        name="test-index",
        dimension=1536,
        metric="cosine",
        ready=True,
        status="Ready",
    )
    with (
        patch("runtime.load_settings", return_value=settings),
        patch("runtime.validate_existing_pinecone_index", return_value=preflight),
        patch(
            "runtime.create_pinecone_document_store",
            side_effect=RuntimeError("store creation failed"),
        ),
        patch("runtime.assemble_runtime") as mock_assemble,
    ):
        with pytest.raises(RuntimeError, match="store creation failed"):
            runtime.build_runtime()

    mock_assemble.assert_not_called()


def test_run_polling_calls_infinity_polling_once_with_supported_kwargs(bot: MagicMock) -> None:
    runtime.run_polling(bot)

    bot.infinity_polling.assert_called_once_with(**runtime.POLLING_KWARGS)
    bot.polling.assert_not_called()
    bot.get_updates.assert_not_called()
    bot.get_me.assert_not_called()
    bot.send_message.assert_not_called()
    bot.stop_polling.assert_called_once_with()


def test_run_polling_keyboard_interrupt_performs_cleanup(bot: MagicMock) -> None:
    bot.infinity_polling.side_effect = KeyboardInterrupt()

    runtime.run_polling(bot)

    bot.infinity_polling.assert_called_once_with(**runtime.POLLING_KWARGS)
    bot.stop_polling.assert_called_once_with()


def test_run_polling_unexpected_error_is_not_swallowed(bot: MagicMock) -> None:
    bot.infinity_polling.side_effect = RuntimeError("polling failed")

    with pytest.raises(RuntimeError, match="polling failed"):
        runtime.run_polling(bot)

    bot.infinity_polling.assert_called_once_with(**runtime.POLLING_KWARGS)
    bot.stop_polling.assert_called_once_with()


def test_run_polling_does_not_log_token(bot: MagicMock, caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level(logging.INFO)
    bot.infinity_polling.side_effect = RuntimeError("secret-telegram-token leaked")

    with pytest.raises(RuntimeError):
        runtime.run_polling(bot)

    combined = caplog.text + str(bot.infinity_polling.call_args)
    assert "secret-telegram-token" not in combined


def test_build_runtime_logs_safe_fields_only(
    settings: Settings,
    caplog: pytest.LogCaptureFixture,
) -> None:
    preflight = PineconeIndexInfo(
        name="test-index",
        dimension=1536,
        metric="cosine",
        ready=True,
        status="Ready",
    )
    caplog.set_level(logging.INFO)

    with (
        patch("runtime.load_settings", return_value=settings),
        patch("runtime.validate_existing_pinecone_index", return_value=preflight),
        patch("runtime.create_pinecone_document_store", return_value=MagicMock()),
        patch("runtime.assemble_runtime", return_value=MagicMock(spec=runtime.RuntimeComponents)),
    ):
        runtime.build_runtime()

    assert settings.telegram_bot_token not in caplog.text
    assert settings.openai_api_key not in caplog.text
    assert settings.pinecone_api_key not in caplog.text
    assert "test-index" in caplog.text
    assert settings.pinecone_namespace in caplog.text
    assert settings.openai_model in caplog.text
