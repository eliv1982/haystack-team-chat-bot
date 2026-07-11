"""Tests for live smoke-test helper functions."""

from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

import pytest
from haystack import Document

from documents import chat_message_to_document
from models import ChatMessage
from scripts.smoke_test_indexing import (
    SmokeCleanupError,
    SmokeVerificationError,
    SmokeVisibilityTimeoutError,
    build_session_filter,
    build_smoke_message,
    find_document_by_id,
    run_smoke_test,
    verify_retrieved_document,
    wait_for_document_absent,
    wait_for_document_visible,
)


def _expected_message(
    session_id: str = "smoke-test-session",
    *,
    message_id: int = 12345,
) -> ChatMessage:
    return ChatMessage(
        chat_id=-999_000_001,
        message_id=message_id,
        user_id=900_000_001,
        session_id=session_id,
        author_name="Smoke Test Bot",
        username=None,
        text="haystack-team-chat-bot smoke-test marker",
        sent_at=datetime(2024, 1, 15, 12, 30, tzinfo=timezone.utc),
    )


def _expected_document(
    session_id: str = "smoke-test-session",
    *,
    message_id: int = 12345,
) -> Document:
    return chat_message_to_document(
        _expected_message(session_id=session_id, message_id=message_id)
    )


def _retrieved_document(
    expected: Document,
    *,
    include_embedding: bool = True,
    embedding_dimension: int = 1536,
) -> Document:
    embedding = [0.1] * embedding_dimension if include_embedding else None
    return Document(
        id=expected.id,
        content=expected.content,
        meta=dict(expected.meta),
        embedding=embedding,
    )


def test_build_smoke_message_uses_unique_session_id() -> None:
    message_one, session_one = build_smoke_message()
    message_two, session_two = build_smoke_message()

    assert session_one != session_two
    assert message_one.chat_id == -999_000_001
    assert message_one.message_id > 0
    assert message_one.user_id > 0
    assert message_one.sent_at.tzinfo is not None


def test_wait_for_document_visible_succeeds_after_retries() -> None:
    expected = _expected_document()
    document_store = MagicMock()
    other = _retrieved_document(_expected_document(session_id="other-session", message_id=99999))
    found = _retrieved_document(expected)
    document_store.filter_documents.side_effect = [[other], [other], [found]]

    attempts, retrieved = wait_for_document_visible(
        document_store,
        expected,
        expected.meta["session_id"],
        1536,
        timeout_seconds=1.0,
        poll_interval_seconds=0.0,
        sleep=lambda _: None,
        monotonic=lambda: 0.0,
    )

    assert attempts == 3
    assert retrieved.id == expected.id


def test_wait_for_document_visible_ignores_unrelated_documents() -> None:
    expected = _expected_document()
    document_store = MagicMock()
    document_store.filter_documents.return_value = [
        _retrieved_document(_expected_document(session_id="other-session", message_id=99999))
    ]

    with pytest.raises(SmokeVisibilityTimeoutError):
        wait_for_document_visible(
            document_store,
            expected,
            expected.meta["session_id"],
            1536,
            timeout_seconds=0.0,
            poll_interval_seconds=0.0,
            sleep=lambda _: None,
            monotonic=lambda: 0.0,
        )


def test_wait_for_document_visible_times_out() -> None:
    expected = _expected_document()
    document_store = MagicMock()
    document_store.filter_documents.return_value = []

    with pytest.raises(SmokeVisibilityTimeoutError):
        wait_for_document_visible(
            document_store,
            expected,
            expected.meta["session_id"],
            1536,
            timeout_seconds=0.0,
            poll_interval_seconds=0.0,
            sleep=lambda _: None,
            monotonic=lambda: 0.0,
        )


def test_wait_for_document_absent_succeeds_after_retries() -> None:
    expected = _expected_document()
    document_store = MagicMock()
    document_store.filter_documents.side_effect = [[_retrieved_document(expected)], []]

    attempts = wait_for_document_absent(
        document_store,
        expected.id,
        expected.meta["session_id"],
        timeout_seconds=1.0,
        poll_interval_seconds=0.0,
        sleep=lambda _: None,
        monotonic=lambda: 0.0,
    )

    assert attempts == 2


def test_wait_for_document_absent_ignores_unrelated_documents() -> None:
    expected = _expected_document()
    document_store = MagicMock()
    other = _retrieved_document(_expected_document(session_id="other-session", message_id=99999))
    document_store.filter_documents.return_value = [other]

    attempts = wait_for_document_absent(
        document_store,
        expected.id,
        expected.meta["session_id"],
        timeout_seconds=1.0,
        poll_interval_seconds=0.0,
        sleep=lambda _: None,
        monotonic=lambda: 0.0,
    )

    assert attempts == 1


def test_wait_for_document_absent_times_out() -> None:
    expected = _expected_document()
    document_store = MagicMock()
    document_store.filter_documents.return_value = [_retrieved_document(expected)]

    with pytest.raises(SmokeCleanupError):
        wait_for_document_absent(
            document_store,
            expected.id,
            expected.meta["session_id"],
            timeout_seconds=0.0,
            poll_interval_seconds=0.0,
            sleep=lambda _: None,
            monotonic=lambda: 0.0,
        )


@pytest.mark.parametrize(
    "retrieved",
    [
        Document(id="other", content="x", meta={"session_id": "smoke-test-session"}),
        _retrieved_document(_expected_document(), include_embedding=False),
        _retrieved_document(_expected_document(), embedding_dimension=10),
        Document(
            id=_expected_document().id,
            content=_expected_document().content,
            meta={"session_id": "different-session"},
        ),
    ],
)
def test_verify_retrieved_document_rejects_malformed_data(retrieved: Document) -> None:
    expected = _expected_document()

    with pytest.raises(SmokeVerificationError):
        verify_retrieved_document(retrieved, expected, 1536)


def test_find_document_by_id_returns_exact_match() -> None:
    expected = _expected_document()
    documents = [
        Document(id="other", content="x"),
        expected,
    ]

    assert find_document_by_id(documents, expected.id) is expected


def test_build_session_filter_uses_exact_session_id() -> None:
    assert build_session_filter("smoke-abc") == {
        "field": "session_id",
        "operator": "==",
        "value": "smoke-abc",
    }


@patch("scripts.smoke_test_indexing.wait_for_document_absent")
@patch("scripts.smoke_test_indexing.wait_for_document_visible")
@patch("scripts.smoke_test_indexing.IndexingService")
@patch("scripts.smoke_test_indexing.create_indexing_pipeline")
@patch("scripts.smoke_test_indexing.create_pinecone_document_store")
@patch("scripts.smoke_test_indexing.validate_existing_pinecone_index")
@patch("scripts.smoke_test_indexing.load_settings")
def test_run_smoke_test_success(
    mock_load_settings: MagicMock,
    mock_validate: MagicMock,
    mock_create_store: MagicMock,
    mock_create_pipeline: MagicMock,
    mock_indexing_service_cls: MagicMock,
    mock_wait_visible: MagicMock,
    mock_wait_absent: MagicMock,
) -> None:
    settings = MagicMock()
    settings.pinecone_index_name = "test-index"
    settings.pinecone_namespace = "haystack-team-chat-homework"
    settings.pinecone_dimension = 1536
    settings.pinecone_metric = "cosine"
    mock_load_settings.return_value = settings
    mock_validate.return_value = MagicMock(
        name="test-index",
        dimension=1536,
        metric="cosine",
        ready=True,
        status="Ready",
    )
    document_store = MagicMock()
    mock_create_store.return_value = document_store
    service = MagicMock()
    service.index_messages.return_value = 1
    mock_indexing_service_cls.return_value = service
    expected = _expected_document(session_id="smoke-run")
    mock_wait_visible.return_value = (2, _retrieved_document(expected))
    mock_wait_absent.return_value = 3

    with (
        patch("scripts.smoke_test_indexing.build_smoke_message") as mock_build_message,
        patch("scripts.smoke_test_indexing.chat_message_to_document", return_value=expected),
    ):
        mock_build_message.return_value = (_expected_message(session_id="smoke-run"), "smoke-run")
        exit_code = run_smoke_test()

    assert exit_code == 0
    service.index_messages.assert_called_once()
    document_store.delete_documents.assert_called_once_with([expected.id])
    mock_wait_absent.assert_called_once()


@patch("scripts.smoke_test_indexing.wait_for_document_absent")
@patch("scripts.smoke_test_indexing.wait_for_document_visible")
@patch("scripts.smoke_test_indexing.IndexingService")
@patch("scripts.smoke_test_indexing.create_indexing_pipeline")
@patch("scripts.smoke_test_indexing.create_pinecone_document_store")
@patch("scripts.smoke_test_indexing.validate_existing_pinecone_index")
@patch("scripts.smoke_test_indexing.load_settings")
def test_run_smoke_test_fails_when_documents_written_not_one(
    mock_load_settings: MagicMock,
    mock_validate: MagicMock,
    mock_create_store: MagicMock,
    mock_create_pipeline: MagicMock,
    mock_indexing_service_cls: MagicMock,
    mock_wait_visible: MagicMock,
    mock_wait_absent: MagicMock,
) -> None:
    settings = MagicMock()
    settings.pinecone_index_name = "test-index"
    settings.pinecone_namespace = "haystack-team-chat-homework"
    settings.pinecone_dimension = 1536
    settings.pinecone_metric = "cosine"
    mock_load_settings.return_value = settings
    mock_validate.return_value = MagicMock()
    document_store = MagicMock()
    mock_create_store.return_value = document_store
    service = MagicMock()
    service.index_messages.return_value = 0
    mock_indexing_service_cls.return_value = service
    expected = _expected_document(session_id="smoke-run")
    mock_wait_absent.return_value = 1

    with (
        patch("scripts.smoke_test_indexing.build_smoke_message") as mock_build_message,
        patch("scripts.smoke_test_indexing.chat_message_to_document", return_value=expected),
    ):
        mock_build_message.return_value = (_expected_message(session_id="smoke-run"), "smoke-run")
        exit_code = run_smoke_test()

    assert exit_code == 1
    mock_wait_visible.assert_not_called()
    document_store.delete_documents.assert_called_once_with([expected.id])


@patch("scripts.smoke_test_indexing.wait_for_document_absent")
@patch("scripts.smoke_test_indexing.wait_for_document_visible")
@patch("scripts.smoke_test_indexing.IndexingService")
@patch("scripts.smoke_test_indexing.create_indexing_pipeline")
@patch("scripts.smoke_test_indexing.create_pinecone_document_store")
@patch("scripts.smoke_test_indexing.validate_existing_pinecone_index")
@patch("scripts.smoke_test_indexing.load_settings")
def test_run_smoke_test_runs_cleanup_on_verification_failure(
    mock_load_settings: MagicMock,
    mock_validate: MagicMock,
    mock_create_store: MagicMock,
    mock_create_pipeline: MagicMock,
    mock_indexing_service_cls: MagicMock,
    mock_wait_visible: MagicMock,
    mock_wait_absent: MagicMock,
) -> None:
    settings = MagicMock()
    settings.pinecone_index_name = "test-index"
    settings.pinecone_namespace = "haystack-team-chat-homework"
    settings.pinecone_dimension = 1536
    settings.pinecone_metric = "cosine"
    mock_load_settings.return_value = settings
    mock_validate.return_value = MagicMock()
    document_store = MagicMock()
    mock_create_store.return_value = document_store
    service = MagicMock()
    service.index_messages.return_value = 1
    mock_indexing_service_cls.return_value = service
    expected = _expected_document(session_id="smoke-run")
    mock_wait_visible.side_effect = SmokeVisibilityTimeoutError("timeout")
    mock_wait_absent.return_value = 1

    with (
        patch("scripts.smoke_test_indexing.build_smoke_message") as mock_build_message,
        patch("scripts.smoke_test_indexing.chat_message_to_document", return_value=expected),
    ):
        mock_build_message.return_value = (_expected_message(session_id="smoke-run"), "smoke-run")
        exit_code = run_smoke_test()

    assert exit_code == 1
    document_store.delete_documents.assert_called_once_with([expected.id])


@patch("scripts.smoke_test_indexing.wait_for_document_absent")
@patch("scripts.smoke_test_indexing.wait_for_document_visible")
@patch("scripts.smoke_test_indexing.IndexingService")
@patch("scripts.smoke_test_indexing.create_indexing_pipeline")
@patch("scripts.smoke_test_indexing.create_pinecone_document_store")
@patch("scripts.smoke_test_indexing.validate_existing_pinecone_index")
@patch("scripts.smoke_test_indexing.load_settings")
def test_run_smoke_test_fails_when_cleanup_not_confirmed(
    mock_load_settings: MagicMock,
    mock_validate: MagicMock,
    mock_create_store: MagicMock,
    mock_create_pipeline: MagicMock,
    mock_indexing_service_cls: MagicMock,
    mock_wait_visible: MagicMock,
    mock_wait_absent: MagicMock,
) -> None:
    settings = MagicMock()
    settings.pinecone_index_name = "test-index"
    settings.pinecone_namespace = "haystack-team-chat-homework"
    settings.pinecone_dimension = 1536
    settings.pinecone_metric = "cosine"
    mock_load_settings.return_value = settings
    mock_validate.return_value = MagicMock()
    document_store = MagicMock()
    mock_create_store.return_value = document_store
    service = MagicMock()
    service.index_messages.return_value = 1
    mock_indexing_service_cls.return_value = service
    expected = _expected_document(session_id="smoke-run")
    mock_wait_visible.return_value = (1, _retrieved_document(expected))
    mock_wait_absent.side_effect = SmokeCleanupError("still visible")

    with (
        patch("scripts.smoke_test_indexing.build_smoke_message") as mock_build_message,
        patch("scripts.smoke_test_indexing.chat_message_to_document", return_value=expected),
    ):
        mock_build_message.return_value = (_expected_message(session_id="smoke-run"), "smoke-run")
        exit_code = run_smoke_test()

    assert exit_code == 1
    document_store.delete_documents.assert_called_once_with([expected.id])
