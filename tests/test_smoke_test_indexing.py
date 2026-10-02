"""Tests for the helpers and control flow of the live indexing smoke script.

The script itself needs real OpenAI and Pinecone access and is run by hand. These
tests cover only its own logic (polling, verification, exit codes and cleanup); the
indexing service it exercises is tested in test_indexing_service.py and
test_indexing_failure.py.
"""

from __future__ import annotations

from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from haystack import Document

from documents import chat_message_to_document
from fakes import ticking_clock
from models import ChatMessage
from scripts import smoke_test_indexing as smoke_script
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
    return chat_message_to_document(_expected_message(session_id=session_id, message_id=message_id))


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


def _other_session_document() -> Document:
    return _retrieved_document(_expected_document(session_id="other-session", message_id=99999))


def test_build_smoke_message_uses_unique_session_id() -> None:
    message_one, session_one = build_smoke_message()
    message_two, session_two = build_smoke_message()

    assert session_one != session_two
    assert message_one.chat_id == -999_000_001
    assert message_one.message_id > 0
    assert message_one.user_id > 0
    assert message_one.sent_at.tzinfo is not None


def test_build_session_filter_uses_exact_session_id() -> None:
    assert build_session_filter("smoke-abc") == {
        "field": "session_id",
        "operator": "==",
        "value": "smoke-abc",
    }


def test_find_document_by_id_returns_exact_match() -> None:
    expected = _expected_document()
    documents = [Document(id="other", content="x"), expected]

    assert find_document_by_id(documents, expected.id) is expected


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
    ids=["other-document", "missing-embedding", "wrong-dimension", "other-session"],
)
def test_verify_retrieved_document_rejects_malformed_data(retrieved: Document) -> None:
    with pytest.raises(SmokeVerificationError):
        verify_retrieved_document(retrieved, _expected_document(), 1536)


# --- polling for visibility and cleanup ------------------------------------------


def test_wait_for_document_visible_succeeds_after_retries() -> None:
    expected = _expected_document()
    document_store = MagicMock()
    found = _retrieved_document(expected)
    document_store.filter_documents.side_effect = [
        [_other_session_document()],
        [_other_session_document()],
        [found],
    ]

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


@pytest.mark.parametrize(
    "stored",
    [[], [_other_session_document()]],
    ids=["nothing-visible", "only-an-unrelated-document"],
)
def test_wait_for_document_visible_keeps_polling_until_it_gives_up(stored: list[Document]) -> None:
    expected = _expected_document()
    document_store = MagicMock()
    document_store.filter_documents.return_value = stored

    with pytest.raises(SmokeVisibilityTimeoutError):
        wait_for_document_visible(
            document_store,
            expected,
            expected.meta["session_id"],
            1536,
            timeout_seconds=3.0,
            poll_interval_seconds=0.0,
            sleep=lambda _: None,
            monotonic=ticking_clock(),
        )

    assert document_store.filter_documents.call_count >= 2  # it really polled


def test_wait_for_document_visible_refuses_a_visible_document_that_is_wrong() -> None:
    expected = _expected_document()
    document_store = MagicMock()
    document_store.filter_documents.return_value = [
        _retrieved_document(expected, embedding_dimension=10)
    ]

    with pytest.raises(SmokeVerificationError):
        wait_for_document_visible(
            document_store,
            expected,
            expected.meta["session_id"],
            1536,
            timeout_seconds=3.0,
            poll_interval_seconds=0.0,
            sleep=lambda _: None,
            monotonic=ticking_clock(),
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
    document_store.filter_documents.return_value = [_other_session_document()]

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


def test_wait_for_document_absent_gives_up_while_the_document_remains() -> None:
    expected = _expected_document()
    document_store = MagicMock()
    document_store.filter_documents.return_value = [_retrieved_document(expected)]

    with pytest.raises(SmokeCleanupError):
        wait_for_document_absent(
            document_store,
            expected.id,
            expected.meta["session_id"],
            timeout_seconds=3.0,
            poll_interval_seconds=0.0,
            sleep=lambda _: None,
            monotonic=ticking_clock(),
        )

    assert document_store.filter_documents.call_count >= 2  # it really polled


# --- run_smoke_test: exit codes and the guarantee that cleanup always runs -------


@pytest.fixture
def smoke_run(monkeypatch: pytest.MonkeyPatch) -> SimpleNamespace:
    """run_smoke_test with everything that would reach Pinecone or OpenAI replaced."""
    expected = _expected_document(session_id="smoke-run")
    run = SimpleNamespace(
        expected=expected,
        document_store=MagicMock(name="document_store"),
        indexing_service=MagicMock(name="indexing_service"),
        wait_visible=MagicMock(name="wait_visible", return_value=(2, _retrieved_document(expected))),
        wait_absent=MagicMock(name="wait_absent", return_value=3),
    )
    run.indexing_service.index_messages.return_value = 1
    settings = MagicMock(
        pinecone_index_name="test-index",
        pinecone_namespace="haystack-team-chat-homework",
        pinecone_dimension=1536,
        pinecone_metric="cosine",
    )
    replacements = {
        "load_settings": lambda: settings,
        "validate_existing_pinecone_index": MagicMock(),
        "create_pinecone_document_store": lambda _settings: run.document_store,
        "create_indexing_pipeline": MagicMock(),
        "IndexingService": lambda _pipeline: run.indexing_service,
        "build_smoke_message": lambda: (_expected_message(session_id="smoke-run"), "smoke-run"),
        "chat_message_to_document": lambda _message: expected,
        "wait_for_document_visible": run.wait_visible,
        "wait_for_document_absent": run.wait_absent,
    }
    for name, replacement in replacements.items():
        monkeypatch.setattr(smoke_script, name, replacement)
    return run


def _assert_cleaned_up(run: SimpleNamespace) -> None:
    run.document_store.delete_documents.assert_called_once_with([run.expected.id])
    run.wait_absent.assert_called_once()


def test_run_smoke_test_success(smoke_run: SimpleNamespace) -> None:
    assert run_smoke_test() == 0

    smoke_run.indexing_service.index_messages.assert_called_once()
    _assert_cleaned_up(smoke_run)


def test_run_smoke_test_fails_and_cleans_up_when_documents_written_is_not_one(
    smoke_run: SimpleNamespace,
) -> None:
    smoke_run.indexing_service.index_messages.return_value = 0

    assert run_smoke_test() == 1

    smoke_run.wait_visible.assert_not_called()
    _assert_cleaned_up(smoke_run)


def test_run_smoke_test_cleans_up_after_a_verification_failure(smoke_run: SimpleNamespace) -> None:
    smoke_run.wait_visible.side_effect = SmokeVisibilityTimeoutError("timeout")

    assert run_smoke_test() == 1

    _assert_cleaned_up(smoke_run)


def test_run_smoke_test_fails_when_cleanup_is_not_confirmed(smoke_run: SimpleNamespace) -> None:
    smoke_run.wait_absent.side_effect = SmokeCleanupError("still visible")

    assert run_smoke_test() == 1

    smoke_run.document_store.delete_documents.assert_called_once_with([smoke_run.expected.id])
