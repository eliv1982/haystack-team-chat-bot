"""Tests for live retrieval smoke-test helper functions."""

from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

import pytest
from haystack import Document

from documents import chat_message_to_document
from models import RetrievalRequest
from scripts.smoke_test_retrieval import (
    OTHER_CHAT_ID,
    RETRIEVAL_QUERY,
    TARGET_CHAT_ID,
    SmokeCleanupError,
    SmokeIsolationError,
    SmokeProbeError,
    SmokeVerificationError,
    SmokeVisibilityTimeoutError,
    build_retrieval_requests,
    build_smoke_corpus,
    find_document_by_id,
    run_smoke_test,
    validate_probe_a,
    validate_probe_b,
    validate_probe_c,
    verify_visible_document,
    wait_for_all_documents_absent,
    wait_for_all_documents_visible,
)


def _corpus() -> object:
    return build_smoke_corpus(
        run_id="fixed-run",
        base_message_id=100_001,
        sent_at=datetime(2026, 3, 10, 12, 0, tzinfo=timezone.utc),
    )


def _retrieved_document(
    expected: Document,
    *,
    include_embedding: bool = True,
    embedding_dimension: int = 1536,
    score: float = 0.91,
) -> Document:
    embedding = [0.1] * embedding_dimension if include_embedding else None
    return Document(
        id=expected.id,
        content=expected.content,
        meta=dict(expected.meta),
        embedding=embedding,
        score=score,
    )


def test_build_smoke_corpus_creates_six_messages() -> None:
    corpus = _corpus()

    assert len(corpus.messages) == 6
    assert len(corpus.documents) == 6
    assert len(corpus.expected_document_ids) == 6
    assert len(set(message.message_id for message in corpus.messages)) == 6
    assert all(message.message_id > 0 for message in corpus.messages)


def test_build_smoke_corpus_partitions_contexts() -> None:
    corpus = _corpus()

    target_session_docs = [
        document
        for document in corpus.documents
        if document.meta["chat_id"] == str(TARGET_CHAT_ID)
        and document.meta["session_id"] == corpus.target_session_id
    ]
    other_session_docs = [
        document
        for document in corpus.documents
        if document.meta["chat_id"] == str(TARGET_CHAT_ID)
        and document.meta["session_id"] == corpus.other_session_id
    ]
    other_chat_docs = [
        document
        for document in corpus.documents
        if document.meta["chat_id"] == str(OTHER_CHAT_ID)
        and document.meta["session_id"] == corpus.target_session_id
    ]

    assert len(target_session_docs) == 3
    assert len(other_session_docs) == 2
    assert len(other_chat_docs) == 1


def test_build_smoke_corpus_identifies_key_document_ids() -> None:
    corpus = _corpus()

    assert corpus.target_fact_id == corpus.documents[0].id
    assert corpus.friday_distractor_id == corpus.documents[3].id
    assert corpus.cancelled_distractor_id == corpus.documents[5].id
    assert len(set(corpus.expected_document_ids)) == 6


def test_build_retrieval_requests_use_expected_chat_session_pairs() -> None:
    corpus = _corpus()

    probe_a, probe_b, probe_c = build_retrieval_requests(corpus)

    assert probe_a == RetrievalRequest(
        query=RETRIEVAL_QUERY,
        chat_id=TARGET_CHAT_ID,
        session_id=corpus.target_session_id,
    )
    assert probe_b == RetrievalRequest(
        query=RETRIEVAL_QUERY,
        chat_id=TARGET_CHAT_ID,
        session_id=corpus.other_session_id,
    )
    assert probe_c == RetrievalRequest(
        query=RETRIEVAL_QUERY,
        chat_id=OTHER_CHAT_ID,
        session_id=corpus.target_session_id,
    )


def test_wait_for_all_documents_visible_succeeds_after_retries() -> None:
    corpus = _corpus()
    document_store = MagicMock()
    state = {"ready": False}

    def filter_documents(*, filters: dict[str, object]) -> list[Document]:
        if not state["ready"]:
            return []
        session_id = filters["conditions"][1]["value"]  # type: ignore[index]
        return [
            _retrieved_document(document)
            for document in corpus.documents
            if document.meta["session_id"] == session_id
        ]

    document_store.filter_documents.side_effect = filter_documents

    attempts = wait_for_all_documents_visible(
        document_store,
        corpus.documents,
        1536,
        timeout_seconds=1.0,
        poll_interval_seconds=0.0,
        sleep=lambda _: state.update({"ready": True}),
        monotonic=lambda: 0.0,
    )

    assert attempts == 2


def test_wait_for_all_documents_visible_rejects_partial_set() -> None:
    corpus = _corpus()
    document_store = MagicMock()
    partial = [_retrieved_document(corpus.documents[0])]
    document_store.filter_documents.return_value = partial

    with pytest.raises(SmokeVisibilityTimeoutError):
        wait_for_all_documents_visible(
            document_store,
            corpus.documents,
            1536,
            timeout_seconds=0.0,
            poll_interval_seconds=0.0,
            sleep=lambda _: None,
            monotonic=lambda: 0.0,
        )


def test_wait_for_all_documents_visible_ignores_unrelated_documents() -> None:
    corpus = _corpus()
    unrelated = chat_message_to_document(corpus.messages[0])
    object.__setattr__(unrelated, "meta", {**unrelated.meta, "session_id": "foreign"})
    document_store = MagicMock()
    document_store.filter_documents.return_value = [_retrieved_document(unrelated)]

    with pytest.raises(SmokeVisibilityTimeoutError):
        wait_for_all_documents_visible(
            document_store,
            corpus.documents[:1],
            1536,
            timeout_seconds=0.0,
            poll_interval_seconds=0.0,
            sleep=lambda _: None,
            monotonic=lambda: 0.0,
        )


def test_verify_visible_document_rejects_missing_embedding() -> None:
    corpus = _corpus()
    retrieved = _retrieved_document(corpus.documents[0], include_embedding=False)

    with pytest.raises(SmokeVerificationError, match="embedding"):
        verify_visible_document(retrieved, corpus.documents[0], 1536)


def test_verify_visible_document_rejects_wrong_dimension() -> None:
    corpus = _corpus()
    retrieved = _retrieved_document(corpus.documents[0], embedding_dimension=10)

    with pytest.raises(SmokeVerificationError, match="dimension"):
        verify_visible_document(retrieved, corpus.documents[0], 1536)


def test_verify_visible_document_rejects_metadata_mismatch() -> None:
    corpus = _corpus()
    retrieved = _retrieved_document(corpus.documents[0])
    retrieved.meta["session_id"] = "wrong"

    with pytest.raises(SmokeVerificationError, match="session_id"):
        verify_visible_document(retrieved, corpus.documents[0], 1536)


def test_validate_probe_a_passes_for_target_top_one() -> None:
    corpus = _corpus()
    results = (
        _retrieved_document(corpus.documents[0], score=0.95),
        _retrieved_document(corpus.documents[1], score=0.80),
        _retrieved_document(corpus.documents[2], score=0.70),
    )

    validate_probe_a(results, corpus, top_k=50)


def test_validate_probe_a_rejects_empty_result() -> None:
    corpus = _corpus()

    with pytest.raises(SmokeProbeError):
        validate_probe_a((), corpus, top_k=50)


def test_validate_probe_a_rejects_wrong_rank_zero() -> None:
    corpus = _corpus()
    results = (
        _retrieved_document(corpus.documents[1], score=0.95),
        _retrieved_document(corpus.documents[0], score=0.80),
    )

    with pytest.raises(SmokeProbeError):
        validate_probe_a(results, corpus, top_k=50)


def test_validate_probe_a_rejects_foreign_session_id() -> None:
    corpus = _corpus()
    foreign = _retrieved_document(corpus.documents[3], score=0.99)
    results = (_retrieved_document(corpus.documents[0], score=0.95), foreign)

    with pytest.raises(SmokeIsolationError):
        validate_probe_a(results, corpus, top_k=50)


def test_validate_probe_a_rejects_foreign_chat_id() -> None:
    corpus = _corpus()
    foreign = _retrieved_document(corpus.documents[5], score=0.99)
    results = (_retrieved_document(corpus.documents[0], score=0.95), foreign)

    with pytest.raises(SmokeIsolationError):
        validate_probe_a(results, corpus, top_k=50)


def test_validate_probe_a_rejects_content_mismatch() -> None:
    corpus = _corpus()
    wrong_content = Document(
        id=corpus.documents[0].id,
        content="Project Aurora launch meeting is scheduled for Monday at 10:00 UTC.",
        meta=dict(corpus.documents[0].meta),
        score=0.95,
    )

    with pytest.raises(SmokeProbeError):
        validate_probe_a((wrong_content,), corpus, top_k=50)


def test_validate_probe_b_passes_for_friday_context() -> None:
    corpus = _corpus()
    results = (
        _retrieved_document(corpus.documents[3], score=0.94),
        _retrieved_document(corpus.documents[4], score=0.70),
    )

    validate_probe_b(results, corpus, top_k=50)


def test_validate_probe_b_rejects_target_tuesday_leakage() -> None:
    corpus = _corpus()
    results = (
        _retrieved_document(corpus.documents[3], score=0.94),
        _retrieved_document(corpus.documents[0], score=0.70),
    )

    with pytest.raises(SmokeIsolationError):
        validate_probe_b(results, corpus, top_k=50)


def test_validate_probe_b_rejects_other_chat_leakage() -> None:
    corpus = _corpus()
    results = (
        _retrieved_document(corpus.documents[3], score=0.94),
        _retrieved_document(corpus.documents[5], score=0.70),
    )

    with pytest.raises(SmokeIsolationError):
        validate_probe_b(results, corpus, top_k=50)


def test_validate_probe_c_passes_for_cancelled_context() -> None:
    corpus = _corpus()
    results = (_retrieved_document(corpus.documents[5], score=0.93),)

    validate_probe_c(results, corpus, top_k=50)


def test_validate_probe_c_rejects_tuesday_leakage() -> None:
    corpus = _corpus()
    results = (
        _retrieved_document(corpus.documents[5], score=0.93),
        _retrieved_document(corpus.documents[0], score=0.70),
    )

    with pytest.raises(SmokeIsolationError):
        validate_probe_c(results, corpus, top_k=50)


def test_validate_probe_c_rejects_friday_leakage() -> None:
    corpus = _corpus()
    results = (
        _retrieved_document(corpus.documents[5], score=0.93),
        _retrieved_document(corpus.documents[3], score=0.70),
    )

    with pytest.raises(SmokeIsolationError):
        validate_probe_c(results, corpus, top_k=50)


def test_wait_for_all_documents_absent_succeeds_after_retries() -> None:
    corpus = _corpus()
    document_store = MagicMock()
    visible = [_retrieved_document(corpus.documents[0])]
    document_store.filter_documents.side_effect = [visible, [], [], [], [], [], []]

    attempts = wait_for_all_documents_absent(
        document_store,
        corpus.documents,
        timeout_seconds=1.0,
        poll_interval_seconds=0.0,
        sleep=lambda _: None,
        monotonic=lambda: 0.0,
    )

    assert attempts == 2


def test_wait_for_all_documents_absent_ignores_unrelated_documents() -> None:
    corpus = _corpus()
    document_store = MagicMock()
    document_store.filter_documents.return_value = [_retrieved_document(corpus.documents[1])]

    attempts = wait_for_all_documents_absent(
        document_store,
        corpus.documents[:1],
        timeout_seconds=1.0,
        poll_interval_seconds=0.0,
        sleep=lambda _: None,
        monotonic=lambda: 0.0,
    )

    assert attempts == 1


def test_wait_for_all_documents_absent_times_out_when_expected_id_remains() -> None:
    corpus = _corpus()
    document_store = MagicMock()
    document_store.filter_documents.return_value = [_retrieved_document(corpus.documents[0])]

    with pytest.raises(SmokeCleanupError):
        wait_for_all_documents_absent(
            document_store,
            corpus.documents[:1],
            timeout_seconds=0.0,
            poll_interval_seconds=0.0,
            sleep=lambda _: None,
            monotonic=lambda: 0.0,
        )


@patch("scripts.smoke_test_retrieval.wait_for_all_documents_absent")
@patch("scripts.smoke_test_retrieval.wait_for_all_documents_visible")
@patch("scripts.smoke_test_retrieval.RetrievalService")
@patch("scripts.smoke_test_retrieval.create_query_pipeline")
@patch("scripts.smoke_test_retrieval.IndexingService")
@patch("scripts.smoke_test_retrieval.create_indexing_pipeline")
@patch("scripts.smoke_test_retrieval.create_pinecone_document_store")
@patch("scripts.smoke_test_retrieval.validate_existing_pinecone_index")
@patch("scripts.smoke_test_retrieval.load_settings")
def test_run_smoke_test_success(
    mock_load_settings: MagicMock,
    mock_validate: MagicMock,
    mock_create_store: MagicMock,
    mock_create_indexing_pipeline: MagicMock,
    mock_indexing_service_cls: MagicMock,
    mock_create_query_pipeline: MagicMock,
    mock_retrieval_service_cls: MagicMock,
    mock_wait_visible: MagicMock,
    mock_wait_absent: MagicMock,
) -> None:
    settings = MagicMock()
    settings.pinecone_index_name = "test-index"
    settings.pinecone_namespace = "haystack-team-chat-homework"
    settings.pinecone_dimension = 1536
    settings.pinecone_metric = "cosine"
    settings.retrieval_top_k = 50
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
    indexing_service = MagicMock()
    indexing_service.index_messages.return_value = 6
    mock_indexing_service_cls.return_value = indexing_service
    retrieval_service = MagicMock()
    corpus = _corpus()
    retrieval_service.retrieve.side_effect = [
        tuple(_retrieved_document(document) for document in corpus.documents[:3]),
        tuple(_retrieved_document(document) for document in corpus.documents[3:5]),
        (_retrieved_document(corpus.documents[5]),),
    ]
    mock_retrieval_service_cls.return_value = retrieval_service
    mock_wait_visible.return_value = 2
    mock_wait_absent.return_value = 3

    with patch("scripts.smoke_test_retrieval.build_smoke_corpus", return_value=corpus):
        exit_code = run_smoke_test()

    assert exit_code == 0
    indexing_service.index_messages.assert_called_once()
    assert len(indexing_service.index_messages.call_args.args[0]) == 6
    assert retrieval_service.retrieve.call_count == 3
    document_store.delete_documents.assert_called_once_with(list(corpus.expected_document_ids))


@patch("scripts.smoke_test_retrieval.wait_for_all_documents_absent")
@patch("scripts.smoke_test_retrieval.wait_for_all_documents_visible")
@patch("scripts.smoke_test_retrieval.IndexingService")
@patch("scripts.smoke_test_retrieval.create_indexing_pipeline")
@patch("scripts.smoke_test_retrieval.create_pinecone_document_store")
@patch("scripts.smoke_test_retrieval.validate_existing_pinecone_index")
@patch("scripts.smoke_test_retrieval.load_settings")
def test_run_smoke_test_rejects_non_six_documents_written(
    mock_load_settings: MagicMock,
    mock_validate: MagicMock,
    mock_create_store: MagicMock,
    mock_create_indexing_pipeline: MagicMock,
    mock_indexing_service_cls: MagicMock,
    mock_wait_visible: MagicMock,
    mock_wait_absent: MagicMock,
) -> None:
    settings = MagicMock()
    settings.pinecone_index_name = "test-index"
    settings.pinecone_namespace = "haystack-team-chat-homework"
    settings.pinecone_dimension = 1536
    mock_load_settings.return_value = settings
    mock_validate.return_value = MagicMock()
    document_store = MagicMock()
    mock_create_store.return_value = document_store
    indexing_service = MagicMock()
    indexing_service.index_messages.return_value = 5
    mock_indexing_service_cls.return_value = indexing_service
    corpus = _corpus()
    mock_wait_absent.return_value = 1

    with patch("scripts.smoke_test_retrieval.build_smoke_corpus", return_value=corpus):
        exit_code = run_smoke_test()

    assert exit_code == 1
    mock_wait_visible.assert_not_called()


@patch("scripts.smoke_test_retrieval.wait_for_all_documents_absent")
@patch("scripts.smoke_test_retrieval.wait_for_all_documents_visible")
@patch("scripts.smoke_test_retrieval.RetrievalService")
@patch("scripts.smoke_test_retrieval.create_query_pipeline")
@patch("scripts.smoke_test_retrieval.IndexingService")
@patch("scripts.smoke_test_retrieval.create_indexing_pipeline")
@patch("scripts.smoke_test_retrieval.create_pinecone_document_store")
@patch("scripts.smoke_test_retrieval.validate_existing_pinecone_index")
@patch("scripts.smoke_test_retrieval.load_settings")
def test_run_smoke_test_visibility_error_skips_retrieval(
    mock_load_settings: MagicMock,
    mock_validate: MagicMock,
    mock_create_store: MagicMock,
    mock_create_indexing_pipeline: MagicMock,
    mock_indexing_service_cls: MagicMock,
    mock_create_query_pipeline: MagicMock,
    mock_retrieval_service_cls: MagicMock,
    mock_wait_visible: MagicMock,
    mock_wait_absent: MagicMock,
) -> None:
    settings = MagicMock()
    settings.pinecone_index_name = "test-index"
    settings.pinecone_namespace = "haystack-team-chat-homework"
    settings.pinecone_dimension = 1536
    mock_load_settings.return_value = settings
    mock_validate.return_value = MagicMock()
    document_store = MagicMock()
    mock_create_store.return_value = document_store
    indexing_service = MagicMock()
    indexing_service.index_messages.return_value = 6
    mock_indexing_service_cls.return_value = indexing_service
    mock_wait_visible.side_effect = SmokeVisibilityTimeoutError("timeout")
    corpus = _corpus()
    mock_wait_absent.return_value = 1

    with patch("scripts.smoke_test_retrieval.build_smoke_corpus", return_value=corpus):
        exit_code = run_smoke_test()

    assert exit_code == 1
    mock_create_query_pipeline.assert_not_called()
    mock_retrieval_service_cls.assert_not_called()


@patch("scripts.smoke_test_retrieval.wait_for_all_documents_absent")
@patch("scripts.smoke_test_retrieval.wait_for_all_documents_visible")
@patch("scripts.smoke_test_retrieval.RetrievalService")
@patch("scripts.smoke_test_retrieval.create_query_pipeline")
@patch("scripts.smoke_test_retrieval.IndexingService")
@patch("scripts.smoke_test_retrieval.create_indexing_pipeline")
@patch("scripts.smoke_test_retrieval.create_pinecone_document_store")
@patch("scripts.smoke_test_retrieval.validate_existing_pinecone_index")
@patch("scripts.smoke_test_retrieval.load_settings")
def test_run_smoke_test_probe_a_failure_is_not_masked(
    mock_load_settings: MagicMock,
    mock_validate: MagicMock,
    mock_create_store: MagicMock,
    mock_create_indexing_pipeline: MagicMock,
    mock_indexing_service_cls: MagicMock,
    mock_create_query_pipeline: MagicMock,
    mock_retrieval_service_cls: MagicMock,
    mock_wait_visible: MagicMock,
    mock_wait_absent: MagicMock,
) -> None:
    settings = MagicMock()
    settings.pinecone_index_name = "test-index"
    settings.pinecone_namespace = "haystack-team-chat-homework"
    settings.pinecone_dimension = 1536
    settings.retrieval_top_k = 50
    mock_load_settings.return_value = settings
    mock_validate.return_value = MagicMock()
    document_store = MagicMock()
    mock_create_store.return_value = document_store
    indexing_service = MagicMock()
    indexing_service.index_messages.return_value = 6
    mock_indexing_service_cls.return_value = indexing_service
    corpus = _corpus()
    retrieval_service = MagicMock()
    retrieval_service.retrieve.return_value = (
        _retrieved_document(corpus.documents[1], score=0.95),
        _retrieved_document(corpus.documents[0], score=0.80),
    )
    mock_retrieval_service_cls.return_value = retrieval_service
    mock_wait_visible.return_value = 1
    mock_wait_absent.return_value = 1

    with patch("scripts.smoke_test_retrieval.build_smoke_corpus", return_value=corpus):
        exit_code = run_smoke_test()

    assert exit_code == 1
    assert retrieval_service.retrieve.call_count == 1


@patch("scripts.smoke_test_retrieval.wait_for_all_documents_absent")
@patch("scripts.smoke_test_retrieval.wait_for_all_documents_visible")
@patch("scripts.smoke_test_retrieval.RetrievalService")
@patch("scripts.smoke_test_retrieval.create_query_pipeline")
@patch("scripts.smoke_test_retrieval.IndexingService")
@patch("scripts.smoke_test_retrieval.create_indexing_pipeline")
@patch("scripts.smoke_test_retrieval.create_pinecone_document_store")
@patch("scripts.smoke_test_retrieval.validate_existing_pinecone_index")
@patch("scripts.smoke_test_retrieval.load_settings")
def test_run_smoke_test_fails_when_cleanup_not_confirmed(
    mock_load_settings: MagicMock,
    mock_validate: MagicMock,
    mock_create_store: MagicMock,
    mock_create_indexing_pipeline: MagicMock,
    mock_indexing_service_cls: MagicMock,
    mock_create_query_pipeline: MagicMock,
    mock_retrieval_service_cls: MagicMock,
    mock_wait_visible: MagicMock,
    mock_wait_absent: MagicMock,
) -> None:
    settings = MagicMock()
    settings.pinecone_index_name = "test-index"
    settings.pinecone_namespace = "haystack-team-chat-homework"
    settings.pinecone_dimension = 1536
    settings.retrieval_top_k = 50
    mock_load_settings.return_value = settings
    mock_validate.return_value = MagicMock()
    document_store = MagicMock()
    mock_create_store.return_value = document_store
    indexing_service = MagicMock()
    indexing_service.index_messages.return_value = 6
    mock_indexing_service_cls.return_value = indexing_service
    corpus = _corpus()
    retrieval_service = MagicMock()
    retrieval_service.retrieve.side_effect = [
        tuple(_retrieved_document(document) for document in corpus.documents[:3]),
        tuple(_retrieved_document(document) for document in corpus.documents[3:5]),
        (_retrieved_document(corpus.documents[5]),),
    ]
    mock_retrieval_service_cls.return_value = retrieval_service
    mock_wait_visible.return_value = 1
    mock_wait_absent.side_effect = SmokeCleanupError("still visible")

    with patch("scripts.smoke_test_retrieval.build_smoke_corpus", return_value=corpus):
        exit_code = run_smoke_test()

    assert exit_code == 1
    document_store.delete_documents.assert_called_once_with(list(corpus.expected_document_ids))


def test_find_document_by_id_returns_exact_match() -> None:
    corpus = _corpus()
    documents = [corpus.documents[1], corpus.documents[0]]

    assert find_document_by_id(documents, corpus.documents[0].id) is corpus.documents[0]
