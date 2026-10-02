"""Tests for the helpers and control flow of the live retrieval smoke script.

The script itself needs real OpenAI and Pinecone access and is run by hand. These
tests cover only its own logic (corpus, polling, probe validation, exit codes and
cleanup); the retrieval service it exercises is tested in test_retrieval_service.py.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import replace
from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from haystack import Document

from fakes import ticking_clock
from models import RetrievalRequest
from scripts import smoke_test_retrieval as smoke_script
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


def _corpus() -> smoke_script.SmokeCorpus:
    return build_smoke_corpus(
        run_id="fixed-run",
        base_message_id=100_001,
        sent_at=datetime(2026, 3, 10, 12, 0, tzinfo=timezone.utc),
    )


def _retrieved_document(expected: Document, *, score: float = 0.91) -> Document:
    """The document as the store returns it: same content and metadata, plus a vector."""
    return Document(
        id=expected.id,
        content=expected.content,
        meta=dict(expected.meta),
        embedding=[0.1] * 1536,
        score=score,
    )


# Each entry turns a correctly stored document into one that must be refused.
_MISMATCHES: dict[str, Callable[[Document], Document]] = {
    "other-document-id": lambda d: replace(d, id="another-id"),
    "other-content": lambda d: replace(d, content="foreign content"),
    "other-chat": lambda d: replace(d, meta={**d.meta, "chat_id": "-1"}),
    "other-session": lambda d: replace(d, meta={**d.meta, "session_id": "wrong"}),
    "other-message": lambda d: replace(d, meta={**d.meta, "message_id": "1"}),
    "missing-embedding": lambda d: replace(d, embedding=None),
    "wrong-dimension": lambda d: replace(d, embedding=[0.1] * 10),
}


# --- corpus and requests ---------------------------------------------------------


def test_build_smoke_corpus_creates_six_messages() -> None:
    corpus = _corpus()

    assert len(corpus.messages) == 6
    assert len(corpus.documents) == 6
    assert len(corpus.expected_document_ids) == 6
    assert len(set(message.message_id for message in corpus.messages)) == 6
    assert all(message.message_id > 0 for message in corpus.messages)


def test_build_smoke_corpus_partitions_contexts() -> None:
    corpus = _corpus()

    def count(chat_id: int, session_id: str) -> int:
        return sum(
            1
            for document in corpus.documents
            if document.meta["chat_id"] == str(chat_id) and document.meta["session_id"] == session_id
        )

    assert count(TARGET_CHAT_ID, corpus.target_session_id) == 3
    assert count(TARGET_CHAT_ID, corpus.other_session_id) == 2
    assert count(OTHER_CHAT_ID, corpus.target_session_id) == 1


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
        query=RETRIEVAL_QUERY, chat_id=TARGET_CHAT_ID, session_id=corpus.target_session_id
    )
    assert probe_b == RetrievalRequest(
        query=RETRIEVAL_QUERY, chat_id=TARGET_CHAT_ID, session_id=corpus.other_session_id
    )
    assert probe_c == RetrievalRequest(
        query=RETRIEVAL_QUERY, chat_id=OTHER_CHAT_ID, session_id=corpus.target_session_id
    )


def test_find_document_by_id_returns_exact_match() -> None:
    corpus = _corpus()
    documents = [corpus.documents[1], corpus.documents[0]]

    assert find_document_by_id(documents, corpus.documents[0].id) is corpus.documents[0]


# --- polling for visibility and cleanup ------------------------------------------


def test_verify_visible_document_accepts_the_document_that_was_stored() -> None:
    expected = _corpus().documents[0]

    verify_visible_document(_retrieved_document(expected), expected, 1536)


@pytest.mark.parametrize("mismatch", _MISMATCHES)
def test_verify_visible_document_refuses_a_document_that_differs_from_the_stored_one(
    mismatch: str,
) -> None:
    expected = _corpus().documents[0]

    with pytest.raises(SmokeVerificationError):
        verify_visible_document(_MISMATCHES[mismatch](_retrieved_document(expected)), expected, 1536)


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


@pytest.mark.parametrize(
    ("expected_count", "stored"),
    [
        pytest.param(6, lambda docs: [], id="nothing-visible"),
        pytest.param(6, lambda docs: [_retrieved_document(docs[0])], id="only-one-of-six"),
        pytest.param(1, lambda docs: [_retrieved_document(docs[1])], id="only-an-unrelated-document"),
        *[
            pytest.param(
                1,
                lambda docs, build=build: [build(_retrieved_document(docs[0]))],
                id=f"stored-document-differs:{name}",
            )
            for name, build in _MISMATCHES.items()
        ],
    ],
)
def test_wait_for_all_documents_visible_keeps_polling_and_never_accepts_a_wrong_result(
    expected_count: int,
    stored: Callable[[tuple[Document, ...]], list[Document]],
) -> None:
    corpus = _corpus()
    document_store = MagicMock()
    document_store.filter_documents.return_value = stored(corpus.documents)

    with pytest.raises(SmokeVisibilityTimeoutError):
        wait_for_all_documents_visible(
            document_store,
            corpus.documents[:expected_count],
            1536,
            timeout_seconds=3.0,
            poll_interval_seconds=0.0,
            sleep=lambda _: None,
            monotonic=ticking_clock(),
        )

    assert document_store.filter_documents.call_count >= 2  # it really polled


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


def test_wait_for_all_documents_absent_gives_up_when_an_expected_document_remains() -> None:
    corpus = _corpus()
    document_store = MagicMock()
    document_store.filter_documents.return_value = [_retrieved_document(corpus.documents[0])]

    with pytest.raises(SmokeCleanupError):
        wait_for_all_documents_absent(
            document_store,
            corpus.documents[:1],
            timeout_seconds=3.0,
            poll_interval_seconds=0.0,
            sleep=lambda _: None,
            monotonic=ticking_clock(),
        )

    assert document_store.filter_documents.call_count >= 2  # it really polled


# --- probe validation ------------------------------------------------------------
# Corpus order: 0 Tuesday fact (target) · 1, 2 other target messages · 3, 4 other
# session (3 is the Friday distractor) · 5 other chat (the cancelled distractor).


def _ranked(corpus: smoke_script.SmokeCorpus, *indexes: int) -> tuple[Document, ...]:
    return tuple(
        _retrieved_document(corpus.documents[index], score=0.95 - 0.05 * rank)
        for rank, index in enumerate(indexes)
    )


def test_validate_probe_a_passes_for_target_top_one() -> None:
    corpus = _corpus()

    validate_probe_a(_ranked(corpus, 0, 1, 2), corpus, top_k=50)


@pytest.mark.parametrize(
    ("indexes", "error"),
    [
        pytest.param((), SmokeProbeError, id="empty-result"),
        pytest.param((1, 0), SmokeProbeError, id="wrong-document-at-rank-zero"),
        pytest.param((0, 3), SmokeIsolationError, id="foreign-session-document"),
        pytest.param((0, 5), SmokeIsolationError, id="foreign-chat-document"),
    ],
)
def test_validate_probe_a_refuses_a_wrong_result(indexes: tuple[int, ...], error: type[Exception]) -> None:
    corpus = _corpus()

    with pytest.raises(error):
        validate_probe_a(_ranked(corpus, *indexes), corpus, top_k=50)


def test_validate_probe_a_rejects_content_mismatch() -> None:
    corpus = _corpus()
    wrong_content = replace(
        _retrieved_document(corpus.documents[0]),
        content="Project Aurora launch meeting is scheduled for Monday at 10:00 UTC.",
    )

    with pytest.raises(SmokeProbeError):
        validate_probe_a((wrong_content,), corpus, top_k=50)


def test_validate_probe_b_passes_for_friday_context() -> None:
    corpus = _corpus()

    validate_probe_b(_ranked(corpus, 3, 4), corpus, top_k=50)


@pytest.mark.parametrize("leaked_index", [0, 5], ids=["target-tuesday-leaked", "other-chat-leaked"])
def test_validate_probe_b_refuses_leakage(leaked_index: int) -> None:
    corpus = _corpus()

    with pytest.raises(SmokeIsolationError):
        validate_probe_b(_ranked(corpus, 3, leaked_index), corpus, top_k=50)


def test_validate_probe_c_passes_for_cancelled_context() -> None:
    corpus = _corpus()

    validate_probe_c(_ranked(corpus, 5), corpus, top_k=50)


@pytest.mark.parametrize("leaked_index", [0, 3], ids=["tuesday-leaked", "friday-leaked"])
def test_validate_probe_c_refuses_leakage(leaked_index: int) -> None:
    corpus = _corpus()

    with pytest.raises(SmokeIsolationError):
        validate_probe_c(_ranked(corpus, 5, leaked_index), corpus, top_k=50)


# --- run_smoke_test: exit codes and the guarantee that cleanup always runs -------


@pytest.fixture
def smoke_run(monkeypatch: pytest.MonkeyPatch) -> SimpleNamespace:
    """run_smoke_test with everything that would reach Pinecone or OpenAI replaced."""
    corpus = _corpus()
    run = SimpleNamespace(
        corpus=corpus,
        document_store=MagicMock(name="document_store"),
        indexing_service=MagicMock(name="indexing_service"),
        retrieval_service=MagicMock(name="retrieval_service"),
        create_query_pipeline=MagicMock(name="create_query_pipeline"),
        wait_visible=MagicMock(name="wait_visible", return_value=2),
        wait_absent=MagicMock(name="wait_absent", return_value=3),
    )
    run.indexing_service.index_messages.return_value = 6
    run.retrieval_service.retrieve.side_effect = [
        _ranked(corpus, 0, 1, 2),
        _ranked(corpus, 3, 4),
        _ranked(corpus, 5),
    ]
    run.retrieval_service_cls = MagicMock(name="RetrievalService", return_value=run.retrieval_service)
    settings = MagicMock(
        pinecone_index_name="test-index",
        pinecone_namespace="haystack-team-chat-homework",
        pinecone_dimension=1536,
        pinecone_metric="cosine",
        retrieval_top_k=50,
    )
    replacements = {
        "load_settings": lambda: settings,
        "validate_existing_pinecone_index": MagicMock(),
        "create_pinecone_document_store": lambda _settings: run.document_store,
        "create_indexing_pipeline": MagicMock(),
        "IndexingService": lambda _pipeline: run.indexing_service,
        "create_query_pipeline": run.create_query_pipeline,
        "RetrievalService": run.retrieval_service_cls,
        "wait_for_all_documents_visible": run.wait_visible,
        "wait_for_all_documents_absent": run.wait_absent,
        "build_smoke_corpus": lambda: corpus,
    }
    for name, replacement in replacements.items():
        monkeypatch.setattr(smoke_script, name, replacement)
    return run


def _assert_cleaned_up(run: SimpleNamespace) -> None:
    run.document_store.delete_documents.assert_called_once_with(list(run.corpus.expected_document_ids))
    run.wait_absent.assert_called_once()


def test_run_smoke_test_success(smoke_run: SimpleNamespace) -> None:
    assert run_smoke_test() == 0

    smoke_run.indexing_service.index_messages.assert_called_once()
    assert len(smoke_run.indexing_service.index_messages.call_args.args[0]) == 6
    assert smoke_run.retrieval_service.retrieve.call_count == 3
    _assert_cleaned_up(smoke_run)


def test_run_smoke_test_fails_and_cleans_up_when_not_six_documents_were_written(
    smoke_run: SimpleNamespace,
) -> None:
    smoke_run.indexing_service.index_messages.return_value = 5

    assert run_smoke_test() == 1

    smoke_run.wait_visible.assert_not_called()
    _assert_cleaned_up(smoke_run)


def test_run_smoke_test_visibility_error_skips_retrieval(smoke_run: SimpleNamespace) -> None:
    smoke_run.wait_visible.side_effect = SmokeVisibilityTimeoutError("timeout")

    assert run_smoke_test() == 1

    smoke_run.create_query_pipeline.assert_not_called()
    smoke_run.retrieval_service_cls.assert_not_called()
    _assert_cleaned_up(smoke_run)


def test_run_smoke_test_probe_a_failure_is_not_masked(smoke_run: SimpleNamespace) -> None:
    smoke_run.retrieval_service.retrieve.side_effect = [_ranked(smoke_run.corpus, 1, 0)]

    assert run_smoke_test() == 1

    assert smoke_run.retrieval_service.retrieve.call_count == 1  # probes B and C never ran
    _assert_cleaned_up(smoke_run)


def test_run_smoke_test_fails_when_cleanup_is_not_confirmed(smoke_run: SimpleNamespace) -> None:
    smoke_run.wait_absent.side_effect = SmokeCleanupError("still visible")

    assert run_smoke_test() == 1

    smoke_run.document_store.delete_documents.assert_called_once_with(
        list(smoke_run.corpus.expected_document_ids)
    )
