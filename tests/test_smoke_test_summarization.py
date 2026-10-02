"""Tests for the helpers and control flow of the live summarization smoke script.

The script itself needs real OpenAI and Pinecone access and is run by hand. These
tests cover only its own logic (corpus, polling, acceptance checks, exit codes and
cleanup); the application code it exercises is tested elsewhere.
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
from models import SummarizationRequest, SummarizationResult
from scripts import smoke_test_summarization as smoke_script
from scripts.smoke_test_summarization import (
    EXPECTED_DOCUMENT_COUNT,
    OTHER_CHAT_ID,
    SUMMARIZATION_INSTRUCTION,
    TARGET_CHAT_ID,
    SmokeCleanupError,
    SmokeGroundingError,
    SmokeIndexingError,
    SmokeRecommendationError,
    SmokeSourceIsolationError,
    SmokeSummaryStructureError,
    SmokeVerificationError,
    SmokeVisibilityTimeoutError,
    build_smoke_corpus,
    build_summarization_request,
    find_document_by_id,
    normalize_summary_text,
    run_smoke_test,
    validate_decision_and_positions,
    validate_forbidden_foreign_facts,
    validate_grounded_facts,
    validate_no_architecture_terms,
    validate_recommendation,
    validate_source_isolation,
    validate_summary_acceptance,
    validate_summary_structure,
    verify_visible_document,
    wait_for_all_documents_absent,
    wait_for_all_documents_visible,
)


def _corpus() -> smoke_script.SmokeCorpus:
    return build_smoke_corpus(
        run_id="fixed-run",
        base_message_id=200_001,
        sent_at=datetime(2026, 3, 10, 12, 0, tzinfo=timezone.utc),
    )


def _retrieved_document(expected: Document) -> Document:
    """The document as the store returns it: same content and metadata, plus a vector."""
    return Document(
        id=expected.id,
        content=expected.content,
        meta=dict(expected.meta),
        embedding=[0.1] * 1536,
    )


def _valid_summary_text() -> str:
    return (
        "Тема\n"
        "Обсуждение запуска Project Aurora.\n\n"
        "Ключевые позиции\n"
        "Анна предложила провести запуск Project Aurora во вторник в 16:00 UTC. "
        "Борис предложил провести запуск Project Aurora в среду в 10:00 UTC.\n\n"
        "Решения\n"
        "Команда зафиксировала решение: запуск Project Aurora состоится во вторник в 16:00 UTC.\n\n"
        "Следующие действия\n"
        "Марта подготовит release checklist к понедельнику, 12:00 UTC; "
        "резервный канал связи — email.\n\n"
        "Нерешенные вопросы\n"
        "Явные открытые пункты не зафиксированы.\n\n"
        "Рекомендация AI\n"
        "Следующий организационный шаг: подтвердить release checklist и канал связи перед запуском."
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


# --- corpus and request ----------------------------------------------------------


def test_build_smoke_corpus_creates_seven_messages() -> None:
    corpus = _corpus()

    assert len(corpus.messages) == 7
    assert len(corpus.documents) == 7
    assert len(corpus.expected_document_ids) == 7
    assert len(set(message.message_id for message in corpus.messages)) == 7
    assert all(message.message_id > 0 for message in corpus.messages)


def test_build_smoke_corpus_partitions_contexts() -> None:
    corpus = _corpus()

    def count(chat_id: int, session_id: str) -> int:
        return sum(
            1
            for document in corpus.documents
            if document.meta["chat_id"] == str(chat_id) and document.meta["session_id"] == session_id
        )

    assert count(TARGET_CHAT_ID, corpus.target_session_id) == 4
    assert count(TARGET_CHAT_ID, corpus.other_session_id) == 2
    assert count(OTHER_CHAT_ID, corpus.target_session_id) == 1


def test_build_smoke_corpus_identifies_key_document_ids() -> None:
    corpus = _corpus()

    assert corpus.decision_document_id == corpus.documents[2].id
    assert corpus.action_document_id == corpus.documents[3].id
    assert corpus.friday_distractor_id == corpus.documents[4].id
    assert corpus.daniel_distractor_id == corpus.documents[5].id
    assert corpus.cancelled_distractor_id == corpus.documents[6].id
    assert corpus.target_document_ids == frozenset(document.id for document in corpus.documents[:4])
    assert corpus.foreign_document_ids == frozenset(document.id for document in corpus.documents[4:])
    assert len(set(corpus.expected_document_ids)) == 7


def test_build_summarization_request_uses_exact_values() -> None:
    corpus = _corpus()

    request = build_summarization_request(corpus)

    assert request == SummarizationRequest(
        instruction=SUMMARIZATION_INSTRUCTION,
        chat_id=TARGET_CHAT_ID,
        session_id=corpus.target_session_id,
        expected_message_count=4,
    )
    # The smoke session is registered with exactly its four target messages.
    assert request.expected_message_count == len(corpus.target_document_ids)


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
        pytest.param(7, lambda docs: [], id="nothing-visible"),
        pytest.param(7, lambda docs: [_retrieved_document(docs[0])], id="only-one-of-seven"),
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
    document_store.filter_documents.side_effect = [visible, [], [], [], [], [], [], []]

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


# --- acceptance checks on the summary -------------------------------------------


def _result_with_sources(ids: tuple[str, ...]) -> SimpleNamespace:
    # SummarizationResult itself refuses duplicates and empty sets, so a stand-in is
    # used to show that the script's own check would catch them too.
    return SimpleNamespace(source_document_ids=ids, text="summary")


def test_validate_source_isolation_passes_for_exact_target_set() -> None:
    corpus = _corpus()
    result = SummarizationResult(text="summary", source_document_ids=tuple(corpus.target_document_ids))

    validate_source_isolation(result, corpus)


@pytest.mark.parametrize(
    "build_ids",
    [
        pytest.param(
            lambda c: tuple(i for i in c.target_document_ids if i != c.decision_document_id),
            id="decision-source-missing",
        ),
        pytest.param(
            lambda c: tuple(i for i in c.target_document_ids if i != c.action_document_id),
            id="action-source-missing",
        ),
        pytest.param(
            lambda c: (
                *sorted(c.target_document_ids - {c.decision_document_id, c.action_document_id})[:1],
                c.decision_document_id,
                c.action_document_id,
            ),
            id="another-target-source-missing",
        ),
        pytest.param(
            lambda c: (*c.target_document_ids, c.friday_distractor_id), id="other-session-leaked"
        ),
        pytest.param(
            lambda c: (*c.target_document_ids, c.cancelled_distractor_id), id="other-chat-leaked"
        ),
        pytest.param(
            lambda c: (*c.target_document_ids, next(iter(c.target_document_ids))), id="duplicate"
        ),
    ],
)
def test_validate_source_isolation_refuses_a_wrong_source_set(
    build_ids: Callable[[smoke_script.SmokeCorpus], tuple[str, ...]],
) -> None:
    corpus = _corpus()

    with pytest.raises(SmokeSourceIsolationError):
        validate_source_isolation(_result_with_sources(build_ids(corpus)), corpus)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "heading",
    ["Тема", "Ключевые позиции", "Решения", "Следующие действия", "Нерешенные вопросы", "Рекомендация AI"],
)
def test_validate_summary_structure_requires_each_heading(heading: str) -> None:
    text = _valid_summary_text().replace(heading, "", 1)

    with pytest.raises(SmokeSummaryStructureError):
        validate_summary_structure(text)


def test_validate_summary_structure_allows_markdown_heading() -> None:
    validate_summary_structure(_valid_summary_text().replace("Тема", "## Тема", 1))


@pytest.mark.parametrize("markup", ["\n<html>", "\n<table>", "\n| a | b |\n| --- | --- |"])
def test_validate_summary_structure_rejects_html_and_markdown_tables(markup: str) -> None:
    with pytest.raises(SmokeSummaryStructureError):
        validate_summary_structure(_valid_summary_text() + markup)


def test_validate_grounded_facts_passes_for_complete_summary() -> None:
    validate_grounded_facts(_valid_summary_text())


@pytest.mark.parametrize(
    ("validator", "original", "replacement"),
    [
        (validate_grounded_facts, "вторник в 16:00 UTC", "без даты"),
        (validate_decision_and_positions, "Марта подготовит release checklist", "Команда подготовит release checklist"),
        (validate_grounded_facts, "email", "chat"),
    ],
    ids=["tuesday-decision", "marta-action-item", "email-backup-channel"],
)
def test_grounding_checks_reject_a_summary_missing_a_target_fact(
    validator: Callable[[str], None], original: str, replacement: str
) -> None:
    with pytest.raises(SmokeGroundingError):
        validator(_valid_summary_text().replace(original, replacement))


@pytest.mark.parametrize(
    "foreign_fact",
    ["пятницу в 09:00 UTC", "09:00", "Даниил", "четвергу", "отменен", "новой даты нет"],
)
def test_validate_forbidden_foreign_facts_rejects_leakage(foreign_fact: str) -> None:
    with pytest.raises(SmokeGroundingError):
        validate_forbidden_foreign_facts(_valid_summary_text() + f"\n{foreign_fact}")


@pytest.mark.parametrize("term", ["pinecone", "embedding", "retrieval"])
def test_validate_no_architecture_terms_rejects_internal_terms(term: str) -> None:
    with pytest.raises(SmokeGroundingError):
        validate_no_architecture_terms(_valid_summary_text() + f"\n{term}", expected_document_ids=[])


def test_validate_no_architecture_terms_rejects_document_id() -> None:
    corpus = _corpus()

    with pytest.raises(SmokeGroundingError):
        validate_no_architecture_terms(
            _valid_summary_text() + f"\n{corpus.decision_document_id}",
            expected_document_ids=corpus.expected_document_ids,
        )


_RECOMMENDATION = "Следующий организационный шаг: подтвердить release checklist и канал связи перед запуском."


def test_validate_recommendation_passes_for_valid_section() -> None:
    validate_recommendation(_valid_summary_text())


def test_validate_recommendation_rejects_missing_section() -> None:
    with pytest.raises(SmokeRecommendationError):
        validate_recommendation(_valid_summary_text().replace("Рекомендация AI", "Дополнительно"))


@pytest.mark.parametrize(
    "recommendation",
    [
        "Следующий организационный шаг: зафиксировать решение на среду в 10:00 UTC.",
        "Следующий организационный шаг: перенести запуск на пятницу в 09:00 UTC.",
        "Следующий организационный шаг: Даниил подготовит release checklist к четвергу.",
    ],
    ids=["changes-the-decision", "invents-a-date", "invents-a-responsible-person"],
)
def test_validate_recommendation_rejects_a_recommendation_that_changes_the_facts(
    recommendation: str,
) -> None:
    with pytest.raises(SmokeRecommendationError):
        validate_recommendation(_valid_summary_text().replace(_RECOMMENDATION, recommendation))


def test_validate_summary_acceptance_passes_for_valid_result() -> None:
    corpus = _corpus()
    result = SummarizationResult(
        text=_valid_summary_text(), source_document_ids=tuple(corpus.target_document_ids)
    )

    validate_summary_acceptance(result, corpus)


def test_normalize_summary_text_collapses_whitespace() -> None:
    assert normalize_summary_text("  Тема\n\nКлючевые   позиции  ") == "тема ключевые позиции"


# --- run_smoke_test: exit codes and the guarantee that cleanup always runs -------


@pytest.fixture
def smoke_run(monkeypatch: pytest.MonkeyPatch) -> SimpleNamespace:
    """run_smoke_test with everything that would reach Pinecone or OpenAI replaced."""
    corpus = _corpus()
    run = SimpleNamespace(
        corpus=corpus,
        document_store=MagicMock(name="document_store"),
        indexing_service=MagicMock(name="indexing_service"),
        summarization_service=MagicMock(name="summarization_service"),
        session_document_service_cls=MagicMock(name="SessionDocumentService"),
        wait_visible=MagicMock(name="wait_visible", return_value=2),
        wait_absent=MagicMock(name="wait_absent", return_value=3),
    )
    run.indexing_service.index_messages.return_value = EXPECTED_DOCUMENT_COUNT
    run.summarization_service.summarize.return_value = SummarizationResult(
        text=_valid_summary_text(), source_document_ids=tuple(corpus.target_document_ids)
    )
    run.summarization_service_cls = MagicMock(
        name="SummarizationService", return_value=run.summarization_service
    )
    settings = MagicMock(
        pinecone_index_name="test-index",
        pinecone_namespace="haystack-team-chat-homework",
        pinecone_dimension=1536,
        pinecone_metric="cosine",
        retrieval_top_k=50,
        openai_model="gpt-4o-mini",
    )
    replacements = {
        "load_settings": lambda: settings,
        "validate_existing_pinecone_index": MagicMock(),
        "create_pinecone_document_store": lambda _settings: run.document_store,
        "create_indexing_pipeline": MagicMock(),
        "IndexingService": lambda _pipeline: run.indexing_service,
        "SessionDocumentService": run.session_document_service_cls,
        "create_summarization_pipeline": MagicMock(),
        "SummarizationService": run.summarization_service_cls,
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
    assert len(smoke_run.indexing_service.index_messages.call_args.args[0]) == EXPECTED_DOCUMENT_COUNT
    smoke_run.summarization_service.summarize.assert_called_once()
    _assert_cleaned_up(smoke_run)


def test_run_smoke_test_fails_and_cleans_up_when_not_seven_documents_were_written(
    smoke_run: SimpleNamespace,
) -> None:
    smoke_run.indexing_service.index_messages.return_value = 6

    assert run_smoke_test() == 1

    smoke_run.wait_visible.assert_not_called()
    _assert_cleaned_up(smoke_run)


def test_run_smoke_test_indexing_error_does_not_start_summarization(
    smoke_run: SimpleNamespace,
) -> None:
    smoke_run.indexing_service.index_messages.side_effect = SmokeIndexingError("indexing failed")

    assert run_smoke_test() == 1

    smoke_run.session_document_service_cls.assert_not_called()
    smoke_run.summarization_service_cls.assert_not_called()
    _assert_cleaned_up(smoke_run)


def test_run_smoke_test_visibility_error_skips_summarization(smoke_run: SimpleNamespace) -> None:
    smoke_run.wait_visible.side_effect = SmokeVisibilityTimeoutError("timeout")

    assert run_smoke_test() == 1

    smoke_run.session_document_service_cls.assert_not_called()
    smoke_run.summarization_service_cls.assert_not_called()
    _assert_cleaned_up(smoke_run)


def test_run_smoke_test_grounding_failure_is_not_masked(smoke_run: SimpleNamespace) -> None:
    smoke_run.summarization_service.summarize.return_value = SummarizationResult(
        text="Тема\nКлючевые позиции\nРешения\nСледующие действия\nНерешенные вопросы\nРекомендация AI\nforeign",
        source_document_ids=tuple(smoke_run.corpus.target_document_ids),
    )

    assert run_smoke_test() == 1

    smoke_run.summarization_service.summarize.assert_called_once()
    _assert_cleaned_up(smoke_run)


def test_run_smoke_test_fails_when_cleanup_is_not_confirmed(smoke_run: SimpleNamespace) -> None:
    smoke_run.wait_absent.side_effect = SmokeCleanupError("still visible")

    assert run_smoke_test() == 1

    smoke_run.document_store.delete_documents.assert_called_once_with(
        list(smoke_run.corpus.expected_document_ids)
    )
