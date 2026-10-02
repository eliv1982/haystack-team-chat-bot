"""Tests for live summarization smoke-test helper functions."""

from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

import pytest
from haystack import Document
from haystack.dataclasses.chat_message import ChatMessage as HaystackChatMessage

from documents import chat_message_to_document
from models import SummarizationRequest, SummarizationResult
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
    wait_for_all_documents_absent,
    wait_for_all_documents_visible,
)


def _corpus() -> object:
    return build_smoke_corpus(
        run_id="fixed-run",
        base_message_id=200_001,
        sent_at=datetime(2026, 3, 10, 12, 0, tzinfo=timezone.utc),
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


def _valid_result(corpus: object) -> SummarizationResult:
    return SummarizationResult(
        text=_valid_summary_text(),
        source_document_ids=tuple(sorted(corpus.target_document_ids, key=lambda doc_id: doc_id)),
    )


def test_build_smoke_corpus_creates_seven_messages() -> None:
    corpus = _corpus()

    assert len(corpus.messages) == 7
    assert len(corpus.documents) == 7
    assert len(corpus.expected_document_ids) == 7
    assert len(set(message.message_id for message in corpus.messages)) == 7
    assert all(message.message_id > 0 for message in corpus.messages)


def test_build_smoke_corpus_partitions_contexts() -> None:
    corpus = _corpus()

    target_docs = [
        document
        for document in corpus.documents
        if document.meta["chat_id"] == str(TARGET_CHAT_ID)
        and document.meta["session_id"] == corpus.target_session_id
    ]
    foreign_session_docs = [
        document
        for document in corpus.documents
        if document.meta["chat_id"] == str(TARGET_CHAT_ID)
        and document.meta["session_id"] == corpus.other_session_id
    ]
    foreign_chat_docs = [
        document
        for document in corpus.documents
        if document.meta["chat_id"] == str(OTHER_CHAT_ID)
        and document.meta["session_id"] == corpus.target_session_id
    ]

    assert len(target_docs) == 4
    assert len(foreign_session_docs) == 2
    assert len(foreign_chat_docs) == 1


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
    assert request.instruction == SUMMARIZATION_INSTRUCTION
    # The smoke session is registered with exactly its four target messages.
    assert request.expected_message_count == len(corpus.target_document_ids) == 4


def test_build_summarization_request_does_not_mutate_inputs() -> None:
    corpus = _corpus()
    original_instruction = SUMMARIZATION_INSTRUCTION

    build_summarization_request(corpus)

    assert SUMMARIZATION_INSTRUCTION == original_instruction


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
    document_store.filter_documents.return_value = [_retrieved_document(corpus.documents[0])]

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


def test_wait_for_all_documents_visible_rejects_missing_embedding() -> None:
    corpus = _corpus()
    document_store = MagicMock()
    document_store.filter_documents.return_value = [
        _retrieved_document(corpus.documents[0], include_embedding=False)
    ]

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


def test_wait_for_all_documents_visible_rejects_wrong_dimension() -> None:
    corpus = _corpus()
    document_store = MagicMock()
    document_store.filter_documents.return_value = [
        _retrieved_document(corpus.documents[0], embedding_dimension=10)
    ]

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


def test_wait_for_all_documents_visible_rejects_content_mismatch() -> None:
    corpus = _corpus()
    retrieved = _retrieved_document(corpus.documents[0])
    retrieved.content = "foreign content"
    document_store = MagicMock()
    document_store.filter_documents.return_value = [retrieved]

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


def test_wait_for_all_documents_visible_rejects_metadata_mismatch() -> None:
    corpus = _corpus()
    retrieved = _retrieved_document(corpus.documents[0])
    retrieved.meta["session_id"] = "wrong"
    document_store = MagicMock()
    document_store.filter_documents.return_value = [retrieved]

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


def test_validate_source_isolation_passes_for_exact_target_set() -> None:
    corpus = _corpus()
    result = SummarizationResult(
        text="summary",
        source_document_ids=tuple(corpus.target_document_ids),
    )

    validate_source_isolation(result, corpus)


def test_validate_source_isolation_rejects_missing_target_source() -> None:
    corpus = _corpus()
    missing = tuple(document_id for document_id in corpus.target_document_ids if document_id != corpus.decision_document_id)
    result = SummarizationResult(text="summary", source_document_ids=missing)

    with pytest.raises(SmokeSourceIsolationError):
        validate_source_isolation(result, corpus)


def test_validate_source_isolation_rejects_foreign_session_source() -> None:
    corpus = _corpus()
    leaked = tuple(corpus.target_document_ids) + (corpus.friday_distractor_id,)
    result = SummarizationResult(text="summary", source_document_ids=leaked)

    with pytest.raises(SmokeSourceIsolationError):
        validate_source_isolation(result, corpus)


def test_validate_source_isolation_rejects_foreign_chat_source() -> None:
    corpus = _corpus()
    leaked = tuple(corpus.target_document_ids) + (corpus.cancelled_distractor_id,)
    result = SummarizationResult(text="summary", source_document_ids=leaked)

    with pytest.raises(SmokeSourceIsolationError):
        validate_source_isolation(result, corpus)


def test_validate_source_isolation_rejects_duplicate_source_id() -> None:
    corpus = _corpus()
    target_ids = list(corpus.target_document_ids)
    result = MagicMock()
    result.source_document_ids = tuple(target_ids + [target_ids[0]])

    with pytest.raises(SmokeSourceIsolationError):
        validate_source_isolation(result, corpus)


def test_validate_source_isolation_rejects_missing_decision_id() -> None:
    corpus = _corpus()
    without_decision = tuple(
        document_id
        for document_id in corpus.target_document_ids
        if document_id != corpus.decision_document_id
    )
    result = SummarizationResult(text="summary", source_document_ids=without_decision)

    with pytest.raises(SmokeSourceIsolationError):
        validate_source_isolation(result, corpus)


def test_validate_source_isolation_rejects_missing_action_id() -> None:
    corpus = _corpus()
    without_action = tuple(
        document_id
        for document_id in corpus.target_document_ids
        if document_id != corpus.action_document_id
    )
    result = SummarizationResult(text="summary", source_document_ids=without_action)

    with pytest.raises(SmokeSourceIsolationError):
        validate_source_isolation(result, corpus)


@pytest.mark.parametrize(
    ("heading",),
    [
        ("тема",),
        ("ключевые позиции",),
        ("решения",),
        ("следующие действия",),
        ("нерешенные вопросы",),
        ("рекомендация ai",),
    ],
)
def test_validate_summary_structure_requires_each_heading(heading: str) -> None:
    text = _valid_summary_text().replace(heading.title() if heading == "тема" else heading.capitalize(), "", 1)
    if heading == "тема":
        text = text.replace("Тема", "", 1)
    elif heading == "ключевые позиции":
        text = text.replace("Ключевые позиции", "", 1)
    elif heading == "решения":
        text = text.replace("Решения", "", 1)
    elif heading == "следующие действия":
        text = text.replace("Следующие действия", "", 1)
    elif heading == "нерешенные вопросы":
        text = text.replace("Нерешенные вопросы", "", 1)
    else:
        text = text.replace("Рекомендация AI", "", 1)

    with pytest.raises(SmokeSummaryStructureError):
        validate_summary_structure(text)


def test_validate_summary_structure_allows_markdown_heading() -> None:
    text = _valid_summary_text().replace("Тема", "## Тема", 1)

    validate_summary_structure(text)


def test_validate_summary_structure_rejects_html() -> None:
    with pytest.raises(SmokeSummaryStructureError):
        validate_summary_structure(_valid_summary_text() + "\n<html>")

    with pytest.raises(SmokeSummaryStructureError):
        validate_summary_structure(_valid_summary_text() + "\n<table>")


def test_validate_summary_structure_rejects_markdown_table() -> None:
    with pytest.raises(SmokeSummaryStructureError):
        validate_summary_structure(_valid_summary_text() + "\n| a | b |\n| --- | --- |")


def test_validate_grounded_facts_passes_for_complete_summary() -> None:
    validate_grounded_facts(_valid_summary_text())


def test_validate_grounded_facts_rejects_missing_final_tuesday_decision() -> None:
    text = _valid_summary_text().replace("вторник в 16:00 UTC", "без даты")

    with pytest.raises(SmokeGroundingError):
        validate_grounded_facts(text)


def test_validate_grounded_facts_rejects_missing_marta_action_item() -> None:
    text = _valid_summary_text().replace("Марта подготовит release checklist", "Команда подготовит release checklist")

    with pytest.raises(SmokeGroundingError):
        validate_decision_and_positions(text)


def test_validate_grounded_facts_rejects_missing_email() -> None:
    text = _valid_summary_text().replace("email", "chat")

    with pytest.raises(SmokeGroundingError):
        validate_grounded_facts(text)


@pytest.mark.parametrize(
    ("foreign_fact",),
    [
        ("пятницу в 09:00 UTC",),
        ("09:00",),
        ("Даниил",),
        ("четвергу",),
        ("отменен",),
        ("новой даты нет",),
    ],
)
def test_validate_forbidden_foreign_facts_rejects_leakage(foreign_fact: str) -> None:
    with pytest.raises(SmokeGroundingError):
        validate_forbidden_foreign_facts(_valid_summary_text() + f"\n{foreign_fact}")


def test_validate_no_architecture_terms_rejects_internal_terms() -> None:
    for term in ("pinecone", "embedding", "retrieval"):
        with pytest.raises(SmokeGroundingError):
            validate_no_architecture_terms(_valid_summary_text() + f"\n{term}", expected_document_ids=[])


def test_validate_no_architecture_terms_rejects_document_id() -> None:
    corpus = _corpus()

    with pytest.raises(SmokeGroundingError):
        validate_no_architecture_terms(
            _valid_summary_text() + f"\n{corpus.decision_document_id}",
            expected_document_ids=corpus.expected_document_ids,
        )


def test_validate_recommendation_passes_for_valid_section() -> None:
    validate_recommendation(_valid_summary_text())


def test_validate_recommendation_rejects_missing_section() -> None:
    text = _valid_summary_text().replace("Рекомендация AI", "Дополнительно")

    with pytest.raises(SmokeRecommendationError):
        validate_recommendation(text)


def test_validate_recommendation_rejects_decision_change() -> None:
    text = _valid_summary_text().replace(
        "Следующий организационный шаг: подтвердить release checklist и канал связи перед запуском.",
        "Следующий организационный шаг: зафиксировать решение на среду в 10:00 UTC.",
    )

    with pytest.raises(SmokeRecommendationError):
        validate_recommendation(text)


def test_validate_recommendation_rejects_invented_date() -> None:
    text = _valid_summary_text().replace(
        "Следующий организационный шаг: подтвердить release checklist и канал связи перед запуском.",
        "Следующий организационный шаг: перенести запуск на пятницу в 09:00 UTC.",
    )

    with pytest.raises(SmokeRecommendationError):
        validate_recommendation(text)


def test_validate_recommendation_rejects_invented_responsible_person() -> None:
    text = _valid_summary_text().replace(
        "Следующий организационный шаг: подтвердить release checklist и канал связи перед запуском.",
        "Следующий организационный шаг: Даниил подготовит release checklist к четвергу.",
    )

    with pytest.raises(SmokeRecommendationError):
        validate_recommendation(text)


def test_validate_summary_acceptance_passes_for_valid_result() -> None:
    corpus = _corpus()
    result = SummarizationResult(
        text=_valid_summary_text(),
        source_document_ids=tuple(corpus.target_document_ids),
    )

    validate_summary_acceptance(result, corpus)


def test_normalize_summary_text_collapses_whitespace() -> None:
    assert normalize_summary_text("  Тема\n\nКлючевые   позиции  ") == "тема ключевые позиции"


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


@patch("scripts.smoke_test_summarization.wait_for_all_documents_absent")
@patch("scripts.smoke_test_summarization.wait_for_all_documents_visible")
@patch("scripts.smoke_test_summarization.SummarizationService")
@patch("scripts.smoke_test_summarization.create_summarization_pipeline")
@patch("scripts.smoke_test_summarization.SessionDocumentService")
@patch("scripts.smoke_test_summarization.IndexingService")
@patch("scripts.smoke_test_summarization.create_indexing_pipeline")
@patch("scripts.smoke_test_summarization.create_pinecone_document_store")
@patch("scripts.smoke_test_summarization.validate_existing_pinecone_index")
@patch("scripts.smoke_test_summarization.load_settings")
def test_run_smoke_test_success(
    mock_load_settings: MagicMock,
    mock_validate: MagicMock,
    mock_create_store: MagicMock,
    mock_create_indexing_pipeline: MagicMock,
    mock_indexing_service_cls: MagicMock,
    mock_session_document_service_cls: MagicMock,
    mock_create_summarization_pipeline: MagicMock,
    mock_summarization_service_cls: MagicMock,
    mock_wait_visible: MagicMock,
    mock_wait_absent: MagicMock,
) -> None:
    settings = MagicMock()
    settings.pinecone_index_name = "test-index"
    settings.pinecone_namespace = "haystack-team-chat-homework"
    settings.pinecone_dimension = 1536
    settings.pinecone_metric = "cosine"
    settings.retrieval_top_k = 50
    settings.openai_model = "gpt-4o-mini"
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
    indexing_service.index_messages.return_value = EXPECTED_DOCUMENT_COUNT
    mock_indexing_service_cls.return_value = indexing_service
    summarization_service = MagicMock()
    corpus = _corpus()
    summarization_service.summarize.return_value = SummarizationResult(
        text=_valid_summary_text(),
        source_document_ids=tuple(corpus.target_document_ids),
    )
    mock_summarization_service_cls.return_value = summarization_service
    mock_wait_visible.return_value = 2
    mock_wait_absent.return_value = 3

    with patch("scripts.smoke_test_summarization.build_smoke_corpus", return_value=corpus):
        exit_code = run_smoke_test()

    assert exit_code == 0
    indexing_service.index_messages.assert_called_once()
    assert len(indexing_service.index_messages.call_args.args[0]) == EXPECTED_DOCUMENT_COUNT
    summarization_service.summarize.assert_called_once()
    document_store.delete_documents.assert_called_once_with(list(corpus.expected_document_ids))


@patch("scripts.smoke_test_summarization.wait_for_all_documents_absent")
@patch("scripts.smoke_test_summarization.wait_for_all_documents_visible")
@patch("scripts.smoke_test_summarization.IndexingService")
@patch("scripts.smoke_test_summarization.create_indexing_pipeline")
@patch("scripts.smoke_test_summarization.create_pinecone_document_store")
@patch("scripts.smoke_test_summarization.validate_existing_pinecone_index")
@patch("scripts.smoke_test_summarization.load_settings")
def test_run_smoke_test_rejects_non_seven_documents_written(
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
    settings.openai_model = "gpt-4o-mini"
    mock_load_settings.return_value = settings
    mock_validate.return_value = MagicMock()
    document_store = MagicMock()
    mock_create_store.return_value = document_store
    indexing_service = MagicMock()
    indexing_service.index_messages.return_value = 6
    mock_indexing_service_cls.return_value = indexing_service
    corpus = _corpus()
    mock_wait_absent.return_value = 1

    with patch("scripts.smoke_test_summarization.build_smoke_corpus", return_value=corpus):
        exit_code = run_smoke_test()

    assert exit_code == 1
    mock_wait_visible.assert_not_called()


@patch("scripts.smoke_test_summarization.wait_for_all_documents_absent")
@patch("scripts.smoke_test_summarization.wait_for_all_documents_visible")
@patch("scripts.smoke_test_summarization.SummarizationService")
@patch("scripts.smoke_test_summarization.create_summarization_pipeline")
@patch("scripts.smoke_test_summarization.SessionDocumentService")
@patch("scripts.smoke_test_summarization.IndexingService")
@patch("scripts.smoke_test_summarization.create_indexing_pipeline")
@patch("scripts.smoke_test_summarization.create_pinecone_document_store")
@patch("scripts.smoke_test_summarization.validate_existing_pinecone_index")
@patch("scripts.smoke_test_summarization.load_settings")
def test_run_smoke_test_visibility_error_skips_summarization(
    mock_load_settings: MagicMock,
    mock_validate: MagicMock,
    mock_create_store: MagicMock,
    mock_create_indexing_pipeline: MagicMock,
    mock_indexing_service_cls: MagicMock,
    mock_session_document_service_cls: MagicMock,
    mock_create_summarization_pipeline: MagicMock,
    mock_summarization_service_cls: MagicMock,
    mock_wait_visible: MagicMock,
    mock_wait_absent: MagicMock,
) -> None:
    settings = MagicMock()
    settings.pinecone_index_name = "test-index"
    settings.pinecone_namespace = "haystack-team-chat-homework"
    settings.pinecone_dimension = 1536
    settings.openai_model = "gpt-4o-mini"
    mock_load_settings.return_value = settings
    mock_validate.return_value = MagicMock()
    document_store = MagicMock()
    mock_create_store.return_value = document_store
    indexing_service = MagicMock()
    indexing_service.index_messages.return_value = EXPECTED_DOCUMENT_COUNT
    mock_indexing_service_cls.return_value = indexing_service
    mock_wait_visible.side_effect = SmokeVisibilityTimeoutError("timeout")
    corpus = _corpus()
    mock_wait_absent.return_value = 1

    with patch("scripts.smoke_test_summarization.build_smoke_corpus", return_value=corpus):
        exit_code = run_smoke_test()

    assert exit_code == 1
    mock_session_document_service_cls.assert_not_called()
    mock_summarization_service_cls.assert_not_called()


@patch("scripts.smoke_test_summarization.wait_for_all_documents_absent")
@patch("scripts.smoke_test_summarization.wait_for_all_documents_visible")
@patch("scripts.smoke_test_summarization.SummarizationService")
@patch("scripts.smoke_test_summarization.create_summarization_pipeline")
@patch("scripts.smoke_test_summarization.SessionDocumentService")
@patch("scripts.smoke_test_summarization.IndexingService")
@patch("scripts.smoke_test_summarization.create_indexing_pipeline")
@patch("scripts.smoke_test_summarization.create_pinecone_document_store")
@patch("scripts.smoke_test_summarization.validate_existing_pinecone_index")
@patch("scripts.smoke_test_summarization.load_settings")
def test_run_smoke_test_grounding_failure_is_not_masked(
    mock_load_settings: MagicMock,
    mock_validate: MagicMock,
    mock_create_store: MagicMock,
    mock_create_indexing_pipeline: MagicMock,
    mock_indexing_service_cls: MagicMock,
    mock_session_document_service_cls: MagicMock,
    mock_create_summarization_pipeline: MagicMock,
    mock_summarization_service_cls: MagicMock,
    mock_wait_visible: MagicMock,
    mock_wait_absent: MagicMock,
) -> None:
    settings = MagicMock()
    settings.pinecone_index_name = "test-index"
    settings.pinecone_namespace = "haystack-team-chat-homework"
    settings.pinecone_dimension = 1536
    settings.retrieval_top_k = 50
    settings.openai_model = "gpt-4o-mini"
    mock_load_settings.return_value = settings
    mock_validate.return_value = MagicMock()
    document_store = MagicMock()
    mock_create_store.return_value = document_store
    indexing_service = MagicMock()
    indexing_service.index_messages.return_value = EXPECTED_DOCUMENT_COUNT
    mock_indexing_service_cls.return_value = indexing_service
    corpus = _corpus()
    summarization_service = MagicMock()
    summarization_service.summarize.return_value = SummarizationResult(
        text="Тема\nКлючевые позиции\nРешения\nСледующие действия\nНерешенные вопросы\nРекомендация AI\nforeign",
        source_document_ids=tuple(corpus.target_document_ids),
    )
    mock_summarization_service_cls.return_value = summarization_service
    mock_wait_visible.return_value = 1
    mock_wait_absent.return_value = 1

    with patch("scripts.smoke_test_summarization.build_smoke_corpus", return_value=corpus):
        exit_code = run_smoke_test()

    assert exit_code == 1
    summarization_service.summarize.assert_called_once()


@patch("scripts.smoke_test_summarization.wait_for_all_documents_absent")
@patch("scripts.smoke_test_summarization.wait_for_all_documents_visible")
@patch("scripts.smoke_test_summarization.SummarizationService")
@patch("scripts.smoke_test_summarization.create_summarization_pipeline")
@patch("scripts.smoke_test_summarization.SessionDocumentService")
@patch("scripts.smoke_test_summarization.IndexingService")
@patch("scripts.smoke_test_summarization.create_indexing_pipeline")
@patch("scripts.smoke_test_summarization.create_pinecone_document_store")
@patch("scripts.smoke_test_summarization.validate_existing_pinecone_index")
@patch("scripts.smoke_test_summarization.load_settings")
def test_run_smoke_test_fails_when_cleanup_not_confirmed(
    mock_load_settings: MagicMock,
    mock_validate: MagicMock,
    mock_create_store: MagicMock,
    mock_create_indexing_pipeline: MagicMock,
    mock_indexing_service_cls: MagicMock,
    mock_session_document_service_cls: MagicMock,
    mock_create_summarization_pipeline: MagicMock,
    mock_summarization_service_cls: MagicMock,
    mock_wait_visible: MagicMock,
    mock_wait_absent: MagicMock,
) -> None:
    settings = MagicMock()
    settings.pinecone_index_name = "test-index"
    settings.pinecone_namespace = "haystack-team-chat-homework"
    settings.pinecone_dimension = 1536
    settings.retrieval_top_k = 50
    settings.openai_model = "gpt-4o-mini"
    mock_load_settings.return_value = settings
    mock_validate.return_value = MagicMock()
    document_store = MagicMock()
    mock_create_store.return_value = document_store
    indexing_service = MagicMock()
    indexing_service.index_messages.return_value = EXPECTED_DOCUMENT_COUNT
    mock_indexing_service_cls.return_value = indexing_service
    corpus = _corpus()
    summarization_service = MagicMock()
    summarization_service.summarize.return_value = SummarizationResult(
        text=_valid_summary_text(),
        source_document_ids=tuple(corpus.target_document_ids),
    )
    mock_summarization_service_cls.return_value = summarization_service
    mock_wait_visible.return_value = 1
    mock_wait_absent.side_effect = SmokeCleanupError("still visible")

    with patch("scripts.smoke_test_summarization.build_smoke_corpus", return_value=corpus):
        exit_code = run_smoke_test()

    assert exit_code == 1
    document_store.delete_documents.assert_called_once_with(list(corpus.expected_document_ids))


def test_find_document_by_id_returns_exact_match() -> None:
    corpus = _corpus()
    documents = [corpus.documents[1], corpus.documents[0]]

    assert find_document_by_id(documents, corpus.documents[0].id) is corpus.documents[0]


def test_summarization_service_orchestration_single_fetch_and_pipeline_call() -> None:
    session_documents = MagicMock()
    pipeline = MagicMock()
    corpus = _corpus()
    documents = list(corpus.documents[:4])
    session_documents.fetch.return_value = documents
    pipeline.run.return_value = {
        "llm": {"replies": [HaystackChatMessage.from_assistant(_valid_summary_text())]}
    }

    from summarization_service import SummarizationService

    service = SummarizationService(session_documents, pipeline)
    request = build_summarization_request(corpus)
    result = service.summarize(request)

    session_documents.fetch.assert_called_once()
    pipeline.run.assert_called_once()
    assert set(result.source_document_ids) == corpus.target_document_ids


def test_summarization_service_empty_session_does_not_call_pipeline() -> None:
    session_documents = MagicMock()
    pipeline = MagicMock()
    session_documents.fetch.return_value = []
    corpus = _corpus()

    from summarization_service import NoSummarizationContextError, SummarizationService

    service = SummarizationService(session_documents, pipeline)

    with pytest.raises(NoSummarizationContextError):
        service.summarize(build_summarization_request(corpus))

    pipeline.run.assert_not_called()


def test_summarization_service_fetch_exception_does_not_call_pipeline() -> None:
    session_documents = MagicMock()
    pipeline = MagicMock()
    session_documents.fetch.side_effect = RuntimeError("retrieval failed")
    corpus = _corpus()

    from summarization_service import SummarizationService

    service = SummarizationService(session_documents, pipeline)

    with pytest.raises(RuntimeError, match="retrieval failed"):
        service.summarize(build_summarization_request(corpus))

    pipeline.run.assert_not_called()


def test_summarization_service_llm_exception_does_not_retry_pipeline() -> None:
    session_documents = MagicMock()
    pipeline = MagicMock()
    corpus = _corpus()
    session_documents.fetch.return_value = list(corpus.documents[:4])
    pipeline.run.side_effect = RuntimeError("llm failed")

    from summarization_service import SummarizationService

    service = SummarizationService(session_documents, pipeline)

    with pytest.raises(RuntimeError, match="llm failed"):
        service.summarize(build_summarization_request(corpus))

    pipeline.run.assert_called_once()


def test_indexing_service_rejects_unexpected_documents_written() -> None:
    pipeline = MagicMock()
    pipeline.run.return_value = {"writer": {"documents_written": 5}}

    from indexing_service import IndexingService

    service = IndexingService(pipeline)
    documents_written = service.index_messages(list(_corpus().messages))

    assert documents_written == 5


def test_run_smoke_test_indexing_error_does_not_start_summarization() -> None:
    with (
        patch("scripts.smoke_test_summarization.load_settings") as mock_load_settings,
        patch("scripts.smoke_test_summarization.validate_existing_pinecone_index") as mock_validate,
        patch("scripts.smoke_test_summarization.create_pinecone_document_store") as mock_create_store,
        patch("scripts.smoke_test_summarization.create_indexing_pipeline"),
        patch("scripts.smoke_test_summarization.IndexingService") as mock_indexing_service_cls,
        patch(
            "scripts.smoke_test_summarization.SessionDocumentService"
        ) as mock_session_document_service_cls,
        patch("scripts.smoke_test_summarization.SummarizationService") as mock_summarization_service_cls,
        patch("scripts.smoke_test_summarization.wait_for_all_documents_absent", return_value=1),
    ):
        settings = MagicMock()
        settings.pinecone_index_name = "test-index"
        settings.pinecone_namespace = "haystack-team-chat-homework"
        settings.pinecone_dimension = 1536
        settings.openai_model = "gpt-4o-mini"
        mock_load_settings.return_value = settings
        mock_validate.return_value = MagicMock()
        mock_create_store.return_value = MagicMock()
        indexing_service = MagicMock()
        indexing_service.index_messages.side_effect = SmokeIndexingError("indexing failed")
        mock_indexing_service_cls.return_value = indexing_service
        corpus = _corpus()

        with patch("scripts.smoke_test_summarization.build_smoke_corpus", return_value=corpus):
            exit_code = run_smoke_test()

    assert exit_code == 1
    mock_session_document_service_cls.assert_not_called()
    mock_summarization_service_cls.assert_not_called()
