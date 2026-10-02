"""Live smoke test for grounded summarization with chat/session isolation."""

from __future__ import annotations

import re
import sys
import time
import uuid
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from haystack import Document
from haystack.document_stores.types import DocumentStore

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import Settings, load_settings  # noqa: E402
from document_store import create_pinecone_document_store  # noqa: E402
from documents import chat_message_to_document  # noqa: E402
from indexing_service import IndexingService  # noqa: E402
from models import ChatMessage, SummarizationRequest, SummarizationResult  # noqa: E402
from pinecone_preflight import PineconeIndexInfo, validate_existing_pinecone_index  # noqa: E402
from pipelines import (  # noqa: E402
    create_indexing_pipeline,
    create_summarization_pipeline,
)
from session_documents import SessionDocumentService  # noqa: E402
from summarization_service import SummarizationService  # noqa: E402

TARGET_CHAT_ID = -1_003_004_005_001
OTHER_CHAT_ID = -1_003_004_005_999
EXPECTED_DOCUMENT_COUNT = 7

SUMMARIZATION_INSTRUCTION = (
    "Подведи итог обсуждения на русском языке. "
    "Отрази тему, ключевые позиции участников, зафиксированное решение, "
    "следующие действия и нерешенные вопросы. "
    "Добавь краткую рекомендацию AI о следующем организационном шаге."
)

ANNA_POSITION_TEXT = (
    "Анна предложила провести запуск Project Aurora во вторник в 16:00 UTC."
)
BORIS_POSITION_TEXT = (
    "Борис предложил провести запуск Project Aurora в среду в 10:00 UTC, "
    "чтобы оставить дополнительный день на проверку."
)
DECISION_TEXT = (
    "После обсуждения команда зафиксировала решение: "
    "запуск Project Aurora состоится во вторник в 16:00 UTC."
)
ACTION_TEXT = (
    "Марта подготовит release checklist к понедельнику, 12:00 UTC; "
    "резервный канал связи — email."
)
FRIDAY_DISTRACTOR_TEXT = (
    "В этой сессии запуск Project Aurora назначен на пятницу в 09:00 UTC."
)
DANIEL_DISTRACTOR_TEXT = "Даниил подготовит release checklist к четвергу."
CANCELLED_DISTRACTOR_TEXT = "Запуск Project Aurora отменен, новой даты нет."

VISIBILITY_TIMEOUT_SECONDS = 90.0
CLEANUP_TIMEOUT_SECONDS = 90.0
POLL_INTERVAL_SECONDS = 2.0

REQUIRED_HEADINGS: tuple[str, ...] = (
    "тема",
    "ключевые позиции",
    "решения",
    "следующие действия",
    "нерешенные вопросы",
    "рекомендация ai",
)

REQUIRED_GROUNDED_FACTS: tuple[str, ...] = (
    "project aurora",
    "анна",
    "вторник",
    "16:00",
    "борис",
    "сред",
    "10:00",
    "марта",
    "release checklist",
    "понедельник",
    "12:00",
    "email",
)

FORBIDDEN_FOREIGN_FACTS: tuple[str, ...] = (
    "пятниц",
    "09:00",
    "даниил",
    "четверг",
    "отмен",
    "новой даты нет",
)

FORBIDDEN_ARCHITECTURE_TERMS: tuple[str, ...] = (
    "pinecone",
    "embedding",
    "retrieval",
)


class SmokeTestError(Exception):
    """Base class for summarization smoke test failures."""


class SmokeIndexingError(SmokeTestError):
    """Raised when indexing did not write the expected number of documents."""


class SmokeVerificationError(SmokeTestError):
    """Raised when retrieved document data is invalid."""


class SmokeVisibilityTimeoutError(SmokeTestError):
    """Raised when expected documents did not become visible in time."""


class SmokeSourceIsolationError(SmokeTestError):
    """Raised when summarization source IDs leak foreign context."""


class SmokeSummaryStructureError(SmokeTestError):
    """Raised when summary structure is invalid."""


class SmokeGroundingError(SmokeTestError):
    """Raised when summary text is not grounded in target context."""


class SmokeRecommendationError(SmokeTestError):
    """Raised when AI recommendation section is invalid."""


class SmokeCleanupError(SmokeTestError):
    """Raised when cleanup could not be confirmed."""


@dataclass(frozen=True, slots=True)
class SmokeCorpus:
    """Smoke corpus metadata and expected documents for one run."""

    run_id: str
    target_session_id: str
    other_session_id: str
    messages: tuple[ChatMessage, ...]
    documents: tuple[Document, ...]
    expected_document_ids: tuple[str, ...]
    target_document_ids: frozenset[str]
    foreign_document_ids: frozenset[str]
    decision_document_id: str
    action_document_id: str
    friday_distractor_id: str
    daniel_distractor_id: str
    cancelled_distractor_id: str


def build_chat_session_filter(chat_id: int, session_id: str) -> dict[str, Any]:
    """Build a Haystack metadata filter for a chat/session pair."""
    return {
        "operator": "AND",
        "conditions": [
            {
                "field": "meta.chat_id",
                "operator": "==",
                "value": str(chat_id),
            },
            {
                "field": "meta.session_id",
                "operator": "==",
                "value": session_id,
            },
        ],
    }


def build_smoke_corpus(
    *,
    run_id: str | None = None,
    base_message_id: int | None = None,
    sent_at: datetime | None = None,
) -> SmokeCorpus:
    """Create the seven-message smoke corpus for one run."""
    resolved_run_id = run_id or str(uuid.uuid4())
    resolved_base_message_id = base_message_id or (int(uuid.uuid4().int % 1_000_000_000) + 1)
    base_time = sent_at or datetime(2026, 3, 10, 12, 0, tzinfo=timezone.utc)

    target_session_id = f"summarization-smoke-target-{resolved_run_id}"
    other_session_id = f"summarization-smoke-other-{resolved_run_id}"

    message_specs: list[tuple[int, int, str, str, int, str]] = [
        (TARGET_CHAT_ID, resolved_base_message_id, target_session_id, "Анна Planner", 920_001, ANNA_POSITION_TEXT),
        (
            TARGET_CHAT_ID,
            resolved_base_message_id + 1,
            target_session_id,
            "Борис Planner",
            920_002,
            BORIS_POSITION_TEXT,
        ),
        (
            TARGET_CHAT_ID,
            resolved_base_message_id + 2,
            target_session_id,
            "Команда Facilitator",
            920_003,
            DECISION_TEXT,
        ),
        (
            TARGET_CHAT_ID,
            resolved_base_message_id + 3,
            target_session_id,
            "Марта Ops",
            920_004,
            ACTION_TEXT,
        ),
        (
            TARGET_CHAT_ID,
            resolved_base_message_id + 4,
            other_session_id,
            "Nora Alt",
            920_005,
            FRIDAY_DISTRACTOR_TEXT,
        ),
        (
            TARGET_CHAT_ID,
            resolved_base_message_id + 5,
            other_session_id,
            "Daniel Alt",
            920_006,
            DANIEL_DISTRACTOR_TEXT,
        ),
        (
            OTHER_CHAT_ID,
            resolved_base_message_id + 6,
            target_session_id,
            "Evan Remote",
            920_007,
            CANCELLED_DISTRACTOR_TEXT,
        ),
    ]

    messages: list[ChatMessage] = []
    for index, (chat_id, message_id, session_id, author_name, user_id, text) in enumerate(message_specs):
        messages.append(
            ChatMessage(
                chat_id=chat_id,
                message_id=message_id,
                user_id=user_id,
                session_id=session_id,
                author_name=author_name,
                username=None,
                text=text,
                sent_at=base_time + timedelta(minutes=index),
            )
        )

    documents = tuple(chat_message_to_document(message) for message in messages)
    expected_document_ids = tuple(document.id for document in documents)
    target_document_ids = frozenset(document.id for document in documents[:4])
    foreign_document_ids = frozenset(document.id for document in documents[4:])

    return SmokeCorpus(
        run_id=resolved_run_id,
        target_session_id=target_session_id,
        other_session_id=other_session_id,
        messages=tuple(messages),
        documents=documents,
        expected_document_ids=expected_document_ids,
        target_document_ids=target_document_ids,
        foreign_document_ids=foreign_document_ids,
        decision_document_id=documents[2].id,
        action_document_id=documents[3].id,
        friday_distractor_id=documents[4].id,
        daniel_distractor_id=documents[5].id,
        cancelled_distractor_id=documents[6].id,
    )


def build_summarization_request(corpus: SmokeCorpus) -> SummarizationRequest:
    """Build the production summarization request for the target session."""
    return SummarizationRequest(
        instruction=SUMMARIZATION_INSTRUCTION,
        chat_id=TARGET_CHAT_ID,
        session_id=corpus.target_session_id,
        expected_message_count=len(corpus.target_document_ids),
    )


def find_document_by_id(documents: Sequence[Document], document_id: str) -> Document | None:
    """Return the document with the exact ID, if present."""
    for document in documents:
        if document.id == document_id:
            return document
    return None


def verify_visible_document(
    retrieved: Document,
    expected: Document,
    expected_dimension: int,
) -> None:
    """Validate that a visible document matches the expected smoke record."""
    if retrieved.id != expected.id:
        raise SmokeVerificationError("Retrieved document ID does not match expected ID")
    if retrieved.content != expected.content:
        raise SmokeVerificationError("Retrieved document content does not match expected content")

    for key in ("chat_id", "session_id", "message_id"):
        if retrieved.meta.get(key) != expected.meta.get(key):
            raise SmokeVerificationError(f"Retrieved document metadata mismatch for field: {key}")

    embedding = retrieved.embedding
    if embedding is None:
        raise SmokeVerificationError("Retrieved document is missing an embedding")
    if len(embedding) != expected_dimension:
        raise SmokeVerificationError(
            f"Retrieved embedding dimension mismatch: expected {expected_dimension}, actual {len(embedding)}"
        )


def wait_for_all_documents_visible(
    document_store: DocumentStore,
    expected_documents: Sequence[Document],
    expected_dimension: int,
    *,
    timeout_seconds: float = VISIBILITY_TIMEOUT_SECONDS,
    poll_interval_seconds: float = POLL_INTERVAL_SECONDS,
    sleep: Callable[[float], None] = time.sleep,
    monotonic: Callable[[], float] = time.monotonic,
) -> int:
    """Poll until all expected documents are visible and valid."""
    deadline = monotonic() + timeout_seconds
    attempts = 0

    while monotonic() < deadline:
        attempts += 1
        all_visible = True
        for expected in expected_documents:
            filters = build_chat_session_filter(
                int(expected.meta["chat_id"]),
                str(expected.meta["session_id"]),
            )
            documents = document_store.filter_documents(filters=filters)
            retrieved = find_document_by_id(documents, expected.id)
            if retrieved is None:
                all_visible = False
                break
            try:
                verify_visible_document(retrieved, expected, expected_dimension)
            except SmokeVerificationError:
                all_visible = False
                break

        if all_visible:
            return attempts
        sleep(poll_interval_seconds)

    raise SmokeVisibilityTimeoutError(
        f"Expected documents did not all become visible within {timeout_seconds:.0f} seconds"
    )


def wait_for_all_documents_absent(
    document_store: DocumentStore,
    expected_documents: Sequence[Document],
    *,
    timeout_seconds: float = CLEANUP_TIMEOUT_SECONDS,
    poll_interval_seconds: float = POLL_INTERVAL_SECONDS,
    sleep: Callable[[float], None] = time.sleep,
    monotonic: Callable[[], float] = time.monotonic,
) -> int:
    """Poll until all expected document IDs are no longer visible."""
    deadline = monotonic() + timeout_seconds
    attempts = 0

    while monotonic() < deadline:
        attempts += 1
        any_visible = False
        for expected in expected_documents:
            filters = build_chat_session_filter(
                int(expected.meta["chat_id"]),
                str(expected.meta["session_id"]),
            )
            documents = document_store.filter_documents(filters=filters)
            if find_document_by_id(documents, expected.id) is not None:
                any_visible = True
                break

        if not any_visible:
            return attempts
        sleep(poll_interval_seconds)

    raise SmokeCleanupError(
        f"Expected documents remained visible after cleanup within {timeout_seconds:.0f} seconds"
    )


def normalize_summary_text(text: str) -> str:
    """Normalize summary text for acceptance checks only."""
    return re.sub(r"\s+", " ", text.strip().lower())


def validate_source_isolation(result: SummarizationResult, corpus: SmokeCorpus) -> None:
    """Validate that source IDs match exactly the four target documents."""
    source_ids = result.source_document_ids
    if not source_ids:
        raise SmokeSourceIsolationError("source_document_ids must not be empty")

    if len(source_ids) != len(set(source_ids)):
        raise SmokeSourceIsolationError("source_document_ids must not contain duplicates")

    source_set = set(source_ids)
    if source_set != set(corpus.target_document_ids):
        raise SmokeSourceIsolationError("source_document_ids must match the four target document IDs")

    leaked = source_set & corpus.foreign_document_ids
    if leaked:
        raise SmokeSourceIsolationError("source_document_ids must not include foreign document IDs")

    if corpus.decision_document_id not in source_set:
        raise SmokeSourceIsolationError("source_document_ids must include the decision document ID")
    if corpus.action_document_id not in source_set:
        raise SmokeSourceIsolationError("source_document_ids must include the action document ID")

    forbidden = {
        corpus.friday_distractor_id,
        corpus.daniel_distractor_id,
        corpus.cancelled_distractor_id,
    }
    if source_set & forbidden:
        raise SmokeSourceIsolationError("source_document_ids must not include distractor document IDs")


def validate_summary_structure(text: str) -> None:
    """Validate that the summary contains required semantic headings."""
    normalized = normalize_summary_text(text)
    if "<table" in normalized or "<html" in normalized:
        raise SmokeSummaryStructureError("summary must not contain HTML")
    if re.search(r"\|\s*[-:]+\s*\|", text):
        raise SmokeSummaryStructureError("summary must not contain Markdown tables")

    for heading in REQUIRED_HEADINGS:
        if heading not in normalized:
            raise SmokeSummaryStructureError(f"summary missing required heading: {heading}")


def validate_grounded_facts(text: str) -> None:
    """Validate that required grounded facts appear in the summary."""
    normalized = normalize_summary_text(text)
    for fact in REQUIRED_GROUNDED_FACTS:
        if fact not in normalized:
            raise SmokeGroundingError(f"summary missing required grounded fact: {fact}")


def validate_decision_and_positions(text: str) -> None:
    """Validate that positions and the final decision are represented correctly."""
    normalized = normalize_summary_text(text)

    if "анна" not in normalized or "борис" not in normalized:
        raise SmokeGroundingError("summary must mention both Anna and Boris positions")

    if "вторник" not in normalized or "16:00" not in normalized:
        raise SmokeGroundingError("summary must mention the Tuesday 16:00 UTC decision")

    if "сред" not in normalized or "10:00" not in normalized:
        raise SmokeGroundingError("summary must mention Boris Wednesday 10:00 UTC proposal")

    if re.search(r"(решени|зафиксир).{0,80}(сред|10:00)", normalized):
        raise SmokeGroundingError("summary must not present Wednesday 10:00 UTC as the accepted decision")

    if "марта" not in normalized or "release checklist" not in normalized:
        raise SmokeGroundingError("summary must associate the checklist with Marta")

    if "понедельник" not in normalized or "12:00" not in normalized:
        raise SmokeGroundingError("summary must mention Monday 12:00 UTC checklist deadline")

    if "email" not in normalized:
        raise SmokeGroundingError("summary must mention email as the backup channel")


def validate_forbidden_foreign_facts(text: str) -> None:
    """Validate that foreign-session and foreign-chat facts are absent."""
    normalized = normalize_summary_text(text)
    for fact in FORBIDDEN_FOREIGN_FACTS:
        if fact in normalized:
            raise SmokeGroundingError(f"summary contains forbidden foreign fact: {fact}")


def validate_no_architecture_terms(text: str, *, expected_document_ids: Sequence[str]) -> None:
    """Validate that internal architecture terms and document IDs are absent."""
    normalized = normalize_summary_text(text)
    for term in FORBIDDEN_ARCHITECTURE_TERMS:
        if term in normalized:
            raise SmokeGroundingError(f"summary contains forbidden architecture term: {term}")

    for document_id in expected_document_ids:
        if document_id in text or document_id.lower() in normalized:
            raise SmokeGroundingError("summary must not contain document IDs")


def _recommendation_section(normalized: str) -> str:
    marker = "рекомендация"
    index = normalized.find(marker)
    if index == -1:
        return ""
    return normalized[index:]


def validate_recommendation(text: str) -> None:
    """Validate the AI recommendation section without requiring exact wording."""
    normalized = normalize_summary_text(text)
    recommendation = _recommendation_section(normalized)
    if not recommendation:
        raise SmokeRecommendationError("summary must include an AI recommendation section")

    if "ai" not in recommendation and "ии" not in recommendation:
        raise SmokeRecommendationError("recommendation must be explicitly marked as AI")

    if re.search(r"(решени|зафиксир).{0,80}(сред|пятниц|четверг|отмен)", recommendation):
        raise SmokeRecommendationError("recommendation must not change the accepted decision")

    if re.search(r"(даниил|пятниц|четверг|09:00|отмен)", recommendation):
        raise SmokeRecommendationError("recommendation must not invent foreign facts")

    if re.search(r"(ответственн|подготовит|checklist).{0,60}(даниил)", recommendation):
        raise SmokeRecommendationError("recommendation must not invent a new responsible person")


def validate_summary_acceptance(
    result: SummarizationResult,
    corpus: SmokeCorpus,
) -> None:
    """Run all summary acceptance checks on the summarization result."""
    validate_source_isolation(result, corpus)
    text = result.text
    validate_summary_structure(text)
    validate_grounded_facts(text)
    validate_decision_and_positions(text)
    validate_forbidden_foreign_facts(text)
    validate_no_architecture_terms(text, expected_document_ids=corpus.expected_document_ids)
    validate_recommendation(text)


def _shorten_document_id(document_id: str) -> str:
    if len(document_id) <= 12:
        return document_id
    return f"{document_id[:8]}...{document_id[-4:]}"


def _print_preflight_result(index_info: PineconeIndexInfo, settings: Settings) -> None:
    print("Preflight PASS")
    print(f"Index: {index_info.name}")
    print(f"Namespace: {settings.pinecone_namespace}")
    print(f"Dimension: expected={settings.pinecone_dimension} actual={index_info.dimension}")
    print(f"Metric: expected={settings.pinecone_metric} actual={index_info.metric}")
    print(f"Ready: {index_info.ready}")
    if index_info.status is not None:
        print(f"Status: {index_info.status}")


def _print_manual_cleanup_hint(
    *,
    index_name: str,
    namespace: str,
    document_ids: Sequence[str],
) -> None:
    print("Manual cleanup required")
    print(f"Index: {index_name}")
    print(f"Namespace: {namespace}")
    for document_id in document_ids:
        print(f"Document ID: {document_id}")


def run_smoke_test() -> int:
    """Execute the live summarization smoke test and return a process exit code."""
    settings = load_settings()
    document_store = None
    corpus: SmokeCorpus | None = None
    main_error: Exception | None = None
    cleanup_error: Exception | None = None
    visibility_attempts = 0
    cleanup_attempts = 0
    documents_written: int | None = None
    summary_result: SummarizationResult | None = None

    try:
        index_info = validate_existing_pinecone_index(settings)
        _print_preflight_result(index_info, settings)

        document_store = create_pinecone_document_store(settings)
        indexing_pipeline = create_indexing_pipeline(settings, document_store)
        indexing_service = IndexingService(indexing_pipeline)

        corpus = build_smoke_corpus()
        print(f"Run ID: {corpus.run_id}")
        print(f"Target session ID: {corpus.target_session_id}")
        print(f"Other session ID: {corpus.other_session_id}")
        print(f"OpenAI model: {settings.openai_model}")
        for document_id in corpus.expected_document_ids:
            print(f"Document ID: {_shorten_document_id(document_id)}")

        documents_written = indexing_service.index_messages(list(corpus.messages))
        print(f"Documents written: {documents_written}")
        if documents_written != EXPECTED_DOCUMENT_COUNT:
            raise SmokeIndexingError(
                "Expected "
                f"documents_written={EXPECTED_DOCUMENT_COUNT}, "
                f"actual documents_written={documents_written}"
            )

        visibility_attempts = wait_for_all_documents_visible(
            document_store,
            corpus.documents,
            settings.pinecone_dimension,
        )
        print(f"Visibility attempts: {visibility_attempts}")
        print("Visibility PASS")

        session_documents = SessionDocumentService(document_store)
        summarization_pipeline = create_summarization_pipeline(settings)
        summarization_service = SummarizationService(
            session_documents=session_documents,
            summarization_pipeline=summarization_pipeline,
        )

        request = build_summarization_request(corpus)
        summary_result = summarization_service.summarize(request)

        print(f"Source count: {len(summary_result.source_document_ids)}")
        validate_summary_acceptance(summary_result, corpus)
        print("Target source set PASS")
        print("Foreign source exclusion PASS")
        print(f"Summary length: {len(summary_result.text)}")
        print("Headings PASS")
        print("Grounded target facts PASS")
        print("Forbidden foreign facts absent")
        print("AI recommendation PASS")
        print("Internal architecture terms absent")
        print("Summarization PASS")
    except Exception as exc:
        main_error = exc
        print(f"Smoke FAIL: {type(exc).__name__}: {exc}")
    finally:
        if document_store is not None and corpus is not None:
            try:
                document_store.delete_documents(list(corpus.expected_document_ids))
                cleanup_attempts = wait_for_all_documents_absent(
                    document_store,
                    corpus.documents,
                )
                print(f"Cleanup attempts: {cleanup_attempts}")
                print("Cleanup PASS")
            except Exception as exc:
                cleanup_error = exc
                print(f"Cleanup FAIL: {type(exc).__name__}: {exc}")
                _print_manual_cleanup_hint(
                    index_name=settings.pinecone_index_name,
                    namespace=settings.pinecone_namespace,
                    document_ids=corpus.expected_document_ids,
                )

    if cleanup_error is not None and main_error is not None:
        print(
            "Both smoke operation and cleanup failed: "
            f"{type(main_error).__name__}; {type(cleanup_error).__name__}"
        )
        return 1
    if main_error is not None:
        return 1
    if cleanup_error is not None:
        return 1

    print("Smoke PASS")
    return 0


def main() -> None:
    raise SystemExit(run_smoke_test())


if __name__ == "__main__":
    main()
