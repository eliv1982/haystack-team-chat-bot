"""Live smoke test for filtered retrieval with chat/session isolation."""

from __future__ import annotations

import math
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
from models import ChatMessage, RetrievalRequest  # noqa: E402
from pinecone_preflight import PineconeIndexInfo, validate_existing_pinecone_index  # noqa: E402
from pipelines import create_indexing_pipeline, create_query_pipeline  # noqa: E402
from retrieval_service import RetrievalService  # noqa: E402

TARGET_CHAT_ID = -1_002_003_004_001
OTHER_CHAT_ID = -1_002_003_004_999
RETRIEVAL_QUERY = (
    "On what day and at what time is the Project Aurora launch meeting scheduled?"
)

VISIBILITY_TIMEOUT_SECONDS = 90.0
CLEANUP_TIMEOUT_SECONDS = 90.0
POLL_INTERVAL_SECONDS = 2.0

TARGET_FACT_TEXT = "Project Aurora launch meeting is scheduled for Tuesday at 16:00 UTC."
TARGET_SUPPORTING_MARTA_TEXT = "Marta will prepare the release checklist for Project Aurora."
TARGET_SUPPORTING_EMAIL_TEXT = "The team selected email as the backup communication channel."
OTHER_SESSION_FRIDAY_TEXT = "Project Aurora launch meeting is scheduled for Friday at 09:00 UTC."
OTHER_SESSION_DANIEL_TEXT = "Daniel will prepare the release checklist for this session."
OTHER_CHAT_CANCELLED_TEXT = (
    "Project Aurora launch meeting was cancelled and has no scheduled time."
)


class SmokeTestError(Exception):
    """Base class for retrieval smoke test failures."""


class SmokeIndexingError(SmokeTestError):
    """Raised when indexing did not write the expected number of documents."""


class SmokeVerificationError(SmokeTestError):
    """Raised when retrieved document data is invalid."""


class SmokeVisibilityTimeoutError(SmokeTestError):
    """Raised when expected documents did not become visible in time."""


class SmokeProbeError(SmokeTestError):
    """Raised when a retrieval probe does not meet acceptance criteria."""


class SmokeIsolationError(SmokeTestError):
    """Raised when retrieval results leak across chat or session boundaries."""


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
    target_fact_id: str
    friday_distractor_id: str
    cancelled_distractor_id: str
    other_session_document_ids: frozenset[str]
    other_chat_document_ids: frozenset[str]


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
    """Create the six-message smoke corpus for one run."""
    resolved_run_id = run_id or str(uuid.uuid4())
    resolved_base_message_id = base_message_id or (int(uuid.uuid4().int % 1_000_000_000) + 1)
    base_time = sent_at or datetime(2026, 3, 10, 12, 0, tzinfo=timezone.utc)

    target_session_id = f"retrieval-smoke-target-{resolved_run_id}"
    other_session_id = f"retrieval-smoke-other-{resolved_run_id}"

    message_specs: list[tuple[int, int, str, str, int, str]] = [
        (TARGET_CHAT_ID, resolved_base_message_id, target_session_id, "Ava Planner", 910_001, TARGET_FACT_TEXT),
        (
            TARGET_CHAT_ID,
            resolved_base_message_id + 1,
            target_session_id,
            "Marta Ops",
            910_002,
            TARGET_SUPPORTING_MARTA_TEXT,
        ),
        (
            TARGET_CHAT_ID,
            resolved_base_message_id + 2,
            target_session_id,
            "Liam Comms",
            910_003,
            TARGET_SUPPORTING_EMAIL_TEXT,
        ),
        (
            TARGET_CHAT_ID,
            resolved_base_message_id + 3,
            other_session_id,
            "Nora Alt",
            910_004,
            OTHER_SESSION_FRIDAY_TEXT,
        ),
        (
            TARGET_CHAT_ID,
            resolved_base_message_id + 4,
            other_session_id,
            "Daniel Alt",
            910_005,
            OTHER_SESSION_DANIEL_TEXT,
        ),
        (
            OTHER_CHAT_ID,
            resolved_base_message_id + 5,
            target_session_id,
            "Evan Remote",
            910_006,
            OTHER_CHAT_CANCELLED_TEXT,
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
    other_session_document_ids = frozenset({documents[3].id, documents[4].id})
    other_chat_document_ids = frozenset({documents[5].id})

    return SmokeCorpus(
        run_id=resolved_run_id,
        target_session_id=target_session_id,
        other_session_id=other_session_id,
        messages=tuple(messages),
        documents=documents,
        expected_document_ids=expected_document_ids,
        target_fact_id=documents[0].id,
        friday_distractor_id=documents[3].id,
        cancelled_distractor_id=documents[5].id,
        other_session_document_ids=other_session_document_ids,
        other_chat_document_ids=other_chat_document_ids,
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


def validate_probe_a(
    results: Sequence[Document],
    corpus: SmokeCorpus,
    *,
    top_k: int,
) -> None:
    """Validate target-session retrieval probe."""
    if not results:
        raise SmokeProbeError("Probe A returned empty result")
    if len(results) > top_k:
        raise SmokeProbeError("Probe A returned more documents than top_k")

    result_ids = {document.id for document in results}
    for document in results:
        if document.meta.get("chat_id") != str(TARGET_CHAT_ID):
            raise SmokeIsolationError("Probe A document has foreign chat_id")
        if document.meta.get("session_id") != corpus.target_session_id:
            raise SmokeIsolationError("Probe A document has foreign session_id")
        if document.meta.get("source") != "telegram":
            raise SmokeIsolationError("Probe A document has invalid source")

    if corpus.target_fact_id not in result_ids:
        raise SmokeProbeError("Probe A missing target fact ID")
    if results[0].id != corpus.target_fact_id:
        raise SmokeProbeError("Probe A expected target fact at rank 0")

    rank_zero_content = results[0].content or ""
    if "Tuesday" not in rank_zero_content or "16:00 UTC" not in rank_zero_content:
        raise SmokeProbeError("Probe A rank-0 content missing expected schedule")

    foreign_ids = corpus.other_session_document_ids | corpus.other_chat_document_ids
    leaked = result_ids & foreign_ids
    if leaked:
        raise SmokeIsolationError("Probe A included foreign document IDs")

    score = results[0].score
    if score is None or isinstance(score, bool) or not isinstance(score, (int, float)):
        raise SmokeProbeError("Probe A rank-0 score is not numeric")
    if not math.isfinite(float(score)):
        raise SmokeProbeError("Probe A rank-0 score is not finite")


def validate_probe_b(
    results: Sequence[Document],
    corpus: SmokeCorpus,
    *,
    top_k: int,
) -> None:
    """Validate other-session retrieval probe."""
    if not results:
        raise SmokeProbeError("Probe B returned empty result")
    if len(results) > top_k:
        raise SmokeProbeError("Probe B returned more documents than top_k")

    result_ids = {document.id for document in results}
    for document in results:
        if document.meta.get("chat_id") != str(TARGET_CHAT_ID):
            raise SmokeIsolationError("Probe B document has foreign chat_id")
        if document.meta.get("session_id") != corpus.other_session_id:
            raise SmokeIsolationError("Probe B document has foreign session_id")
        if document.meta.get("source") != "telegram":
            raise SmokeIsolationError("Probe B document has invalid source")

    if corpus.friday_distractor_id not in result_ids:
        raise SmokeProbeError("Probe B missing Friday distractor ID")
    if results[0].id != corpus.friday_distractor_id:
        raise SmokeProbeError("Probe B expected Friday distractor at rank 0")

    rank_zero_content = results[0].content or ""
    if "Friday" not in rank_zero_content or "09:00 UTC" not in rank_zero_content:
        raise SmokeProbeError("Probe B rank-0 content missing expected schedule")

    if corpus.target_fact_id in result_ids:
        raise SmokeIsolationError("Probe B leaked target Tuesday document")
    if corpus.cancelled_distractor_id in result_ids:
        raise SmokeIsolationError("Probe B leaked other-chat document")

    score = results[0].score
    if score is None or isinstance(score, bool) or not isinstance(score, (int, float)):
        raise SmokeProbeError("Probe B rank-0 score is not numeric")
    if not math.isfinite(float(score)):
        raise SmokeProbeError("Probe B rank-0 score is not finite")


def validate_probe_c(
    results: Sequence[Document],
    corpus: SmokeCorpus,
    *,
    top_k: int,
) -> None:
    """Validate other-chat retrieval probe."""
    if not results:
        raise SmokeProbeError("Probe C returned empty result")
    if len(results) > top_k:
        raise SmokeProbeError("Probe C returned more documents than top_k")

    result_ids = {document.id for document in results}
    for document in results:
        if document.meta.get("chat_id") != str(OTHER_CHAT_ID):
            raise SmokeIsolationError("Probe C document has foreign chat_id")
        if document.meta.get("session_id") != corpus.target_session_id:
            raise SmokeIsolationError("Probe C document has foreign session_id")
        if document.meta.get("source") != "telegram":
            raise SmokeIsolationError("Probe C document has invalid source")

    if corpus.cancelled_distractor_id not in result_ids:
        raise SmokeProbeError("Probe C missing cancelled distractor ID")
    if results[0].id != corpus.cancelled_distractor_id:
        raise SmokeProbeError("Probe C expected cancelled distractor at rank 0")

    rank_zero_content = results[0].content or ""
    lowered = rank_zero_content.lower()
    if "cancelled" not in lowered or "no scheduled time" not in lowered:
        raise SmokeProbeError("Probe C rank-0 content missing expected cancellation")

    if corpus.target_fact_id in result_ids:
        raise SmokeIsolationError("Probe C leaked target Tuesday document")
    if corpus.friday_distractor_id in result_ids:
        raise SmokeIsolationError("Probe C leaked Friday document")

    score = results[0].score
    if score is None or isinstance(score, bool) or not isinstance(score, (int, float)):
        raise SmokeProbeError("Probe C rank-0 score is not numeric")
    if not math.isfinite(float(score)):
        raise SmokeProbeError("Probe C rank-0 score is not finite")


def build_retrieval_requests(corpus: SmokeCorpus) -> tuple[RetrievalRequest, RetrievalRequest, RetrievalRequest]:
    """Build the three retrieval probe requests for a smoke corpus."""
    return (
        RetrievalRequest(
            query=RETRIEVAL_QUERY,
            chat_id=TARGET_CHAT_ID,
            session_id=corpus.target_session_id,
        ),
        RetrievalRequest(
            query=RETRIEVAL_QUERY,
            chat_id=TARGET_CHAT_ID,
            session_id=corpus.other_session_id,
        ),
        RetrievalRequest(
            query=RETRIEVAL_QUERY,
            chat_id=OTHER_CHAT_ID,
            session_id=corpus.target_session_id,
        ),
    )


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


def _print_probe_summary(label: str, results: Sequence[Document]) -> None:
    print(f"{label} result count: {len(results)}")
    if results:
        print(f"{label} top-1 ID: {results[0].id}")
        print(f"{label} top-1 score: {results[0].score}")
    print(f"{label} PASS")


def run_smoke_test() -> int:
    """Execute the live retrieval smoke test and return a process exit code."""
    settings = load_settings()
    document_store = None
    corpus: SmokeCorpus | None = None
    main_error: Exception | None = None
    cleanup_error: Exception | None = None
    visibility_attempts = 0
    cleanup_attempts = 0
    documents_written: int | None = None
    probe_a_results: tuple[Document, ...] = ()
    probe_b_results: tuple[Document, ...] = ()
    probe_c_results: tuple[Document, ...] = ()

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
        for document_id in corpus.expected_document_ids:
            print(f"Document ID: {document_id}")

        documents_written = indexing_service.index_messages(list(corpus.messages))
        print(f"Documents written: {documents_written}")
        if documents_written != 6:
            raise SmokeIndexingError(
                f"Expected documents_written=6, actual documents_written={documents_written}"
            )

        visibility_attempts = wait_for_all_documents_visible(
            document_store,
            corpus.documents,
            settings.pinecone_dimension,
        )
        print(f"Visibility attempts: {visibility_attempts}")
        print("Visibility PASS")

        query_pipeline = create_query_pipeline(settings, document_store)
        retrieval_service = RetrievalService(query_pipeline, top_k=settings.retrieval_top_k)
        probe_a_request, probe_b_request, probe_c_request = build_retrieval_requests(corpus)

        probe_a_results = retrieval_service.retrieve(probe_a_request)
        validate_probe_a(probe_a_results, corpus, top_k=settings.retrieval_top_k)
        _print_probe_summary("Probe A", probe_a_results)

        probe_b_results = retrieval_service.retrieve(probe_b_request)
        validate_probe_b(probe_b_results, corpus, top_k=settings.retrieval_top_k)
        _print_probe_summary("Probe B", probe_b_results)

        probe_c_results = retrieval_service.retrieve(probe_c_request)
        validate_probe_c(probe_c_results, corpus, top_k=settings.retrieval_top_k)
        _print_probe_summary("Probe C", probe_c_results)

        print("Isolation PASS")
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
