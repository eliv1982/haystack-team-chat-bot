"""Live smoke test for the indexing pipeline against an existing Pinecone index."""

from __future__ import annotations

import sys
import time
import uuid
from collections.abc import Callable
from datetime import datetime, timezone
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
from models import ChatMessage  # noqa: E402
from pinecone_preflight import PineconeIndexInfo, validate_existing_pinecone_index  # noqa: E402
from pipelines import create_indexing_pipeline  # noqa: E402

SMOKE_CHAT_ID = -999_000_001
SMOKE_USER_ID = 900_000_001
SMOKE_AUTHOR_NAME = "Smoke Test Bot"
SMOKE_TEXT_MARKER = "haystack-team-chat-bot smoke-test marker"

VISIBILITY_TIMEOUT_SECONDS = 60.0
POLL_INTERVAL_SECONDS = 2.0


class SmokeTestError(Exception):
    """Base class for smoke test failures."""


class SmokeIndexingError(SmokeTestError):
    """Raised when indexing did not write the expected number of documents."""


class SmokeVerificationError(SmokeTestError):
    """Raised when retrieved document data is invalid."""


class SmokeVisibilityTimeoutError(SmokeTestError):
    """Raised when the expected document did not become visible in time."""


class SmokeCleanupError(SmokeTestError):
    """Raised when cleanup could not be confirmed."""


def build_smoke_message() -> tuple[ChatMessage, str]:
    """Create a unique smoke-test chat message and session identifier."""
    session_id = f"smoke-{uuid.uuid4()}"
    message_id = int(datetime.now(timezone.utc).timestamp() * 1000)
    message = ChatMessage(
        chat_id=SMOKE_CHAT_ID,
        message_id=message_id,
        user_id=SMOKE_USER_ID,
        session_id=session_id,
        author_name=SMOKE_AUTHOR_NAME,
        username=None,
        text=SMOKE_TEXT_MARKER,
        sent_at=datetime.now(timezone.utc),
    )
    return message, session_id


def build_session_filter(session_id: str) -> dict[str, Any]:
    """Build a Haystack metadata filter for the smoke session."""
    return {"field": "session_id", "operator": "==", "value": session_id}


def find_document_by_id(documents: list[Document], document_id: str) -> Document | None:
    """Return the document with the exact ID, if present."""
    for document in documents:
        if document.id == document_id:
            return document
    return None


def verify_retrieved_document(
    retrieved: Document,
    expected: Document,
    expected_dimension: int,
) -> None:
    """Validate that the retrieved document matches the smoke expectations."""
    if retrieved.id != expected.id:
        raise SmokeVerificationError("Retrieved document ID does not match expected ID")
    if retrieved.content != expected.content:
        raise SmokeVerificationError("Retrieved document content does not match expected content")

    for key, expected_value in expected.meta.items():
        if retrieved.meta.get(key) != expected_value:
            raise SmokeVerificationError(f"Retrieved document metadata mismatch for field: {key}")

    embedding = retrieved.embedding
    if embedding is None:
        raise SmokeVerificationError("Retrieved document is missing an embedding")
    if len(embedding) != expected_dimension:
        raise SmokeVerificationError(
            f"Retrieved embedding dimension mismatch: expected {expected_dimension}, actual {len(embedding)}"
        )


def wait_for_document_visible(
    document_store: DocumentStore,
    expected_document: Document,
    session_id: str,
    expected_dimension: int,
    *,
    timeout_seconds: float = VISIBILITY_TIMEOUT_SECONDS,
    poll_interval_seconds: float = POLL_INTERVAL_SECONDS,
    sleep: Callable[[float], None] = time.sleep,
    monotonic: Callable[[], float] = time.monotonic,
) -> tuple[int, Document]:
    """Poll until the exact expected document becomes visible."""
    deadline = monotonic() + timeout_seconds
    attempts = 0
    filters = build_session_filter(session_id)

    while monotonic() < deadline:
        attempts += 1
        documents = document_store.filter_documents(filters=filters)
        retrieved = find_document_by_id(documents, expected_document.id)
        if retrieved is not None:
            verify_retrieved_document(retrieved, expected_document, expected_dimension)
            return attempts, retrieved
        sleep(poll_interval_seconds)

    raise SmokeVisibilityTimeoutError(
        f"Expected document did not become visible within {timeout_seconds:.0f} seconds"
    )


def wait_for_document_absent(
    document_store: DocumentStore,
    document_id: str,
    session_id: str,
    *,
    timeout_seconds: float = VISIBILITY_TIMEOUT_SECONDS,
    poll_interval_seconds: float = POLL_INTERVAL_SECONDS,
    sleep: Callable[[float], None] = time.sleep,
    monotonic: Callable[[], float] = time.monotonic,
) -> int:
    """Poll until the exact document ID is no longer visible."""
    deadline = monotonic() + timeout_seconds
    attempts = 0
    filters = build_session_filter(session_id)

    while monotonic() < deadline:
        attempts += 1
        documents = document_store.filter_documents(filters=filters)
        if find_document_by_id(documents, document_id) is None:
            return attempts
        sleep(poll_interval_seconds)

    raise SmokeCleanupError(
        f"Expected document remained visible after cleanup within {timeout_seconds:.0f} seconds"
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
    document_id: str,
) -> None:
    print("Manual cleanup required")
    print(f"Index: {index_name}")
    print(f"Namespace: {namespace}")
    print(f"Document ID: {document_id}")


def run_smoke_test() -> int:
    """Execute the live smoke test and return a process exit code."""
    settings = load_settings()
    document_store = None
    expected_document: Document | None = None
    session_id: str | None = None
    main_error: Exception | None = None
    cleanup_error: Exception | None = None
    visibility_attempts = 0
    cleanup_attempts = 0
    documents_written: int | None = None
    embedding_dimension: int | None = None

    try:
        index_info = validate_existing_pinecone_index(settings)
        _print_preflight_result(index_info, settings)

        document_store = create_pinecone_document_store(settings)
        pipeline = create_indexing_pipeline(settings, document_store)
        service = IndexingService(pipeline)

        message, session_id = build_smoke_message()
        expected_document = chat_message_to_document(message)
        print(f"Session ID: {session_id}")
        print(f"Document ID: {expected_document.id}")

        documents_written = service.index_messages([message])
        print(f"Documents written: {documents_written}")
        if documents_written != 1:
            raise SmokeIndexingError(
                f"Expected documents_written=1, actual documents_written={documents_written}"
            )

        visibility_attempts, retrieved = wait_for_document_visible(
            document_store,
            expected_document,
            session_id,
            settings.pinecone_dimension,
        )
        embedding_dimension = len(retrieved.embedding or [])
        print(f"Visibility attempts: {visibility_attempts}")
        print(f"Embedding dimension: {embedding_dimension}")
        print("Verification PASS")
    except Exception as exc:
        main_error = exc
        print(f"Smoke FAIL: {type(exc).__name__}: {exc}")
    finally:
        if document_store is not None and expected_document is not None and session_id is not None:
            try:
                document_store.delete_documents([expected_document.id])
                cleanup_attempts = wait_for_document_absent(
                    document_store,
                    expected_document.id,
                    session_id,
                )
                print(f"Cleanup attempts: {cleanup_attempts}")
                print("Cleanup PASS")
            except Exception as exc:
                cleanup_error = exc
                print(f"Cleanup FAIL: {type(exc).__name__}: {exc}")
                _print_manual_cleanup_hint(
                    index_name=settings.pinecone_index_name,
                    namespace=settings.pinecone_namespace,
                    document_id=expected_document.id,
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
