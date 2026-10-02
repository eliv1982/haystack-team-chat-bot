"""Complete, chronologically ordered loading of one chat session's documents."""

from __future__ import annotations

from datetime import datetime
from typing import Final

from haystack import Document
from haystack.document_stores.types import DocumentStore

from retrieval_filters import build_session_filter
from retrieval_service import (
    RetrievalInvariantError,
    RetrievalServiceError,
    validate_scoped_document,
)

# PineconeDocumentStore.filter_documents() answers every metadata filter with a
# similarity query capped at 1,000 results and gives no signal when the result is
# truncated beyond a log warning. A result of exactly this size therefore cannot be
# told apart from a truncated one, so it is never accepted: at most
# SESSION_DOCUMENT_LIMIT - 1 documents can be loaded completely, and a session
# registered with that many messages or more is refused without querying the store.
SESSION_DOCUMENT_LIMIT: Final[int] = 1_000


class SessionDocumentsError(Exception):
    """Raised when a session's documents cannot be loaded reliably."""


class SessionTooLargeError(SessionDocumentsError):
    """Raised when a session may hold more documents than can be loaded completely."""

    def __init__(self, limit: int) -> None:
        self.max_messages = limit - 1
        super().__init__(
            f"session has at least {limit} documents; "
            f"only sessions of up to {self.max_messages} can be loaded completely"
        )


class SessionIncompleteError(SessionDocumentsError):
    """Raised when fewer documents are visible than the session registered.

    Typically the vector index has not yet made the newest messages searchable, so
    asking again shortly afterwards may succeed.
    """

    def __init__(self, *, expected: int, fetched: int) -> None:
        self.expected = expected
        self.fetched = fetched
        super().__init__(f"session registered {expected} messages but {fetched} are visible")


class SessionInconsistentError(RetrievalInvariantError):
    """Raised when more documents exist for a session than it registered.

    This cannot be explained by index lag, so it is treated as a data-consistency
    error and is neither retried nor summarized.
    """

    def __init__(self, *, expected: int, fetched: int) -> None:
        self.expected = expected
        self.fetched = fetched
        super().__init__(f"session registered {expected} messages but {fetched} documents exist")


class SessionDocumentService:
    """Loads every document of one chat session, oldest message first."""

    def __init__(
        self,
        document_store: DocumentStore,
        *,
        limit: int = SESSION_DOCUMENT_LIMIT,
    ) -> None:
        if isinstance(limit, bool) or not isinstance(limit, int) or limit <= 0:
            raise SessionDocumentsError("limit must be a positive integer")
        self._document_store = document_store
        self._limit = limit

    def fetch(
        self,
        *,
        chat_id: int,
        session_id: str,
        expected_count: int,
    ) -> tuple[Document, ...]:
        """Return all documents of the session in chronological order, or raise.

        Documents are selected by the hard ``chat_id`` + ``session_id`` metadata
        filter and every returned document is re-validated against that scope. The
        number of valid, unique documents must then equal ``expected_count`` (the
        number of messages the session registry counted), so the result is never a
        partial one:

        * ``SessionTooLargeError`` if ``expected_count`` cannot be loaded completely;
        * ``SessionIncompleteError`` if fewer documents are visible than expected;
        * ``SessionInconsistentError`` if more documents are visible than expected.
        """
        if isinstance(chat_id, bool) or not isinstance(chat_id, int):
            raise SessionDocumentsError("chat_id must be an integer")
        if not isinstance(session_id, str) or not session_id.strip():
            raise SessionDocumentsError("session_id must be a non-empty string")
        if isinstance(expected_count, bool) or not isinstance(expected_count, int):
            raise SessionDocumentsError("expected_count must be an integer")
        if expected_count < 0:
            raise SessionDocumentsError("expected_count must not be negative")

        if expected_count >= self._limit:
            raise SessionTooLargeError(self._limit)

        documents = self._document_store.filter_documents(
            filters=build_session_filter(chat_id, session_id)
        )
        if not isinstance(documents, list):
            raise RetrievalServiceError("document store must return a list of documents")

        seen_ids: set[str] = set()
        keyed: list[tuple[tuple[datetime, int, str], Document]] = []
        for document in documents:
            validate_scoped_document(
                document,
                chat_id=chat_id,
                session_id=session_id,
                seen_ids=seen_ids,
            )
            keyed.append((_chronological_key(document), document))

        # A result at the store's cap is necessarily larger than expected_count here,
        # so truncation is reported as an inconsistency and never as a complete set.
        if len(keyed) > expected_count:
            raise SessionInconsistentError(expected=expected_count, fetched=len(keyed))
        if len(keyed) < expected_count:
            raise SessionIncompleteError(expected=expected_count, fetched=len(keyed))

        keyed.sort(key=lambda item: item[0])
        return tuple(document for _, document in keyed)


def _chronological_key(document: Document) -> tuple[datetime, int, str]:
    """Order by send time, then Telegram message id, then document id."""
    return (
        _parse_sent_at(document.meta["sent_at"]),
        _parse_message_id(document.meta["message_id"]),
        document.id,
    )


def _parse_sent_at(value: object) -> datetime:
    if not isinstance(value, str):
        raise RetrievalInvariantError("retrieved document sent_at must be an ISO-8601 string")
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError as exc:
        raise RetrievalInvariantError(
            "retrieved document sent_at must be an ISO-8601 string"
        ) from exc
    if parsed.tzinfo is None or parsed.tzinfo.utcoffset(parsed) is None:
        raise RetrievalInvariantError("retrieved document sent_at must be timezone-aware")
    return parsed


def _parse_message_id(value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, str)):
        raise RetrievalInvariantError("retrieved document message_id must be an integer")
    try:
        return int(value)
    except ValueError as exc:
        raise RetrievalInvariantError("retrieved document message_id must be an integer") from exc
