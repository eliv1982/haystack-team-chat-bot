"""Tests for complete, chronologically ordered loading of one session's documents."""

from __future__ import annotations

import random
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock

import pytest
from haystack import Document
from haystack.document_stores.in_memory import InMemoryDocumentStore
from haystack.document_stores.types import DuplicatePolicy

from documents import chat_message_to_document
from models import ChatMessage
from retrieval_filters import build_session_filter
from retrieval_service import RetrievalInvariantError, RetrievalServiceError
from session_documents import (
    SESSION_DOCUMENT_LIMIT,
    SessionDocumentService,
    SessionDocumentsError,
    SessionIncompleteError,
    SessionInconsistentError,
    SessionTooLargeError,
)

CHAT_ID = -1001234567890
OTHER_CHAT_ID = -1009999999999
SESSION_ID = "session-1"
OTHER_SESSION_ID = "session-2"
BASE_TIME = datetime(2024, 1, 15, 12, 0, tzinfo=timezone.utc)


class _PineconeLikeStore(InMemoryDocumentStore):
    """In-memory store that, like PineconeDocumentStore, caps every filter query."""

    def filter_documents(self, filters: dict | None = None) -> list[Document]:
        documents = super().filter_documents(filters=filters)
        return documents[:SESSION_DOCUMENT_LIMIT]


def _message(
    index: int,
    *,
    chat_id: int = CHAT_ID,
    session_id: str = SESSION_ID,
    message_id: int | None = None,
    sent_at: datetime | None = None,
) -> ChatMessage:
    return ChatMessage(
        chat_id=chat_id,
        message_id=message_id if message_id is not None else 1_000 + index,
        user_id=7 + index % 3,
        session_id=session_id,
        author_name=f"Author {index % 3}",
        username=None,
        text=f"Message number {index}",
        sent_at=sent_at if sent_at is not None else BASE_TIME + timedelta(seconds=index),
    )


def _store_with(messages: list[ChatMessage], *, seed: int = 7) -> _PineconeLikeStore:
    """Fill a store, deliberately not in chronological order."""
    documents = [chat_message_to_document(message) for message in messages]
    random.Random(seed).shuffle(documents)
    store = _PineconeLikeStore()
    store.write_documents(documents, policy=DuplicatePolicy.OVERWRITE)
    return store


def _service(store: InMemoryDocumentStore, **kwargs: int) -> SessionDocumentService:
    return SessionDocumentService(store, **kwargs)


def _fetch(
    store: object,
    expected: int,
    *,
    chat_id: int = CHAT_ID,
    session_id: str = SESSION_ID,
    **service_kwargs: int,
) -> tuple[Document, ...]:
    return SessionDocumentService(store, **service_kwargs).fetch(  # type: ignore[arg-type]
        chat_id=chat_id,
        session_id=session_id,
        expected_count=expected,
    )


def _expected_ids(messages: list[ChatMessage]) -> list[str]:
    ordered = sorted(messages, key=lambda m: (m.sent_at, m.message_id))
    return [chat_message_to_document(message).id for message in ordered]


# --- completeness: the fetched set must equal the registered message count ----


def test_fetch_returns_every_message_when_the_count_matches_exactly() -> None:
    messages = [_message(index) for index in range(137)]
    store = _store_with(messages)

    documents = _fetch(store, expected=137)

    assert len(documents) == 137
    assert {document.id for document in documents} == {
        chat_message_to_document(message).id for message in messages
    }


def test_fetch_refuses_one_missing_document_of_137() -> None:
    store = _store_with([_message(index) for index in range(136)])

    with pytest.raises(SessionIncompleteError) as excinfo:
        _fetch(store, expected=137)

    assert excinfo.value.expected == 137
    assert excinfo.value.fetched == 136


def test_fetch_refuses_an_extra_document_beyond_the_registered_137_as_inconsistent() -> None:
    store = _store_with([_message(index) for index in range(138)])

    with pytest.raises(SessionInconsistentError) as excinfo:
        _fetch(store, expected=137)

    assert excinfo.value.expected == 137
    assert excinfo.value.fetched == 138
    assert isinstance(excinfo.value, RetrievalInvariantError)


def test_inconsistency_is_an_invariant_error_not_an_incomplete_one() -> None:
    # Extras must never be mistaken for index lag, which callers may retry.
    assert not issubclass(SessionInconsistentError, SessionIncompleteError)
    assert not issubclass(SessionIncompleteError, SessionInconsistentError)
    assert not issubclass(SessionIncompleteError, RetrievalInvariantError)


@pytest.mark.parametrize("visible", [0, 1, 5])
def test_fetch_refuses_any_shortfall_not_only_a_missing_one(visible: int) -> None:
    store = _store_with([_message(index) for index in range(visible)])

    with pytest.raises(SessionIncompleteError) as excinfo:
        _fetch(store, expected=6)

    assert (excinfo.value.expected, excinfo.value.fetched) == (6, visible)


def test_fetch_refuses_any_document_when_none_were_registered() -> None:
    store = _store_with([_message(0)])

    with pytest.raises(SessionInconsistentError):
        _fetch(store, expected=0)


def test_fetch_of_an_empty_session_that_registered_nothing_is_an_empty_tuple() -> None:
    store = _store_with([_message(index, session_id=OTHER_SESSION_ID) for index in range(3)])

    assert _fetch(store, expected=0, session_id="missing") == ()


def test_count_comparison_uses_only_documents_of_this_chat_and_session() -> None:
    session_messages = [_message(index) for index in range(6)]
    foreign = [
        _message(index, session_id=OTHER_SESSION_ID, message_id=5_000 + index) for index in range(4)
    ] + [_message(index, chat_id=OTHER_CHAT_ID, message_id=6_000 + index) for index in range(4)]
    store = _store_with(session_messages + foreign)

    assert len(_fetch(store, expected=6)) == 6
    # Foreign documents cannot make up for a missing one of this session ...
    with pytest.raises(SessionIncompleteError):
        _fetch(store, expected=10)
    # ... nor count as extras of it.
    with pytest.raises(SessionIncompleteError):
        _fetch(store, expected=7)


def test_count_gate_runs_after_scope_validation_so_a_foreign_document_cannot_balance_it() -> None:
    store = MagicMock()
    foreign = replace(
        chat_message_to_document(_message(1)),
        meta={**chat_message_to_document(_message(1)).meta, "chat_id": str(OTHER_CHAT_ID)},
    )
    # Two documents were registered; one of the two returned is out of scope.
    store.filter_documents.return_value = [chat_message_to_document(_message(0)), foreign]

    with pytest.raises(RetrievalInvariantError) as excinfo:
        _fetch(store, expected=2)

    assert not isinstance(excinfo.value, (SessionIncompleteError, SessionInconsistentError))


def test_duplicate_documents_are_not_counted_twice() -> None:
    store = MagicMock()
    document = chat_message_to_document(_message(0))
    store.filter_documents.return_value = [document, document]

    with pytest.raises(RetrievalServiceError):
        _fetch(store, expected=2)


# --- the 999 / 1000 boundary --------------------------------------------------


def test_fetch_is_complete_at_the_documented_maximum() -> None:
    maximum = SESSION_DOCUMENT_LIMIT - 1
    store = _store_with([_message(index) for index in range(maximum)])

    documents = _fetch(store, expected=maximum)

    assert len(documents) == maximum == 999


@pytest.mark.parametrize("registered", [SESSION_DOCUMENT_LIMIT, SESSION_DOCUMENT_LIMIT + 25])
def test_a_session_registered_at_or_above_the_limit_is_refused_without_querying_the_store(
    registered: int,
) -> None:
    store = MagicMock()

    with pytest.raises(SessionTooLargeError) as excinfo:
        _fetch(store, expected=registered)

    assert excinfo.value.max_messages == SESSION_DOCUMENT_LIMIT - 1
    assert str(SESSION_DOCUMENT_LIMIT - 1) in str(excinfo.value)
    store.filter_documents.assert_not_called()


def test_a_store_result_at_its_cap_is_never_accepted_as_complete() -> None:
    # The store cannot tell us whether 1000 results were truncated. For a session
    # registered with 999 messages that result is more than expected, so it fails.
    store = _store_with([_message(index) for index in range(SESSION_DOCUMENT_LIMIT + 25)])

    with pytest.raises(SessionInconsistentError) as excinfo:
        _fetch(store, expected=SESSION_DOCUMENT_LIMIT - 1)

    assert excinfo.value.fetched == SESSION_DOCUMENT_LIMIT


def test_exactly_the_store_limit_of_documents_for_a_session_of_999_is_inconsistent() -> None:
    store = _store_with([_message(index) for index in range(SESSION_DOCUMENT_LIMIT)])

    with pytest.raises(SessionInconsistentError):
        _fetch(store, expected=SESSION_DOCUMENT_LIMIT - 1)
    with pytest.raises(SessionTooLargeError):
        _fetch(store, expected=SESSION_DOCUMENT_LIMIT)


def test_a_configured_smaller_limit_moves_the_boundary_and_never_returns_a_partial_result() -> None:
    store = _store_with([_message(index) for index in range(10)])

    assert len(_fetch(store, expected=10, limit=11)) == 10
    with pytest.raises(SessionTooLargeError):
        _fetch(store, expected=10, limit=10)
    with pytest.raises(SessionInconsistentError):
        _fetch(store, expected=3, limit=4)


# --- chronological order -----------------------------------------------------


def test_fetch_orders_documents_chronologically_whatever_the_store_order() -> None:
    messages = [_message(index) for index in range(80)]
    expected = _expected_ids(messages)

    for seed in (1, 2, 3):
        documents = _fetch(_store_with(messages, seed=seed), expected=80)
        assert [document.id for document in documents] == expected


def test_fetch_orders_by_sent_at_not_by_message_id() -> None:
    # Message ids normally grow with time, but the canonical order is sent_at.
    late_low_id = _message(0, message_id=10, sent_at=BASE_TIME + timedelta(minutes=5))
    early_high_id = _message(1, message_id=99, sent_at=BASE_TIME)
    store = _store_with([late_low_id, early_high_id])

    documents = _fetch(store, expected=2)

    assert [document.meta["message_id"] for document in documents] == ["99", "10"]


def test_fetch_breaks_same_second_ties_by_numeric_message_id() -> None:
    # "9" < "10" numerically but "10" < "9" as strings.
    same_second = [
        _message(index, message_id=message_id, sent_at=BASE_TIME)
        for index, message_id in enumerate([10, 9, 100, 8])
    ]

    for seed in (1, 2, 3):
        documents = _fetch(_store_with(same_second, seed=seed), expected=4)
        assert [document.meta["message_id"] for document in documents] == ["8", "9", "10", "100"]


def test_fetch_compares_timestamps_across_utc_offsets() -> None:
    plus_three = timezone(timedelta(hours=3))
    earlier_instant_later_clock = _message(
        0, message_id=1, sent_at=datetime(2024, 1, 15, 14, 0, tzinfo=plus_three)
    )  # 11:00 UTC
    later_instant_earlier_clock = _message(
        1, message_id=2, sent_at=datetime(2024, 1, 15, 12, 0, tzinfo=timezone.utc)
    )  # 12:00 UTC
    store = _store_with([later_instant_earlier_clock, earlier_instant_later_clock])

    documents = _fetch(store, expected=2)

    assert [document.meta["message_id"] for document in documents] == ["1", "2"]


def test_fetch_is_deterministic_across_calls() -> None:
    store = _store_with([_message(index) for index in range(30)])
    service = _service(store)

    first = service.fetch(chat_id=CHAT_ID, session_id=SESSION_ID, expected_count=30)
    second = service.fetch(chat_id=CHAT_ID, session_id=SESSION_ID, expected_count=30)

    assert [d.id for d in first] == [d.id for d in second]


def test_fetch_returns_an_immutable_tuple_of_documents() -> None:
    store = _store_with([_message(0)])

    documents = _fetch(store, expected=1)

    assert isinstance(documents, tuple)
    assert all(isinstance(document, Document) for document in documents)


# --- hard chat/session isolation --------------------------------------------


def test_fetch_excludes_other_sessions_and_other_chats() -> None:
    target = [_message(index) for index in range(6)]
    other_session_same_chat = [
        _message(index, session_id=OTHER_SESSION_ID, message_id=2_000 + index) for index in range(4)
    ]
    other_chat_same_session_id = [
        _message(index, chat_id=OTHER_CHAT_ID, message_id=3_000 + index) for index in range(4)
    ]
    other_chat_other_session = [
        _message(
            index, chat_id=OTHER_CHAT_ID, session_id=OTHER_SESSION_ID, message_id=4_000 + index
        )
        for index in range(4)
    ]
    store = _store_with(
        target + other_session_same_chat + other_chat_same_session_id + other_chat_other_session
    )

    documents = _fetch(store, expected=6)

    assert [document.id for document in documents] == _expected_ids(target)
    for document in documents:
        assert document.meta["chat_id"] == str(CHAT_ID)
        assert document.meta["session_id"] == SESSION_ID


def test_fetch_sends_the_hard_chat_and_session_filter_to_the_store() -> None:
    store = MagicMock()
    store.filter_documents.return_value = []

    _fetch(store, expected=0)

    store.filter_documents.assert_called_once_with(
        filters=build_session_filter(CHAT_ID, SESSION_ID)
    )
    sent = store.filter_documents.call_args.kwargs["filters"]
    assert sent == {
        "operator": "AND",
        "conditions": [
            {"field": "meta.chat_id", "operator": "==", "value": str(CHAT_ID)},
            {"field": "meta.session_id", "operator": "==", "value": SESSION_ID},
        ],
    }


def _stored(index: int = 0, **meta_overrides: object) -> Document:
    document = chat_message_to_document(_message(index))
    meta = {**document.meta, **meta_overrides}
    return replace(document, meta={k: v for k, v in meta.items() if v is not ...})


@pytest.mark.parametrize(
    "bad_document",
    [
        pytest.param(_stored(1, chat_id=str(OTHER_CHAT_ID)), id="foreign-chat"),
        pytest.param(_stored(1, session_id=OTHER_SESSION_ID), id="foreign-session"),
        pytest.param(_stored(1, source="email"), id="foreign-source"),
        pytest.param(_stored(1, chat_id=...), id="missing-chat"),
        pytest.param(_stored(1, session_id=...), id="missing-session"),
        pytest.param(_stored(1, sent_at=...), id="missing-sent-at"),
        pytest.param(_stored(1, user_id=None), id="null-user-id"),
        pytest.param(_stored(1, author_name=None), id="null-author"),
        pytest.param(_stored(1, message_id=...), id="missing-message-id"),
    ],
)
def test_fetch_fails_closed_if_the_store_returns_an_out_of_scope_or_incomplete_document(
    bad_document: Document,
) -> None:
    store = MagicMock()
    store.filter_documents.return_value = [chat_message_to_document(_message(0)), bad_document]

    # The count matches (2 expected, 2 returned): only validation can refuse it.
    with pytest.raises(RetrievalInvariantError):
        _fetch(store, expected=2)


def test_fetch_fails_closed_on_a_foreign_document_even_for_an_oversized_result() -> None:
    store = MagicMock()
    store.filter_documents.return_value = [
        chat_message_to_document(_message(index)) for index in range(5)
    ] + [_stored(9, chat_id=str(OTHER_CHAT_ID))]

    with pytest.raises(RetrievalInvariantError):
        _fetch(store, expected=2, limit=3)


@pytest.mark.parametrize(
    ("documents", "error"),
    [
        ([object()], RetrievalServiceError),
        ([Document(id="x", content=None, meta={})], RetrievalServiceError),
        (
            [chat_message_to_document(_message(0)), chat_message_to_document(_message(0))],
            RetrievalServiceError,
        ),
        ("not-a-list", RetrievalServiceError),
    ],
    ids=["not-a-document", "no-content", "duplicate-ids", "not-a-list"],
)
def test_fetch_rejects_malformed_store_output(documents: object, error: type[Exception]) -> None:
    store = MagicMock()
    store.filter_documents.return_value = documents

    with pytest.raises(error):
        _fetch(store, expected=1)


@pytest.mark.parametrize(
    "meta_override",
    [
        {"sent_at": "yesterday"},
        {"sent_at": "2024-01-15T12:00:00"},  # naive
        {"sent_at": 1705320000},
        {"message_id": "abc"},
        {"message_id": 1.5},
        {"message_id": True},
    ],
    ids=["unparseable", "naive", "not-a-string", "non-numeric-id", "float-id", "bool-id"],
)
def test_fetch_fails_closed_on_unorderable_documents(meta_override: dict[str, object]) -> None:
    store = MagicMock()
    store.filter_documents.return_value = [_stored(0, **meta_override)]

    with pytest.raises(RetrievalInvariantError):
        _fetch(store, expected=1)


def test_store_errors_are_not_swallowed() -> None:
    store = MagicMock()
    store.filter_documents.side_effect = RuntimeError("store unavailable")

    with pytest.raises(RuntimeError, match="store unavailable"):
        _fetch(store, expected=1)


# --- argument validation -----------------------------------------------------


@pytest.mark.parametrize("limit", [0, -1, True, 1.5, "10", None])
def test_constructor_rejects_invalid_limits(limit: object) -> None:
    with pytest.raises(SessionDocumentsError):
        SessionDocumentService(MagicMock(), limit=limit)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("chat_id", "session_id"),
    [(True, SESSION_ID), ("-100", SESSION_ID), (None, SESSION_ID), (CHAT_ID, ""), (CHAT_ID, "  ")],
)
def test_fetch_rejects_invalid_arguments_without_querying_the_store(
    chat_id: object, session_id: object
) -> None:
    store = MagicMock()

    with pytest.raises(SessionDocumentsError):
        _fetch(store, expected=1, chat_id=chat_id, session_id=session_id)  # type: ignore[arg-type]

    store.filter_documents.assert_not_called()


@pytest.mark.parametrize("expected", [-1, True, False, 1.5, "3", None])
def test_fetch_rejects_an_invalid_expected_count_without_querying_the_store(
    expected: object,
) -> None:
    store = MagicMock()

    with pytest.raises(SessionDocumentsError):
        _fetch(store, expected=expected)  # type: ignore[arg-type]

    store.filter_documents.assert_not_called()


def test_fetch_requires_an_expected_count() -> None:
    # No default exists, so no caller can load a session without the completeness gate.
    with pytest.raises(TypeError):
        SessionDocumentService(MagicMock()).fetch(  # type: ignore[call-arg]
            chat_id=CHAT_ID, session_id=SESSION_ID
        )
