"""Tests for ListeningSession and InMemorySessionStore."""

from __future__ import annotations

import threading
from datetime import datetime, timezone

import pytest

from session_store import (
    InMemorySessionStore,
    InvalidListeningSessionError,
    InvalidSessionStoreInputError,
    ListeningSession,
    NoActiveSessionError,
    SessionAlreadyActiveError,
)


def _started_at() -> datetime:
    return datetime(2024, 1, 15, 12, 0, tzinfo=timezone.utc)


def test_listening_session_valid_model() -> None:
    session = ListeningSession(
        chat_id=-1001234567890,
        session_id="telegram-session-abc",
        started_at=_started_at(),
        started_by_user_id=7,
        started_by_name="Alice",
        message_count=3,
    )
    assert session.chat_id == -1001234567890
    assert session.session_id == "telegram-session-abc"
    assert session.started_by_user_id == 7
    assert session.started_by_name == "Alice"
    assert session.message_count == 3


def test_listening_session_allows_negative_chat_id() -> None:
    session = ListeningSession(
        chat_id=-100,
        session_id="session-1",
        started_at=_started_at(),
        started_by_user_id=1,
        started_by_name="Alice",
    )
    assert session.chat_id == -100


@pytest.mark.parametrize("chat_id", [True, "100"])
def test_listening_session_rejects_invalid_chat_id(chat_id: object) -> None:
    with pytest.raises(InvalidListeningSessionError, match="chat_id"):
        ListeningSession(
            chat_id=chat_id,  # type: ignore[arg-type]
            session_id="session-1",
            started_at=_started_at(),
            started_by_user_id=1,
            started_by_name="Alice",
        )


@pytest.mark.parametrize("session_id", ["", "   "])
def test_listening_session_rejects_empty_session_id(session_id: str) -> None:
    with pytest.raises(InvalidListeningSessionError, match="session_id"):
        ListeningSession(
            chat_id=-100,
            session_id=session_id,
            started_at=_started_at(),
            started_by_user_id=1,
            started_by_name="Alice",
        )


def test_listening_session_trims_session_id() -> None:
    session = ListeningSession(
        chat_id=-100,
        session_id="  session-1  ",
        started_at=_started_at(),
        started_by_user_id=1,
        started_by_name="Alice",
    )
    assert session.session_id == "session-1"


def test_listening_session_rejects_naive_started_at() -> None:
    with pytest.raises(InvalidListeningSessionError, match="started_at"):
        ListeningSession(
            chat_id=-100,
            session_id="session-1",
            started_at=datetime(2024, 1, 15, 12, 0),
            started_by_user_id=1,
            started_by_name="Alice",
        )


@pytest.mark.parametrize("user_id", [0, -1, True])
def test_listening_session_rejects_invalid_user_id(user_id: object) -> None:
    with pytest.raises(InvalidListeningSessionError, match="started_by_user_id"):
        ListeningSession(
            chat_id=-100,
            session_id="session-1",
            started_at=_started_at(),
            started_by_user_id=user_id,  # type: ignore[arg-type]
            started_by_name="Alice",
        )


@pytest.mark.parametrize("name", ["", "   "])
def test_listening_session_rejects_empty_started_by_name(name: str) -> None:
    with pytest.raises(InvalidListeningSessionError, match="started_by_name"):
        ListeningSession(
            chat_id=-100,
            session_id="session-1",
            started_at=_started_at(),
            started_by_user_id=1,
            started_by_name=name,
        )


def test_listening_session_trims_started_by_name() -> None:
    session = ListeningSession(
        chat_id=-100,
        session_id="session-1",
        started_at=_started_at(),
        started_by_user_id=1,
        started_by_name="  Alice  ",
    )
    assert session.started_by_name == "Alice"


@pytest.mark.parametrize("message_count", [-1, True])
def test_listening_session_rejects_invalid_message_count(message_count: object) -> None:
    with pytest.raises(InvalidListeningSessionError, match="message_count"):
        ListeningSession(
            chat_id=-100,
            session_id="session-1",
            started_at=_started_at(),
            started_by_user_id=1,
            started_by_name="Alice",
            message_count=message_count,  # type: ignore[arg-type]
        )


def test_start_session_uses_injected_factory() -> None:
    calls = {"count": 0}

    def factory() -> str:
        calls["count"] += 1
        return "deterministic-id"

    store = InMemorySessionStore(session_id_factory=factory)
    session = store.start_session(
        chat_id=-100,
        started_at=_started_at(),
        started_by_user_id=7,
        started_by_name="Alice",
    )
    assert session.session_id == "deterministic-id"
    assert session.message_count == 0
    assert calls["count"] == 1


def test_start_session_default_id_has_expected_prefix() -> None:
    store = InMemorySessionStore()
    session = store.start_session(
        chat_id=-100,
        started_at=_started_at(),
        started_by_user_id=7,
        started_by_name="Alice",
    )
    assert session.session_id.startswith("telegram-session-")


def test_start_session_rejects_invalid_factory_result() -> None:
    store = InMemorySessionStore(session_id_factory=lambda: "   ")
    with pytest.raises(InvalidSessionStoreInputError, match="session_id_factory"):
        store.start_session(
            chat_id=-100,
            started_at=_started_at(),
            started_by_user_id=7,
            started_by_name="Alice",
        )


def test_start_session_rejects_duplicate_active_session() -> None:
    store = InMemorySessionStore(session_id_factory=lambda: "session-1")
    store.start_session(
        chat_id=-100,
        started_at=_started_at(),
        started_by_user_id=7,
        started_by_name="Alice",
    )
    with pytest.raises(SessionAlreadyActiveError):
        store.start_session(
            chat_id=-100,
            started_at=_started_at(),
            started_by_user_id=8,
            started_by_name="Bob",
        )


def test_start_session_is_independent_per_chat() -> None:
    store = InMemorySessionStore(session_id_factory=lambda: "shared-id")
    first = store.start_session(
        chat_id=-100,
        started_at=_started_at(),
        started_by_user_id=7,
        started_by_name="Alice",
    )
    second = store.start_session(
        chat_id=-200,
        started_at=_started_at(),
        started_by_user_id=8,
        started_by_name="Bob",
    )
    assert first.chat_id == -100
    assert second.chat_id == -200


def test_get_active_session_returns_session() -> None:
    store = InMemorySessionStore(session_id_factory=lambda: "session-1")
    created = store.start_session(
        chat_id=-100,
        started_at=_started_at(),
        started_by_user_id=7,
        started_by_name="Alice",
    )
    active = store.get_active_session(-100)
    assert active == created


def test_get_active_session_unknown_chat_returns_none() -> None:
    store = InMemorySessionStore()
    assert store.get_active_session(-999) is None


def test_record_message_increments_from_zero_to_one() -> None:
    store = InMemorySessionStore(session_id_factory=lambda: "session-1")
    created = store.start_session(
        chat_id=-100,
        started_at=_started_at(),
        started_by_user_id=7,
        started_by_name="Alice",
    )
    updated = store.record_message(-100)
    assert created.message_count == 0
    assert updated.message_count == 1


def test_record_message_increments_sequentially() -> None:
    store = InMemorySessionStore(session_id_factory=lambda: "session-1")
    store.start_session(
        chat_id=-100,
        started_at=_started_at(),
        started_by_user_id=7,
        started_by_name="Alice",
    )
    first = store.record_message(-100)
    second = store.record_message(-100)
    assert first.message_count == 1
    assert second.message_count == 2


def test_record_message_without_active_session_raises() -> None:
    store = InMemorySessionStore()
    with pytest.raises(NoActiveSessionError):
        store.record_message(-100)


def test_record_message_is_independent_per_chat() -> None:
    store = InMemorySessionStore(session_id_factory=lambda: "session-1")
    store.start_session(
        chat_id=-100,
        started_at=_started_at(),
        started_by_user_id=7,
        started_by_name="Alice",
    )
    store.start_session(
        chat_id=-200,
        started_at=_started_at(),
        started_by_user_id=8,
        started_by_name="Bob",
    )
    store.record_message(-100)
    store.record_message(-100)
    store.record_message(-200)
    assert store.get_active_session(-100).message_count == 2
    assert store.get_active_session(-200).message_count == 1


def test_stop_session_returns_final_snapshot() -> None:
    store = InMemorySessionStore(session_id_factory=lambda: "session-1")
    store.start_session(
        chat_id=-100,
        started_at=_started_at(),
        started_by_user_id=7,
        started_by_name="Alice",
    )
    store.record_message(-100)
    stopped = store.stop_session(-100)
    assert stopped.message_count == 1


def test_stop_session_removes_active_session() -> None:
    store = InMemorySessionStore(session_id_factory=lambda: "session-1")
    store.start_session(
        chat_id=-100,
        started_at=_started_at(),
        started_by_user_id=7,
        started_by_name="Alice",
    )
    store.stop_session(-100)
    assert store.get_active_session(-100) is None


def test_stop_session_does_not_affect_other_chat() -> None:
    store = InMemorySessionStore(session_id_factory=lambda: "session-1")
    store.start_session(
        chat_id=-100,
        started_at=_started_at(),
        started_by_user_id=7,
        started_by_name="Alice",
    )
    store.start_session(
        chat_id=-200,
        started_at=_started_at(),
        started_by_user_id=8,
        started_by_name="Bob",
    )
    store.stop_session(-100)
    assert store.get_active_session(-100) is None
    assert store.get_active_session(-200) is not None


@pytest.mark.parametrize(
    "method_name",
    ["get_active_session", "record_message", "stop_session", "get_latest_completed_session"],
)
@pytest.mark.parametrize("chat_id", [True, "-100"], ids=["bool", "string"])
def test_public_methods_reject_a_chat_id_that_is_not_an_integer(
    method_name: str, chat_id: object
) -> None:
    store = InMemorySessionStore()
    with pytest.raises(InvalidSessionStoreInputError, match="chat_id"):
        getattr(store, method_name)(chat_id)


def test_start_session_rejects_bool_chat_id() -> None:
    store = InMemorySessionStore()
    with pytest.raises(InvalidSessionStoreInputError, match="chat_id"):
        store.start_session(
            chat_id=True,  # type: ignore[arg-type]
            started_at=_started_at(),
            started_by_user_id=7,
            started_by_name="Alice",
        )


def test_concurrent_record_message_increments_count() -> None:
    store = InMemorySessionStore(session_id_factory=lambda: "session-1")
    store.start_session(
        chat_id=-100,
        started_at=_started_at(),
        started_by_user_id=7,
        started_by_name="Alice",
    )

    errors: list[BaseException] = []
    thread_count = 25
    barrier = threading.Barrier(thread_count)

    def worker() -> None:
        try:
            barrier.wait()
            store.record_message(-100)
        except BaseException as exc:
            errors.append(exc)

    threads = [threading.Thread(target=worker) for _ in range(thread_count)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert not errors
    active = store.get_active_session(-100)
    assert active is not None
    assert active.message_count == thread_count


def test_stop_session_saves_latest_completed_snapshot() -> None:
    store = InMemorySessionStore(session_id_factory=lambda: "session-1")
    store.start_session(
        chat_id=-100,
        started_at=_started_at(),
        started_by_user_id=7,
        started_by_name="Alice",
    )
    store.record_message(-100)
    stopped = store.stop_session(-100)
    latest = store.get_latest_completed_session(-100)
    assert latest is stopped
    assert latest.message_count == 1


def test_get_latest_completed_session_unknown_chat_returns_none() -> None:
    store = InMemorySessionStore()
    assert store.get_latest_completed_session(-100) is None


def test_new_completed_replaces_previous_only_in_same_chat() -> None:
    store = InMemorySessionStore(session_id_factory=lambda: "session-1")
    store.start_session(
        chat_id=-100,
        started_at=_started_at(),
        started_by_user_id=7,
        started_by_name="Alice",
    )
    store.record_message(-100)
    first = store.stop_session(-100)

    store.start_session(
        chat_id=-100,
        started_at=_started_at(),
        started_by_user_id=7,
        started_by_name="Alice",
    )
    store.record_message(-100)
    store.record_message(-100)
    second = store.stop_session(-100)

    latest = store.get_latest_completed_session(-100)
    assert latest is second
    assert latest.message_count == 2
    assert latest is not first


def test_active_and_latest_completed_can_coexist() -> None:
    counter = {"value": 0}

    def session_id_factory() -> str:
        counter["value"] += 1
        return f"session-{counter['value']}"

    store = InMemorySessionStore(session_id_factory=session_id_factory)
    store.start_session(
        chat_id=-100,
        started_at=_started_at(),
        started_by_user_id=7,
        started_by_name="Alice",
    )
    completed = store.stop_session(-100)

    store.start_session(
        chat_id=-100,
        started_at=_started_at(),
        started_by_user_id=8,
        started_by_name="Bob",
    )
    active = store.get_active_session(-100)

    assert active is not None
    assert store.get_latest_completed_session(-100) is completed
    assert active.session_id != completed.session_id


def test_latest_completed_isolated_between_chats() -> None:
    store = InMemorySessionStore(session_id_factory=lambda: "session-1")
    store.start_session(
        chat_id=-100,
        started_at=_started_at(),
        started_by_user_id=7,
        started_by_name="Alice",
    )
    store.stop_session(-100)

    store.start_session(
        chat_id=-200,
        started_at=_started_at(),
        started_by_user_id=8,
        started_by_name="Bob",
    )
    store.record_message(-200)
    stopped_b = store.stop_session(-200)

    assert store.get_latest_completed_session(-100).message_count == 0
    assert store.get_latest_completed_session(-200) is stopped_b


def test_repeated_stop_does_not_change_latest_completed() -> None:
    store = InMemorySessionStore(session_id_factory=lambda: "session-1")
    store.start_session(
        chat_id=-100,
        started_at=_started_at(),
        started_by_user_id=7,
        started_by_name="Alice",
    )
    completed = store.stop_session(-100)
    with pytest.raises(NoActiveSessionError):
        store.stop_session(-100)
    assert store.get_latest_completed_session(-100) is completed
