"""Tests for retrieval metadata filter construction."""

from __future__ import annotations

from models import RetrievalRequest
from retrieval_filters import build_chat_session_filter


def _request(*, chat_id: int = -1001234567890, session_id: str = "chat:-1001234567890") -> RetrievalRequest:
    return RetrievalRequest(
        query="What happened?",
        chat_id=chat_id,
        session_id=session_id,
    )


def test_build_chat_session_filter_has_and_operator() -> None:
    filters = build_chat_session_filter(_request())

    assert filters == {
        "operator": "AND",
        "conditions": [
            {
                "field": "meta.chat_id",
                "operator": "==",
                "value": "-1001234567890",
            },
            {
                "field": "meta.session_id",
                "operator": "==",
                "value": "chat:-1001234567890",
            },
        ],
    }


def test_build_chat_session_filter_uses_meta_fields() -> None:
    filters = build_chat_session_filter(_request())

    fields = {condition["field"] for condition in filters["conditions"]}  # type: ignore[index]
    assert fields == {"meta.chat_id", "meta.session_id"}


def test_build_chat_session_filter_converts_chat_id_to_string() -> None:
    filters = build_chat_session_filter(_request(chat_id=42))

    chat_condition = filters["conditions"][0]  # type: ignore[index]
    assert chat_condition["value"] == "42"
    assert isinstance(chat_condition["value"], str)


def test_build_chat_session_filter_preserves_negative_chat_id() -> None:
    filters = build_chat_session_filter(_request(chat_id=-999_000_001))

    chat_condition = filters["conditions"][0]  # type: ignore[index]
    assert chat_condition["value"] == "-999000001"


def test_build_chat_session_filter_does_not_include_user_id() -> None:
    filters = build_chat_session_filter(_request())
    serialized = repr(filters)

    assert "user_id" not in serialized


def test_build_chat_session_filter_does_not_mutate_request() -> None:
    request = _request()
    original = (request.query, request.chat_id, request.session_id)

    build_chat_session_filter(request)

    assert (request.query, request.chat_id, request.session_id) == original


def test_build_chat_session_filter_returns_independent_dicts() -> None:
    first = build_chat_session_filter(_request())
    second = build_chat_session_filter(_request())

    assert first == second
    assert first is not second
    assert first["conditions"] is not second["conditions"]


def test_build_chat_session_filter_is_not_affected_by_mutation_of_returned_dict() -> None:
    request = _request()
    first = build_chat_session_filter(request)
    first["operator"] = "OR"
    first["conditions"][0]["value"] = "mutated"

    second = build_chat_session_filter(request)

    assert second["operator"] == "AND"
    assert second["conditions"][0]["value"] == "-1001234567890"
