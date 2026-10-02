"""Tests for the ChatMessage domain model."""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from models import (
    ChatMessage,
    InvalidChatMessageError,
    InvalidRetrievalRequestError,
    InvalidSummarizationRequestError,
    InvalidSummarizationResultError,
    RetrievalRequest,
    SummarizationRequest,
    SummarizationResult,
)


def test_valid_chat_message(sample_message: ChatMessage) -> None:
    assert sample_message.chat_id == -1001234567890
    assert sample_message.message_id == 42
    assert sample_message.user_id == 7
    assert sample_message.session_id == "chat:-1001234567890"
    assert sample_message.author_name == "Alice"
    assert sample_message.username == "alice"
    assert sample_message.text == "Hello, team!"
    assert sample_message.sent_at.tzinfo is not None


@pytest.mark.parametrize("field_name", ["chat_id", "message_id", "user_id"])
def test_bool_instead_of_integer_id_is_rejected(field_name: str) -> None:
    values = {
        "chat_id": -100,
        "message_id": 1,
        "user_id": 1,
        "session_id": "session-1",
        "author_name": "Alice",
        "username": None,
        "text": "Hello",
        "sent_at": datetime(2024, 1, 15, 12, 30, tzinfo=timezone.utc),
    }
    values[field_name] = True

    with pytest.raises(InvalidChatMessageError, match=field_name):
        ChatMessage(**values)


@pytest.mark.parametrize("message_id", [0, -1])
def test_zero_or_negative_message_id_is_rejected(message_id: int) -> None:
    with pytest.raises(InvalidChatMessageError, match="message_id"):
        ChatMessage(
            chat_id=-100,
            message_id=message_id,
            user_id=1,
            session_id="session-1",
            author_name="Alice",
            username=None,
            text="Hello",
            sent_at=datetime(2024, 1, 15, 12, 30, tzinfo=timezone.utc),
        )


@pytest.mark.parametrize("user_id", [0, -1])
def test_zero_or_negative_user_id_is_rejected(user_id: int) -> None:
    with pytest.raises(InvalidChatMessageError, match="user_id"):
        ChatMessage(
            chat_id=-100,
            message_id=1,
            user_id=user_id,
            session_id="session-1",
            author_name="Alice",
            username=None,
            text="Hello",
            sent_at=datetime(2024, 1, 15, 12, 30, tzinfo=timezone.utc),
        )


@pytest.mark.parametrize("field_name", ["session_id", "author_name", "text"])
def test_empty_required_text_fields_are_rejected(field_name: str) -> None:
    values = {
        "chat_id": -100,
        "message_id": 1,
        "user_id": 1,
        "session_id": "session-1",
        "author_name": "Alice",
        "username": None,
        "text": "Hello",
        "sent_at": datetime(2024, 1, 15, 12, 30, tzinfo=timezone.utc),
    }
    values[field_name] = "   "

    with pytest.raises(InvalidChatMessageError, match=field_name):
        ChatMessage(**values)


def test_naive_datetime_is_rejected() -> None:
    with pytest.raises(InvalidChatMessageError, match="sent_at"):
        ChatMessage(
            chat_id=-100,
            message_id=1,
            user_id=1,
            session_id="session-1",
            author_name="Alice",
            username=None,
            text="Hello",
            sent_at=datetime(2024, 1, 15, 12, 30),
        )


def test_whitespace_username_becomes_none() -> None:
    message = ChatMessage(
        chat_id=-100,
        message_id=1,
        user_id=1,
        session_id="session-1",
        author_name="Alice",
        username="   ",
        text="Hello",
        sent_at=datetime(2024, 1, 15, 12, 30, tzinfo=timezone.utc),
    )

    assert message.username is None


def test_valid_retrieval_request() -> None:
    request = RetrievalRequest(
        query="What did Alice say?",
        chat_id=-1001234567890,
        session_id="chat:-1001234567890",
    )

    assert request.query == "What did Alice say?"
    assert request.chat_id == -1001234567890
    assert request.session_id == "chat:-1001234567890"


def test_retrieval_request_rejects_bool_chat_id() -> None:
    with pytest.raises(InvalidRetrievalRequestError, match="chat_id"):
        RetrievalRequest(query="hello", chat_id=True, session_id="session-1")


@pytest.mark.parametrize("chat_id", ["-100", 1.5, None])
def test_retrieval_request_rejects_non_int_chat_id(chat_id: object) -> None:
    with pytest.raises(InvalidRetrievalRequestError, match="chat_id"):
        RetrievalRequest(query="hello", chat_id=chat_id, session_id="session-1")  # type: ignore[arg-type]


@pytest.mark.parametrize("query", ["", "   "])
def test_retrieval_request_rejects_empty_query(query: str) -> None:
    with pytest.raises(InvalidRetrievalRequestError, match="query"):
        RetrievalRequest(query=query, chat_id=-100, session_id="session-1")


def test_retrieval_request_trims_query() -> None:
    request = RetrievalRequest(query="  hello team  ", chat_id=-100, session_id="session-1")

    assert request.query == "hello team"


@pytest.mark.parametrize("session_id", ["", "   "])
def test_retrieval_request_rejects_empty_session_id(session_id: str) -> None:
    with pytest.raises(InvalidRetrievalRequestError, match="session_id"):
        RetrievalRequest(query="hello", chat_id=-100, session_id=session_id)


def test_retrieval_request_trims_session_id() -> None:
    request = RetrievalRequest(query="hello", chat_id=-100, session_id="  session-1  ")

    assert request.session_id == "session-1"


def test_valid_summarization_request() -> None:
    request = SummarizationRequest(
        instruction="Подготовь краткое резюме обсуждения",
        chat_id=-1001234567890,
        session_id="chat:-1001234567890",
        expected_message_count=137,
    )

    assert request.instruction == "Подготовь краткое резюме обсуждения"
    assert request.chat_id == -1001234567890
    assert request.session_id == "chat:-1001234567890"
    assert request.expected_message_count == 137


def test_summarization_request_rejects_bool_chat_id() -> None:
    with pytest.raises(InvalidSummarizationRequestError, match="chat_id"):
        SummarizationRequest(
            instruction="Summarize",
            chat_id=True,
            session_id="session-1",
            expected_message_count=3,
        )


@pytest.mark.parametrize("chat_id", ["-100", 1.5, None])
def test_summarization_request_rejects_non_int_chat_id(chat_id: object) -> None:
    with pytest.raises(InvalidSummarizationRequestError, match="chat_id"):
        SummarizationRequest(
            instruction="Summarize",
            chat_id=chat_id,
            session_id="session-1",
            expected_message_count=3,
        )  # type: ignore[arg-type]


@pytest.mark.parametrize("instruction", ["", "   "])
def test_summarization_request_rejects_empty_instruction(instruction: str) -> None:
    with pytest.raises(InvalidSummarizationRequestError, match="instruction"):
        SummarizationRequest(
            instruction=instruction,
            chat_id=-100,
            session_id="session-1",
            expected_message_count=3,
        )


def test_summarization_request_trims_instruction() -> None:
    request = SummarizationRequest(
        instruction="  Summarize discussion  ",
        chat_id=-100,
        session_id="session-1",
        expected_message_count=3,
    )

    assert request.instruction == "Summarize discussion"


@pytest.mark.parametrize("session_id", ["", "   "])
def test_summarization_request_rejects_empty_session_id(session_id: str) -> None:
    with pytest.raises(InvalidSummarizationRequestError, match="session_id"):
        SummarizationRequest(
            instruction="Summarize",
            chat_id=-100,
            session_id=session_id,
            expected_message_count=3,
        )


def test_summarization_request_trims_session_id() -> None:
    request = SummarizationRequest(
        instruction="Summarize",
        chat_id=-100,
        session_id="  session-1  ",
        expected_message_count=3,
    )

    assert request.session_id == "session-1"


def test_valid_summarization_result() -> None:
    result = SummarizationResult(
        text="Тема\nКлючевые позиции",
        source_document_ids=("doc-1", "doc-2"),
    )

    assert result.text == "Тема\nКлючевые позиции"
    assert result.source_document_ids == ("doc-1", "doc-2")


def test_summarization_result_trims_text() -> None:
    result = SummarizationResult(text="  Summary text  ", source_document_ids=("doc-1",))

    assert result.text == "Summary text"


@pytest.mark.parametrize("text", ["", "   "])
def test_summarization_result_rejects_empty_text(text: str) -> None:
    with pytest.raises(InvalidSummarizationResultError, match="text"):
        SummarizationResult(text=text, source_document_ids=("doc-1",))


def test_summarization_result_rejects_empty_source_ids() -> None:
    with pytest.raises(InvalidSummarizationResultError, match="source_document_ids"):
        SummarizationResult(text="Summary", source_document_ids=())


@pytest.mark.parametrize("document_id", ["", "   "])
def test_summarization_result_rejects_empty_source_id(document_id: str) -> None:
    with pytest.raises(InvalidSummarizationResultError, match="source document id"):
        SummarizationResult(text="Summary", source_document_ids=(document_id,))


def test_summarization_result_rejects_duplicate_source_ids() -> None:
    with pytest.raises(InvalidSummarizationResultError, match="duplicates"):
        SummarizationResult(text="Summary", source_document_ids=("doc-1", "doc-1"))


def test_summarization_result_preserves_source_id_order() -> None:
    result = SummarizationResult(text="Summary", source_document_ids=("doc-3", "doc-1", "doc-2"))

    assert result.source_document_ids == ("doc-3", "doc-1", "doc-2")


def test_summarization_request_allows_zero_expected_messages() -> None:
    request = SummarizationRequest(
        instruction="Summarize",
        chat_id=-100,
        session_id="session-1",
        expected_message_count=0,
    )

    assert request.expected_message_count == 0


def test_summarization_request_requires_expected_message_count() -> None:
    # There is deliberately no default: a request without it could not be gated.
    with pytest.raises(TypeError):
        SummarizationRequest(  # type: ignore[call-arg]
            instruction="Summarize",
            chat_id=-100,
            session_id="session-1",
        )


@pytest.mark.parametrize("expected", [-1, True, False, 1.5, "3", None])
def test_summarization_request_rejects_invalid_expected_message_count(expected: object) -> None:
    with pytest.raises(InvalidSummarizationRequestError, match="expected_message_count"):
        SummarizationRequest(
            instruction="Summarize",
            chat_id=-100,
            session_id="session-1",
            expected_message_count=expected,  # type: ignore[arg-type]
        )
