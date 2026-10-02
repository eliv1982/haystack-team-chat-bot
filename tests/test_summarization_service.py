"""Tests for summarization service behavior and validation."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from haystack import Document, Pipeline
from haystack.dataclasses.chat_message import ChatMessage as HaystackChatMessage
from haystack.dataclasses.chat_message import ToolCall

from models import SummarizationRequest
from session_documents import (
    SessionIncompleteError,
    SessionInconsistentError,
    SessionTooLargeError,
)
from summarization_service import (
    NoSummarizationContextError,
    SummarizationResultError,
    SummarizationService,
)


def _summarization_request(
    *,
    chat_id: int = -1001234567890,
    session_id: str = "chat:-1001234567890",
    instruction: str = "Подготовь резюме",
    expected_message_count: int = 1,
) -> SummarizationRequest:
    return SummarizationRequest(
        instruction=instruction,
        chat_id=chat_id,
        session_id=session_id,
        expected_message_count=expected_message_count,
    )


def _metadata(
    *,
    chat_id: int = -1001234567890,
    session_id: str = "chat:-1001234567890",
    **overrides: object,
) -> dict[str, object]:
    metadata: dict[str, object] = {
        "source": "telegram",
        "schema_version": 1,
        "chat_id": str(chat_id),
        "message_id": "42",
        "user_id": "7",
        "session_id": session_id,
        "author_name": "Alice",
        "sent_at": "2024-01-15T12:30:00+00:00",
    }
    metadata.update(overrides)
    return metadata


def _document(
    *,
    document_id: str = "doc-1",
    content: str = "Hello from Alice",
    score: float = 0.91,
    chat_id: int = -1001234567890,
    session_id: str = "chat:-1001234567890",
    metadata_overrides: dict[str, object] | None = None,
) -> Document:
    metadata = _metadata(chat_id=chat_id, session_id=session_id)
    if metadata_overrides is not None:
        metadata.update(metadata_overrides)
    document = Document(
        id=document_id,
        content=content,
        meta=metadata,
        score=score,
    )
    if document_id == "":
        object.__setattr__(document, "id", "")
    return document


def _assistant_reply(text: str = "Тема\nКлючевые позиции") -> HaystackChatMessage:
    return HaystackChatMessage.from_assistant(text)


def _pipeline_result(reply: HaystackChatMessage | None = None) -> dict[str, object]:
    return {"llm": {"replies": [reply or _assistant_reply()]}}


@pytest.fixture
def session_documents() -> MagicMock:
    return MagicMock()


@pytest.fixture
def pipeline() -> MagicMock:
    return MagicMock(spec=Pipeline)


@pytest.fixture
def service(session_documents: MagicMock, pipeline: MagicMock) -> SummarizationService:
    return SummarizationService(
        session_documents=session_documents,
        summarization_pipeline=pipeline,
    )


def test_summarize_loads_whole_session_once_by_chat_and_session(
    service: SummarizationService,
    session_documents: MagicMock,
    pipeline: MagicMock,
) -> None:
    request = _summarization_request()
    session_documents.fetch.return_value = (_document(),)
    pipeline.run.return_value = _pipeline_result()

    service.summarize(request)

    session_documents.fetch.assert_called_once_with(
        chat_id=request.chat_id,
        session_id=request.session_id,
        expected_count=request.expected_message_count,
    )


def test_summarize_does_not_mutate_request(
    service: SummarizationService,
    session_documents: MagicMock,
    pipeline: MagicMock,
) -> None:
    request = _summarization_request()
    original = (
        request.instruction,
        request.chat_id,
        request.session_id,
        request.expected_message_count,
    )
    session_documents.fetch.return_value = (_document(),)
    pipeline.run.return_value = _pipeline_result()

    service.summarize(request)

    assert (
        request.instruction,
        request.chat_id,
        request.session_id,
        request.expected_message_count,
    ) == original


def test_summarize_raises_when_session_has_no_documents(
    service: SummarizationService,
    session_documents: MagicMock,
    pipeline: MagicMock,
) -> None:
    session_documents.fetch.return_value = ()

    with pytest.raises(NoSummarizationContextError):
        service.summarize(_summarization_request())

    pipeline.run.assert_not_called()


def test_summarize_calls_pipeline_once_with_documents_and_instruction(
    service: SummarizationService,
    session_documents: MagicMock,
    pipeline: MagicMock,
) -> None:
    request = _summarization_request(instruction="Summarize the release plan")
    documents = (_document(document_id="doc-1"), _document(document_id="doc-2", content="Second"))
    session_documents.fetch.return_value = documents
    pipeline.run.return_value = _pipeline_result()

    service.summarize(request)

    pipeline.run.assert_called_once()
    run_args, run_kwargs = pipeline.run.call_args
    assert run_args[0] == {
        "prompt_builder": {
            "documents": list(documents),
            "instruction": "Summarize the release plan",
        }
    }
    assert run_kwargs["include_outputs_from"] == {"llm"}


def test_summarize_returns_valid_result_with_source_ids_in_document_order(
    service: SummarizationService,
    session_documents: MagicMock,
    pipeline: MagicMock,
) -> None:
    documents = (
        _document(document_id="doc-1"),
        _document(document_id="doc-2", content="Second"),
    )
    session_documents.fetch.return_value = documents
    pipeline.run.return_value = _pipeline_result(_assistant_reply("  Summary text  "))

    result = service.summarize(_summarization_request())

    assert result.text == "Summary text"
    assert result.source_document_ids == ("doc-1", "doc-2")
    assert isinstance(result.source_document_ids, tuple)


@pytest.mark.parametrize(
    "documents",
    [
        [object()],
        [_document(document_id="")],
        [_document(), _document(document_id="dup-1"), _document(document_id="dup-1", content="x")],
        [_document(metadata_overrides={"chat_id": "other"})],
        [_document(metadata_overrides={"session_id": "other"})],
        [_document(metadata_overrides={"source": "email"})],
    ],
)
def test_summarize_rejects_invalid_documents_before_pipeline_run(
    service: SummarizationService,
    session_documents: MagicMock,
    pipeline: MagicMock,
    documents: list[object],
) -> None:
    session_documents.fetch.return_value = tuple(documents)

    with pytest.raises(SummarizationResultError):
        service.summarize(_summarization_request())

    pipeline.run.assert_not_called()


@pytest.mark.parametrize(
    "pipeline_result",
    [
        "bad",
        {},
        {"llm": "bad"},
        {"llm": {}},
        {"llm": {"replies": "bad"}},
        {"llm": {"replies": []}},
        {"llm": {"replies": [_assistant_reply(), _assistant_reply()]}},
        {"llm": {"replies": [HaystackChatMessage.from_user("not assistant")]}},
        {"llm": {"replies": [_assistant_reply("   ")]}},
        {
            "llm": {
                "replies": [
                    HaystackChatMessage.from_assistant(
                        text=None,
                        tool_calls=[
                            ToolCall(
                                id="call-1",
                                tool_name="search",
                                arguments={"query": "x"},
                            )
                        ],
                    )
                ]
            }
        },
    ],
)
def test_summarize_rejects_malformed_pipeline_results(
    service: SummarizationService,
    session_documents: MagicMock,
    pipeline: MagicMock,
    pipeline_result: object,
) -> None:
    session_documents.fetch.return_value = (_document(),)
    pipeline.run.return_value = pipeline_result

    with pytest.raises(SummarizationResultError):
        service.summarize(_summarization_request())


def test_summarize_does_not_swallow_document_loading_errors(
    service: SummarizationService,
    session_documents: MagicMock,
    pipeline: MagicMock,
) -> None:
    session_documents.fetch.side_effect = RuntimeError("loading failed")

    with pytest.raises(RuntimeError, match="loading failed"):
        service.summarize(_summarization_request())

    pipeline.run.assert_not_called()


def test_summarize_does_not_swallow_pipeline_errors(
    service: SummarizationService,
    session_documents: MagicMock,
    pipeline: MagicMock,
) -> None:
    session_documents.fetch.return_value = (_document(),)
    pipeline.run.side_effect = RuntimeError("pipeline failed")

    with pytest.raises(RuntimeError, match="pipeline failed"):
        service.summarize(_summarization_request())

    pipeline.run.assert_called_once()


def test_summarize_has_no_direct_openai_or_pinecone_fallback(
    service: SummarizationService,
    session_documents: MagicMock,
    pipeline: MagicMock,
) -> None:
    session_documents.fetch.return_value = (_document(),)
    pipeline.run.side_effect = RuntimeError("pipeline failed")

    with pytest.raises(RuntimeError):
        service.summarize(_summarization_request())

    assert not hasattr(service, "_document_store")
    assert not hasattr(service, "_openai_client")


@pytest.mark.parametrize(
    "gate_error",
    [
        SessionIncompleteError(expected=137, fetched=136),
        SessionInconsistentError(expected=137, fetched=138),
        SessionTooLargeError(1_000),
    ],
    ids=["incomplete", "inconsistent", "too-large"],
)
def test_summarize_never_calls_the_model_when_the_completeness_gate_fails(
    service: SummarizationService,
    session_documents: MagicMock,
    pipeline: MagicMock,
    gate_error: Exception,
) -> None:
    session_documents.fetch.side_effect = gate_error

    with pytest.raises(type(gate_error)):
        service.summarize(_summarization_request(expected_message_count=137))

    pipeline.run.assert_not_called()


def test_summarize_forwards_the_registered_message_count_to_the_gate(
    service: SummarizationService,
    session_documents: MagicMock,
    pipeline: MagicMock,
) -> None:
    session_documents.fetch.return_value = (_document(),)
    pipeline.run.return_value = _pipeline_result()

    service.summarize(_summarization_request(expected_message_count=137))

    assert session_documents.fetch.call_args.kwargs["expected_count"] == 137
