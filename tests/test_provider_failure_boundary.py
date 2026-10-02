"""Where an expected provider failure ends and a programming defect begins.

Everything runs for real except the two remote clients: the production pipelines, the
real ``PineconeDocumentStore``, Haystack's own exception wrapping, the services, the
handlers and TeleBot's dispatch. Only the OpenAI client and Pinecone's index client are
fakes that raise on demand, so each test sees the exception chain the installed
libraries really produce:

* an OpenAI or Pinecone request failure inside a pipeline arrives as
  ``PipelineRuntimeError`` whose ``__cause__`` is the SDK error; the service translates
  it into its own ``...ProviderError`` and the handler answers the chat;
* a ``TypeError``, ``AssertionError`` or malformed-call error inside the same component
  arrives as the same ``PipelineRuntimeError`` but is never translated, so it reaches
  TeleBot instead of being reported to the chat as an outage;
* Pinecone's SDK errors reach ``SessionDocumentService`` unwrapped, and only its
  request failures are translated.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from datetime import datetime, timezone

import httpx
import openai
import pytest
import telebot
from haystack import Document
from haystack.core.errors import PipelineRuntimeError
from pinecone import (
    ApiError,
    ForbiddenError,
    NotFoundError,
    PineconeConnectionError,
    PineconeTimeoutError,
    PineconeValueError,
    RateLimitError,
    ResponseParsingError,
    ServiceError,
    UnauthorizedError,
)
from pinecone.exceptions import PineconeApiTypeError

from config import Settings
from documents import chat_message_to_document
from fakes import (
    BOT_TOKEN,
    GROUP_CHAT_ID,
    FakeClock,
    FakeOpenAIClient,
    FakePineconeIndex,
    pinecone_document_store,
    telegram_message,
)
from indexing_service import IndexingProviderError, IndexingService, IndexingServiceError
from models import ChatMessage, SummarizationRequest
from pipelines import create_indexing_pipeline, create_summarization_pipeline
from session_documents import (
    SessionDocumentProviderError,
    SessionDocumentService,
    SessionDocumentsError,
)
from session_store import InMemorySessionStore
from summarization_service import (
    SummarizationProviderError,
    SummarizationService,
    SummarizationServiceError,
)
from telegram_application import TelegramApplicationService
from telegram_handlers import (
    _RECORD_FAILURE_NOTICE_INTERVAL_SECONDS,
    _RECORD_INTERNAL_ERROR_REPLY,
    _SUMMARY_INTERNAL_ERROR_REPLY,
    register_telegram_handlers,
)
from telegram_summary_application import TelegramSummaryApplicationService

SESSION_ID = "session-1"
BASE_TIME = datetime(2024, 1, 15, 12, 0, tzinfo=timezone.utc)
# Written into every injected error. It must never surface in a translated error or in
# the handlers' log: only exception types may.
PROVIDER_TEXT = "PROVIDER-ECHOED-PRIVATE-TEXT-4711"
REQUEST = httpx.Request("POST", "https://api.example.com/v1/endpoint")

ErrorFactory = Callable[[], Exception]


def _openai_status_error(error_type: type[openai.APIStatusError], status: int) -> ErrorFactory:
    return lambda: error_type(
        PROVIDER_TEXT, response=httpx.Response(status, request=REQUEST), body=None
    )


# Failed requests to OpenAI: the provider answered with an error, or was not reachable.
OPENAI_FAILURES = [
    pytest.param(lambda: openai.APIConnectionError(request=REQUEST), id="APIConnectionError"),
    pytest.param(lambda: openai.APITimeoutError(request=REQUEST), id="APITimeoutError"),
    pytest.param(_openai_status_error(openai.RateLimitError, 429), id="RateLimitError"),
    pytest.param(_openai_status_error(openai.InternalServerError, 500), id="InternalServerError"),
    pytest.param(_openai_status_error(openai.AuthenticationError, 401), id="AuthenticationError"),
    pytest.param(_openai_status_error(openai.BadRequestError, 400), id="BadRequestError"),
]

# Failed requests to Pinecone.
PINECONE_FAILURES = [
    pytest.param(lambda: ServiceError(PROVIDER_TEXT), id="ServiceError"),
    pytest.param(lambda: RateLimitError(PROVIDER_TEXT), id="RateLimitError"),
    pytest.param(lambda: UnauthorizedError(PROVIDER_TEXT), id="UnauthorizedError"),
    pytest.param(lambda: ForbiddenError(PROVIDER_TEXT), id="ForbiddenError"),
    pytest.param(lambda: NotFoundError(PROVIDER_TEXT), id="NotFoundError"),
    pytest.param(lambda: ApiError(PROVIDER_TEXT, 400), id="ApiError-other-status"),
    pytest.param(lambda: PineconeConnectionError(PROVIDER_TEXT), id="PineconeConnectionError"),
    pytest.param(lambda: PineconeTimeoutError(PROVIDER_TEXT), id="PineconeTimeoutError"),
]

# Pinecone's SDK rejecting a call this code made wrongly, or a response it cannot read.
# ``PineconeApiTypeError`` is the SDK's TypeError for a malformed argument.
PINECONE_CONTRACT_DEFECTS = [
    pytest.param(lambda: PineconeApiTypeError(PROVIDER_TEXT), id="PineconeApiTypeError"),
    pytest.param(lambda: PineconeValueError(PROVIDER_TEXT), id="PineconeValueError"),
    pytest.param(lambda: ResponseParsingError(PROVIDER_TEXT), id="ResponseParsingError"),
]

# Bugs inside a component, raised straight out of ``component.run``.
COMPONENT_DEFECTS = [
    pytest.param(lambda: TypeError(PROVIDER_TEXT), id="TypeError"),
    pytest.param(lambda: AssertionError(PROVIDER_TEXT), id="AssertionError"),
]


def _raises(error: Exception) -> Callable[..., object]:
    def run(*args: object, **kwargs: object) -> object:
        raise error

    return run


def _chat_message(index: int) -> ChatMessage:
    return ChatMessage(
        chat_id=GROUP_CHAT_ID,
        message_id=1_000 + index,
        user_id=7,
        session_id=SESSION_ID,
        author_name="Alice",
        username=None,
        text=f"msg-{index:04d}",
        sent_at=BASE_TIME.replace(second=index),
    )


def _documents(count: int) -> list[Document]:
    return [chat_message_to_document(_chat_message(index)) for index in range(count)]


def _summarization_request(count: int) -> SummarizationRequest:
    return SummarizationRequest(
        instruction="Подведи итог",
        chat_id=GROUP_CHAT_ID,
        session_id=SESSION_ID,
        expected_message_count=count,
    )


def _assert_translated_from_pipeline(
    translated: Exception, original: Exception, *, provider_error: type[Exception]
) -> None:
    """The translation keeps the full chain: provider error -> Haystack wrapper -> SDK error."""
    assert type(translated) is provider_error
    wrapper = translated.__cause__
    assert type(wrapper) is PipelineRuntimeError  # Haystack wraps exactly once
    assert wrapper.__cause__ is original
    assert PROVIDER_TEXT not in str(translated)  # a static message, not the provider's


@pytest.fixture(autouse=True)
def _skip_async_openai_clients(monkeypatch: pytest.MonkeyPatch) -> None:
    """Haystack also builds an async OpenAI client, which nothing here uses.

    Creating one loads the system's certificates (about 0.2 s), and these tests build a
    pipeline each.
    """
    for module in (
        "haystack.components.embedders.openai_document_embedder",
        "haystack.components.generators.chat.openai",
    ):
        monkeypatch.setattr(f"{module}.AsyncOpenAI", lambda **_: None)


@pytest.fixture
def index() -> FakePineconeIndex:
    return FakePineconeIndex()


@pytest.fixture
def pipeline_store(index: FakePineconeIndex):  # type: ignore[no-untyped-def]
    return pinecone_document_store(index)


# --- indexing: Haystack wraps everything, the service tells the two apart -----------


@pytest.mark.parametrize("make_error", OPENAI_FAILURES)
def test_openai_failure_while_indexing_is_translated_with_its_chain(
    settings: Settings,
    fake_openai: FakeOpenAIClient,
    pipeline_store,  # type: ignore[no-untyped-def]
    sample_message: ChatMessage,
    make_error: ErrorFactory,
) -> None:
    original = make_error()
    fake_openai.embedding_error = original

    with pytest.raises(IndexingProviderError) as raised:
        IndexingService(create_indexing_pipeline(settings, pipeline_store)).index_messages(
            [sample_message]
        )

    _assert_translated_from_pipeline(raised.value, original, provider_error=IndexingProviderError)
    assert isinstance(raised.value, IndexingServiceError)


@pytest.mark.parametrize("make_error", PINECONE_FAILURES)
def test_pinecone_write_failure_while_indexing_is_translated_with_its_chain(
    settings: Settings,
    fake_openai: FakeOpenAIClient,
    index: FakePineconeIndex,
    pipeline_store,  # type: ignore[no-untyped-def]
    sample_message: ChatMessage,
    make_error: ErrorFactory,
) -> None:
    original = make_error()
    index.upsert_error = original

    with pytest.raises(IndexingProviderError) as raised:
        IndexingService(create_indexing_pipeline(settings, pipeline_store)).index_messages(
            [sample_message]
        )

    _assert_translated_from_pipeline(raised.value, original, provider_error=IndexingProviderError)


@pytest.mark.parametrize("component", ["document_embedder", "writer"])
@pytest.mark.parametrize("make_defect", COMPONENT_DEFECTS)
def test_defect_inside_an_indexing_component_is_wrapped_by_haystack_but_not_translated(
    settings: Settings,
    fake_openai: FakeOpenAIClient,
    pipeline_store,  # type: ignore[no-untyped-def]
    sample_message: ChatMessage,
    monkeypatch: pytest.MonkeyPatch,
    component: str,
    make_defect: ErrorFactory,
) -> None:
    pipeline = create_indexing_pipeline(settings, pipeline_store)
    defect = make_defect()
    monkeypatch.setattr(pipeline.get_component(component), "run", _raises(defect))

    with pytest.raises(PipelineRuntimeError) as raised:
        IndexingService(pipeline).index_messages([sample_message])

    assert type(raised.value) is PipelineRuntimeError  # Haystack's wrapper, passed on untouched
    assert raised.value.__cause__ is defect
    assert not isinstance(raised.value, IndexingServiceError)


@pytest.mark.parametrize("make_defect", COMPONENT_DEFECTS)
def test_defect_inside_the_openai_embedding_call_is_not_translated(
    settings: Settings,
    fake_openai: FakeOpenAIClient,
    pipeline_store,  # type: ignore[no-untyped-def]
    sample_message: ChatMessage,
    make_defect: ErrorFactory,
) -> None:
    defect = make_defect()
    fake_openai.embedding_error = defect

    with pytest.raises(PipelineRuntimeError) as raised:
        IndexingService(create_indexing_pipeline(settings, pipeline_store)).index_messages(
            [sample_message]
        )

    assert type(raised.value) is PipelineRuntimeError
    assert raised.value.__cause__ is defect


@pytest.mark.parametrize("make_defect", PINECONE_CONTRACT_DEFECTS)
def test_pinecone_input_contract_defect_while_indexing_is_not_translated(
    settings: Settings,
    fake_openai: FakeOpenAIClient,
    index: FakePineconeIndex,
    pipeline_store,  # type: ignore[no-untyped-def]
    sample_message: ChatMessage,
    make_defect: ErrorFactory,
) -> None:
    defect = make_defect()
    index.upsert_error = defect

    with pytest.raises(PipelineRuntimeError) as raised:
        IndexingService(create_indexing_pipeline(settings, pipeline_store)).index_messages(
            [sample_message]
        )

    assert type(raised.value) is PipelineRuntimeError
    assert raised.value.__cause__ is defect


def test_a_defect_that_looks_like_an_outage_is_still_a_defect(
    settings: Settings,
    fake_openai: FakeOpenAIClient,
    pipeline_store,  # type: ignore[no-untyped-def]
    sample_message: ChatMessage,
) -> None:
    """Only the exception type decides; wording that sounds like an outage does not."""
    fake_openai.embedding_error = TypeError("503 Service Unavailable: connection reset by openai")

    with pytest.raises(PipelineRuntimeError) as raised:
        IndexingService(create_indexing_pipeline(settings, pipeline_store)).index_messages(
            [sample_message]
        )

    assert type(raised.value) is PipelineRuntimeError


# --- summarization pipeline ----------------------------------------------------------


@pytest.fixture
def summarization_service(
    settings: Settings,
    fake_openai: FakeOpenAIClient,
    index: FakePineconeIndex,
    pipeline_store,  # type: ignore[no-untyped-def]
) -> SummarizationService:
    index.serve(_documents(3))
    return SummarizationService(
        SessionDocumentService(pipeline_store), create_summarization_pipeline(settings)
    )


@pytest.mark.parametrize("make_error", OPENAI_FAILURES)
def test_openai_failure_while_summarizing_is_translated_with_its_chain(
    summarization_service: SummarizationService,
    fake_openai: FakeOpenAIClient,
    make_error: ErrorFactory,
) -> None:
    original = make_error()
    fake_openai.chat_error = original

    with pytest.raises(SummarizationProviderError) as raised:
        summarization_service.summarize(_summarization_request(3))

    _assert_translated_from_pipeline(
        raised.value, original, provider_error=SummarizationProviderError
    )
    assert isinstance(raised.value, SummarizationServiceError)


@pytest.mark.parametrize("component", ["prompt_builder", "llm"])
@pytest.mark.parametrize("make_defect", COMPONENT_DEFECTS)
def test_defect_inside_a_summarization_component_is_wrapped_by_haystack_but_not_translated(
    settings: Settings,
    fake_openai: FakeOpenAIClient,
    index: FakePineconeIndex,
    pipeline_store,  # type: ignore[no-untyped-def]
    monkeypatch: pytest.MonkeyPatch,
    component: str,
    make_defect: ErrorFactory,
) -> None:
    index.serve(_documents(3))
    pipeline = create_summarization_pipeline(settings)
    defect = make_defect()
    monkeypatch.setattr(pipeline.get_component(component), "run", _raises(defect))
    service = SummarizationService(SessionDocumentService(pipeline_store), pipeline)

    with pytest.raises(PipelineRuntimeError) as raised:
        service.summarize(_summarization_request(3))

    assert type(raised.value) is PipelineRuntimeError
    assert raised.value.__cause__ is defect
    assert not isinstance(raised.value, SummarizationServiceError)


def test_defect_inside_the_openai_chat_call_is_not_translated(
    summarization_service: SummarizationService,
    fake_openai: FakeOpenAIClient,
) -> None:
    defect = TypeError(PROVIDER_TEXT)
    fake_openai.chat_error = defect

    with pytest.raises(PipelineRuntimeError) as raised:
        summarization_service.summarize(_summarization_request(3))

    assert type(raised.value) is PipelineRuntimeError
    assert raised.value.__cause__ is defect


def test_summarization_recognizes_only_the_provider_its_pipeline_uses(
    settings: Settings,
    fake_openai: FakeOpenAIClient,
    index: FakePineconeIndex,
    pipeline_store,  # type: ignore[no-untyped-def]
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The summarization pipeline has no Pinecone component, so a Pinecone error from it is not an outage."""
    index.serve(_documents(3))
    pipeline = create_summarization_pipeline(settings)
    monkeypatch.setattr(pipeline.get_component("llm"), "run", _raises(ServiceError(PROVIDER_TEXT)))
    service = SummarizationService(SessionDocumentService(pipeline_store), pipeline)

    with pytest.raises(PipelineRuntimeError) as raised:
        service.summarize(_summarization_request(3))

    assert type(raised.value) is PipelineRuntimeError


# --- direct Pinecone read: no pipeline, SDK errors arrive unwrapped ------------------


@pytest.mark.parametrize("make_error", PINECONE_FAILURES)
def test_pinecone_read_failure_is_translated_and_chained_directly(
    index: FakePineconeIndex,
    pipeline_store,  # type: ignore[no-untyped-def]
    make_error: ErrorFactory,
) -> None:
    original = make_error()
    index.query_error = original

    with pytest.raises(SessionDocumentProviderError) as raised:
        SessionDocumentService(pipeline_store).fetch(
            chat_id=GROUP_CHAT_ID, session_id=SESSION_ID, expected_count=3
        )

    assert raised.value.__cause__ is original  # nothing sits between them
    assert PROVIDER_TEXT not in str(raised.value)
    assert isinstance(raised.value, SessionDocumentsError)


@pytest.mark.parametrize(
    "make_defect",
    [*PINECONE_CONTRACT_DEFECTS, pytest.param(lambda: TypeError(PROVIDER_TEXT), id="TypeError")],
)
def test_pinecone_input_contract_defect_on_read_propagates_unchanged(
    index: FakePineconeIndex,
    pipeline_store,  # type: ignore[no-untyped-def]
    make_defect: ErrorFactory,
) -> None:
    defect = make_defect()
    index.query_error = defect

    with pytest.raises(type(defect)) as raised:
        SessionDocumentService(pipeline_store).fetch(
            chat_id=GROUP_CHAT_ID, session_id=SESSION_ID, expected_count=3
        )

    assert raised.value is defect
    assert not isinstance(raised.value, SessionDocumentsError)


def test_a_successful_read_still_returns_the_documents(
    index: FakePineconeIndex,
    pipeline_store,  # type: ignore[no-untyped-def]
) -> None:
    index.serve(_documents(3))

    documents = SessionDocumentService(pipeline_store).fetch(
        chat_id=GROUP_CHAT_ID, session_id=SESSION_ID, expected_count=3
    )

    assert [document.content.rsplit(": ", 1)[1] for document in documents] == [
        "msg-0000",
        "msg-0001",
        "msg-0002",
    ]


# --- through the handlers: what reaches the chat and what reaches TeleBot -------------


class _Stack:
    """Real handlers, application services, services, pipelines and Pinecone store."""

    def __init__(
        self,
        bot: telebot.TeleBot,
        settings: Settings,
        store: object,
    ) -> None:
        self.bot = bot
        self.clock = FakeClock(1_000.0)
        self.session_store = InMemorySessionStore(session_id_factory=lambda: SESSION_ID)
        indexing_pipeline = create_indexing_pipeline(settings, store)  # type: ignore[arg-type]
        self.pipelines = {
            "indexing": indexing_pipeline,
            "summarization": create_summarization_pipeline(settings),
        }
        application = TelegramApplicationService(
            session_store=self.session_store,
            indexing_service=IndexingService(indexing_pipeline),
        )
        summary_application = TelegramSummaryApplicationService(
            session_store=self.session_store,
            summarization_service=SummarizationService(
                SessionDocumentService(store),  # type: ignore[arg-type]
                self.pipelines["summarization"],
            ),
            sleep=lambda _: None,
        )
        register_telegram_handlers(bot, application, summary_application, clock=self.clock)

    def send(self, text: str, *, message_id: int = 100) -> None:
        self.bot.process_new_messages([telegram_message(text, message_id=message_id)])

    def start_session(self, registered_messages: int = 0) -> None:
        self.send("/start_listening", message_id=1)
        for _ in range(registered_messages):
            self.session_store.record_message(GROUP_CHAT_ID)
        self.bot.reply_to.reset_mock()
        self.bot.send_message.reset_mock()

    def message_count(self) -> int:
        session = self.session_store.get_active_session(GROUP_CHAT_ID)
        assert session is not None
        return session.message_count


@pytest.fixture
def stack(
    telegram_bot: telebot.TeleBot,
    settings: Settings,
    fake_openai: FakeOpenAIClient,
    pipeline_store,  # type: ignore[no-untyped-def]
) -> _Stack:
    return _Stack(telegram_bot, settings, pipeline_store)


def _handler_log(caplog: pytest.LogCaptureFixture) -> str:
    """What this project's handlers logged (libraries such as Haystack log on their own)."""
    return "\n".join(
        record.getMessage() for record in caplog.records if record.name == "telegram_handlers"
    )


@pytest.mark.parametrize("make_error", OPENAI_FAILURES)
def test_openai_outage_while_recording_gets_one_generic_notice_and_counts_nothing(
    stack: _Stack,
    fake_openai: FakeOpenAIClient,
    caplog: pytest.LogCaptureFixture,
    make_error: ErrorFactory,
) -> None:
    caplog.set_level(logging.INFO)
    stack.start_session()
    fake_openai.embedding_error = make_error()

    for index in range(5):
        stack.clock.now += 1
        stack.send(f"message {index}", message_id=10 + index)  # must not raise

    stack.bot.reply_to.assert_called_once()
    assert stack.bot.reply_to.call_args.args[1] == _RECORD_INTERNAL_ERROR_REPLY
    assert stack.message_count() == 0
    log = _handler_log(caplog)
    assert log.count("Handler failed: handler=record_text") == 5
    assert "IndexingProviderError <- PipelineRuntimeError <- " in log
    assert PROVIDER_TEXT not in log
    assert BOT_TOKEN not in log


@pytest.mark.parametrize("make_error", PINECONE_FAILURES)
def test_pinecone_outage_while_recording_gets_one_generic_notice_and_counts_nothing(
    stack: _Stack,
    index: FakePineconeIndex,
    make_error: ErrorFactory,
) -> None:
    stack.start_session()
    index.upsert_error = make_error()

    stack.send("first", message_id=10)  # must not raise
    stack.send("second", message_id=11)

    stack.bot.reply_to.assert_called_once()
    assert stack.bot.reply_to.call_args.args[1] == _RECORD_INTERNAL_ERROR_REPLY
    assert stack.message_count() == 0


def test_the_outage_notice_comes_back_after_the_interval_and_recovery_resumes_recording(
    stack: _Stack,
    fake_openai: FakeOpenAIClient,
    index: FakePineconeIndex,
) -> None:
    stack.start_session()
    fake_openai.embedding_error = openai.APIConnectionError(request=REQUEST)

    stack.send("one", message_id=10)
    stack.clock.now += _RECORD_FAILURE_NOTICE_INTERVAL_SECONDS - 1
    stack.send("two", message_id=11)
    assert stack.bot.reply_to.call_count == 1

    stack.clock.now += 1
    stack.send("three", message_id=12)
    assert stack.bot.reply_to.call_count == 2

    fake_openai.embedding_error = None  # the provider recovers
    stack.send("four", message_id=13)

    assert stack.bot.reply_to.call_count == 2
    assert stack.message_count() == 1
    assert len(index.upserted) == 1


@pytest.mark.parametrize(
    "inject",
    [
        pytest.param(("document_embedder", lambda: TypeError(PROVIDER_TEXT)), id="embedder-TypeError"),
        pytest.param(("writer", lambda: TypeError(PROVIDER_TEXT)), id="writer-TypeError"),
        pytest.param(
            ("document_embedder", lambda: AssertionError(PROVIDER_TEXT)), id="embedder-AssertionError"
        ),
        pytest.param(("writer", lambda: AssertionError(PROVIDER_TEXT)), id="writer-AssertionError"),
    ],
)
def test_defect_in_an_indexing_component_reaches_telebot_and_spares_the_notice_throttle(
    stack: _Stack,
    fake_openai: FakeOpenAIClient,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    inject: tuple[str, ErrorFactory],
) -> None:
    caplog.set_level(logging.DEBUG)
    component, make_defect = inject
    stack.start_session()
    defect = make_defect()
    with monkeypatch.context() as patch:
        patch.setattr(stack.pipelines["indexing"].get_component(component), "run", _raises(defect))

        with pytest.raises(PipelineRuntimeError) as raised:
            stack.send("hello", message_id=10)

    assert raised.value.__cause__ is defect
    stack.bot.reply_to.assert_not_called()  # no generic "operational" reply
    stack.bot.send_message.assert_not_called()
    assert stack.message_count() == 0
    assert "Handler failed" not in _handler_log(caplog)
    assert PROVIDER_TEXT not in _handler_log(caplog)

    # The clock has not moved: had the defect used up the chat's notice allowance, the
    # real outage that follows would be swallowed silently.
    fake_openai.embedding_error = openai.APITimeoutError(request=REQUEST)
    stack.send("hello again", message_id=11)

    stack.bot.reply_to.assert_called_once()
    assert stack.bot.reply_to.call_args.args[1] == _RECORD_INTERNAL_ERROR_REPLY


@pytest.mark.parametrize("make_defect", PINECONE_CONTRACT_DEFECTS)
def test_pinecone_contract_defect_while_recording_reaches_telebot_and_spares_the_throttle(
    stack: _Stack,
    fake_openai: FakeOpenAIClient,
    index: FakePineconeIndex,
    make_defect: ErrorFactory,
) -> None:
    stack.start_session()
    defect = make_defect()
    index.upsert_error = defect

    with pytest.raises(PipelineRuntimeError) as raised:
        stack.send("hello", message_id=10)

    assert raised.value.__cause__ is defect
    stack.bot.reply_to.assert_not_called()
    assert stack.message_count() == 0

    index.upsert_error = ServiceError(PROVIDER_TEXT)  # a real outage follows
    stack.send("hello again", message_id=11)

    stack.bot.reply_to.assert_called_once()
    assert stack.bot.reply_to.call_args.args[1] == _RECORD_INTERNAL_ERROR_REPLY


@pytest.mark.parametrize("make_error", OPENAI_FAILURES)
def test_openai_outage_while_summarizing_gets_the_generic_reply(
    stack: _Stack,
    fake_openai: FakeOpenAIClient,
    index: FakePineconeIndex,
    caplog: pytest.LogCaptureFixture,
    make_error: ErrorFactory,
) -> None:
    caplog.set_level(logging.INFO)
    index.serve(_documents(3))
    stack.start_session(registered_messages=3)
    fake_openai.chat_error = make_error()

    stack.send("/summary", message_id=20)  # must not raise

    stack.bot.send_message.assert_called_once_with(GROUP_CHAT_ID, _SUMMARY_INTERNAL_ERROR_REPLY)
    log = _handler_log(caplog)
    assert "SummarizationProviderError <- PipelineRuntimeError <- " in log
    assert PROVIDER_TEXT not in log


@pytest.mark.parametrize("make_error", PINECONE_FAILURES)
def test_pinecone_outage_while_loading_the_session_gets_the_generic_reply(
    stack: _Stack,
    fake_openai: FakeOpenAIClient,
    index: FakePineconeIndex,
    caplog: pytest.LogCaptureFixture,
    make_error: ErrorFactory,
) -> None:
    caplog.set_level(logging.INFO)
    stack.start_session(registered_messages=3)
    index.query_error = make_error()

    stack.send("/summary", message_id=20)  # must not raise

    stack.bot.send_message.assert_called_once_with(GROUP_CHAT_ID, _SUMMARY_INTERNAL_ERROR_REPLY)
    assert fake_openai.chat_requests == []
    log = _handler_log(caplog)
    assert "SessionDocumentProviderError <- " in log
    assert PROVIDER_TEXT not in log


@pytest.mark.parametrize(
    "make_defect",
    [*PINECONE_CONTRACT_DEFECTS, pytest.param(lambda: TypeError(PROVIDER_TEXT), id="TypeError")],
)
def test_pinecone_contract_defect_while_loading_the_session_reaches_telebot(
    stack: _Stack,
    fake_openai: FakeOpenAIClient,
    index: FakePineconeIndex,
    make_defect: ErrorFactory,
) -> None:
    stack.start_session(registered_messages=3)
    defect = make_defect()
    index.query_error = defect

    with pytest.raises(type(defect)) as raised:
        stack.send("/summary", message_id=20)

    assert raised.value is defect
    stack.bot.send_message.assert_not_called()  # not turned into the generic summary error
    assert fake_openai.chat_requests == []


@pytest.mark.parametrize("component", ["prompt_builder", "llm"])
@pytest.mark.parametrize("make_defect", COMPONENT_DEFECTS)
def test_defect_in_a_summarization_component_reaches_telebot(
    stack: _Stack,
    index: FakePineconeIndex,
    monkeypatch: pytest.MonkeyPatch,
    component: str,
    make_defect: ErrorFactory,
) -> None:
    index.serve(_documents(3))
    stack.start_session(registered_messages=3)
    defect = make_defect()
    monkeypatch.setattr(
        stack.pipelines["summarization"].get_component(component), "run", _raises(defect)
    )

    with pytest.raises(PipelineRuntimeError) as raised:
        stack.send("/summary", message_id=20)

    assert raised.value.__cause__ is defect
    stack.bot.send_message.assert_not_called()


def test_a_whole_summary_still_reaches_the_chat(
    stack: _Stack,
    fake_openai: FakeOpenAIClient,
    index: FakePineconeIndex,
) -> None:
    index.serve(_documents(3))
    stack.start_session(registered_messages=3)

    stack.send("/summary", message_id=20)

    stack.bot.send_message.assert_called_once_with(GROUP_CHAT_ID, fake_openai.reply)
    assert len(fake_openai.chat_requests) == 1
