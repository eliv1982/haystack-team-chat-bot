"""Tests for recognizing provider failures by exception type, never by message text.

The exceptions here are built the way the real stack produces them: Haystack wraps what
a component raised with ``raise PipelineRuntimeError(...) from error``.
"""

from __future__ import annotations

import httpx
import openai
import pytest
from haystack.core.errors import PipelineRuntimeError
from pinecone import (
    ApiError,
    ConflictError,
    ForbiddenError,
    IndexInitFailedError,
    NotFoundError,
    PineconeConnectionError,
    PineconeError,
    PineconeTimeoutError,
    PineconeTypeError,
    PineconeValueError,
    RateLimitError,
    ResponseParsingError,
    ServiceError,
    UnauthorizedError,
)
from pinecone.exceptions import PineconeApiTypeError

from provider_errors import (
    OPENAI_REQUEST_FAILURES,
    PINECONE_REQUEST_FAILURES,
    is_provider_failure,
    raised_by_component,
)

_REQUEST = httpx.Request("POST", "https://api.example.com/v1/embeddings")


def _status_error(error_type: type[openai.APIStatusError], status: int) -> openai.APIStatusError:
    return error_type("provider said no", response=httpx.Response(status, request=_REQUEST), body=None)


def wrapped(cause: BaseException | None, *, layers: int = 1) -> PipelineRuntimeError:
    """What ``Pipeline.run()`` raises when a component raised ``cause``."""
    error: PipelineRuntimeError | None = None
    current = cause
    for _ in range(layers):
        try:
            if current is None:
                raise PipelineRuntimeError("component", None, "failed")
            raise PipelineRuntimeError("component", None, "failed") from current
        except PipelineRuntimeError as raised:
            error = raised
            current = raised
    assert error is not None
    return error


OPENAI_PROVIDER_FAILURES = [
    pytest.param(lambda: openai.APIConnectionError(request=_REQUEST), id="APIConnectionError"),
    pytest.param(lambda: openai.APITimeoutError(request=_REQUEST), id="APITimeoutError"),
    pytest.param(lambda: _status_error(openai.RateLimitError, 429), id="RateLimitError"),
    pytest.param(lambda: _status_error(openai.InternalServerError, 500), id="InternalServerError"),
    pytest.param(lambda: _status_error(openai.AuthenticationError, 401), id="AuthenticationError"),
    pytest.param(lambda: _status_error(openai.BadRequestError, 400), id="BadRequestError"),
]

PINECONE_PROVIDER_FAILURES = [
    pytest.param(lambda: ServiceError(), id="ServiceError"),
    pytest.param(lambda: RateLimitError(), id="RateLimitError"),
    pytest.param(lambda: UnauthorizedError(), id="UnauthorizedError"),
    pytest.param(lambda: ForbiddenError(), id="ForbiddenError"),
    pytest.param(lambda: NotFoundError(), id="NotFoundError"),
    pytest.param(lambda: ConflictError(), id="ConflictError"),
    pytest.param(lambda: ApiError("rejected", 400), id="ApiError-other-status"),
    pytest.param(lambda: PineconeConnectionError("unreachable"), id="PineconeConnectionError"),
    pytest.param(lambda: PineconeTimeoutError("timed out"), id="PineconeTimeoutError"),
]

# Bugs in this code, or in how it uses a library, that must never read as an outage.
# Several of them are what the provider SDKs themselves raise for a malformed call.
NOT_PROVIDER_FAILURES = [
    pytest.param(lambda: TypeError("bug"), id="TypeError"),
    pytest.param(lambda: AssertionError("bug"), id="AssertionError"),
    pytest.param(lambda: ValueError("bug"), id="ValueError"),
    pytest.param(lambda: KeyError("bug"), id="KeyError"),
    pytest.param(lambda: RuntimeError("bug"), id="RuntimeError"),
    pytest.param(lambda: TimeoutError("builtin"), id="builtin-TimeoutError"),
    pytest.param(lambda: ConnectionError("builtin"), id="builtin-ConnectionError"),
    pytest.param(lambda: openai.OpenAIError("client misuse"), id="OpenAIError"),
    pytest.param(
        lambda: openai.APIResponseValidationError(
            response=httpx.Response(200, request=_REQUEST), body=None
        ),
        id="APIResponseValidationError",
    ),
    pytest.param(lambda: PineconeApiTypeError("bad argument type"), id="PineconeApiTypeError"),
    pytest.param(lambda: PineconeTypeError("bad argument type"), id="PineconeTypeError"),
    pytest.param(lambda: PineconeValueError("bad argument value"), id="PineconeValueError"),
    pytest.param(lambda: ResponseParsingError("unparseable body"), id="ResponseParsingError"),
    pytest.param(lambda: IndexInitFailedError("index"), id="IndexInitFailedError"),
    pytest.param(lambda: PineconeError("generic"), id="PineconeError-base"),
]


@pytest.mark.parametrize("make_error", OPENAI_PROVIDER_FAILURES)
def test_openai_request_failures_are_recognized(make_error) -> None:  # type: ignore[no-untyped-def]
    assert is_provider_failure(wrapped(make_error()), OPENAI_REQUEST_FAILURES)


@pytest.mark.parametrize("make_error", PINECONE_PROVIDER_FAILURES)
def test_pinecone_request_failures_are_recognized(make_error) -> None:  # type: ignore[no-untyped-def]
    assert is_provider_failure(wrapped(make_error()), PINECONE_REQUEST_FAILURES)


@pytest.mark.parametrize("make_error", NOT_PROVIDER_FAILURES)
def test_bugs_and_malformed_calls_are_not_provider_failures(make_error) -> None:  # type: ignore[no-untyped-def]
    error = wrapped(make_error())

    assert not is_provider_failure(error, OPENAI_REQUEST_FAILURES)
    assert not is_provider_failure(error, PINECONE_REQUEST_FAILURES)
    assert not is_provider_failure(error, (*OPENAI_REQUEST_FAILURES, *PINECONE_REQUEST_FAILURES))


def test_each_provider_is_recognized_only_where_the_pipeline_uses_it() -> None:
    pinecone_error = wrapped(ServiceError())
    openai_error = wrapped(openai.APIConnectionError(request=_REQUEST))

    assert not is_provider_failure(pinecone_error, OPENAI_REQUEST_FAILURES)
    assert not is_provider_failure(openai_error, PINECONE_REQUEST_FAILURES)


def test_sdk_hierarchy_the_allowlists_rely_on_has_not_shifted() -> None:
    """The allowlists name SDK classes; these facts are what makes them safe to use."""
    # Pinecone: the malformed-call errors are siblings of the operational ones, and the
    # legacy ``PineconeApiTypeError`` is the very class that is also a ``TypeError``.
    assert PineconeApiTypeError is PineconeTypeError
    assert issubclass(PineconeTypeError, TypeError)
    assert issubclass(PineconeValueError, ValueError)
    for malformed_call in (PineconeTypeError, PineconeValueError, ResponseParsingError):
        assert issubclass(malformed_call, PineconeError)
        assert not issubclass(malformed_call, PINECONE_REQUEST_FAILURES)
    # Haystack's integration and old callers catch the base class; it is wider than the allowlist.
    assert issubclass(PineconeTypeError, PineconeError)
    assert PineconeError not in PINECONE_REQUEST_FAILURES
    # OpenAI: timeouts are connection errors, and the base error is wider than the allowlist.
    assert issubclass(openai.APITimeoutError, openai.APIConnectionError)
    assert not issubclass(openai.OpenAIError, OPENAI_REQUEST_FAILURES)


def test_message_text_never_decides() -> None:
    outage_wording = "503 Service Unavailable: rate limit exceeded, connection refused by openai"
    bug = wrapped(TypeError(outage_wording))
    outage = wrapped(PineconeConnectionError("something unrelated"))

    assert not is_provider_failure(bug, (*OPENAI_REQUEST_FAILURES, *PINECONE_REQUEST_FAILURES))
    assert is_provider_failure(outage, PINECONE_REQUEST_FAILURES)


# --- which link of the chain counts ----------------------------------------------


def test_the_exception_the_component_raised_is_the_explicit_cause() -> None:
    component_error = ServiceError()

    assert raised_by_component(wrapped(component_error)) is component_error


def test_a_defect_raised_while_handling_a_provider_failure_is_a_defect() -> None:
    """``__context__`` records what was being handled; only ``__cause__`` says what failed."""
    try:
        try:
            raise ServiceError()
        except ServiceError:
            raise TypeError("bug in the error handling")
    except TypeError as defect:
        implicit = defect
    assert isinstance(implicit.__context__, ServiceError)

    error = wrapped(implicit)

    assert raised_by_component(error) is implicit
    assert not is_provider_failure(error, PINECONE_REQUEST_FAILURES)


def test_a_defect_chained_from_a_provider_failure_is_a_defect() -> None:
    try:
        try:
            raise ServiceError()
        except ServiceError as outage:
            raise TypeError("bug that wraps the outage") from outage
    except TypeError as defect:
        chained = defect

    error = wrapped(chained)

    assert isinstance(chained.__cause__, ServiceError)
    assert not is_provider_failure(error, PINECONE_REQUEST_FAILURES)


def test_pipeline_error_without_a_cause_is_not_a_provider_failure() -> None:
    error = PipelineRuntimeError.from_invalid_output("component", object, 5)

    assert error.__cause__ is None
    assert raised_by_component(error) is None
    assert not is_provider_failure(error, PINECONE_REQUEST_FAILURES)


def test_nested_pipeline_errors_are_looked_through() -> None:
    outage = ServiceError()
    bug = TypeError("bug")

    assert raised_by_component(wrapped(outage, layers=3)) is outage
    assert is_provider_failure(wrapped(outage, layers=3), PINECONE_REQUEST_FAILURES)
    assert raised_by_component(wrapped(bug, layers=3)) is bug
    assert not is_provider_failure(wrapped(bug, layers=3), PINECONE_REQUEST_FAILURES)


def test_absurdly_deep_nesting_fails_safe() -> None:
    error = wrapped(ServiceError(), layers=50)

    assert raised_by_component(error) is None
    assert not is_provider_failure(error, PINECONE_REQUEST_FAILURES)


def test_a_cyclic_cause_chain_terminates() -> None:
    first = PipelineRuntimeError("a", None, "first")
    second = PipelineRuntimeError("b", None, "second")
    first.__cause__ = second
    second.__cause__ = first

    assert raised_by_component(first) is None
    assert not is_provider_failure(first, PINECONE_REQUEST_FAILURES)
