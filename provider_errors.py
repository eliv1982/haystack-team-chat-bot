"""Recognition of expected provider failures, by exception type only.

``Pipeline.run()`` wraps whatever a component raises in ``PipelineRuntimeError``
(``raise ... from``), and that includes a ``TypeError`` or ``AssertionError`` caused by
a bug. A failed pipeline is therefore an outage of OpenAI or Pinecone only if the
exception the component actually raised is a failed request to that provider. This
module answers that question; message text is never inspected.
"""

from __future__ import annotations

from typing import Final

import openai
from haystack.core.errors import PipelineRuntimeError
from pinecone import ApiError, PineconeConnectionError, PineconeTimeoutError

# A request to the provider failed: it answered with an error status (rate limit,
# outage, rejected key, ...) or could not be reached. Deliberately not the SDKs' base
# classes: ``openai.OpenAIError`` also covers client misuse, and ``PineconeError``
# also covers ``PineconeTypeError`` and ``PineconeValueError``, which the SDK raises
# when this code passed it a malformed argument. A response that could not be parsed
# (``openai.APIResponseValidationError``, ``ResponseParsingError``) is left out too: it
# is a contract violation, not an outage.
OPENAI_REQUEST_FAILURES: Final[tuple[type[BaseException], ...]] = (
    openai.APIStatusError,
    openai.APIConnectionError,  # includes APITimeoutError
)
PINECONE_REQUEST_FAILURES: Final[tuple[type[BaseException], ...]] = (
    ApiError,  # every HTTP error response, including the rate limit and 5xx subclasses
    PineconeConnectionError,
    PineconeTimeoutError,
)

# Haystack 2.31 wraps a component's exception exactly once for the pipelines used
# here (observed); the bound only keeps the walk finite if that changes or the chain is cyclic.
_MAX_PIPELINE_ERROR_NESTING: Final[int] = 5


def raised_by_component(error: PipelineRuntimeError) -> BaseException | None:
    """Return the exception a pipeline component raised, or ``None`` if there is none.

    Follows only the explicit ``__cause__`` link that Haystack sets and looks through
    nested ``PipelineRuntimeError`` layers. ``__context__`` is ignored on purpose: it
    also records an unrelated exception that was being handled when a bug occurred.
    """
    current: BaseException | None = error
    for _ in range(_MAX_PIPELINE_ERROR_NESTING):
        if not isinstance(current, PipelineRuntimeError):
            return current
        current = current.__cause__
    return None


def is_provider_failure(
    error: PipelineRuntimeError,
    provider_failures: tuple[type[BaseException], ...],
) -> bool:
    """Whether a failed pipeline run was caused by one of the given provider failures."""
    return isinstance(raised_by_component(error), provider_failures)
