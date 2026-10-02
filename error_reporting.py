"""Credential-safe exception descriptions for logs."""

from __future__ import annotations

import os
from typing import Final

_MAX_CHAIN_DEPTH: Final[int] = 5


def describe_exception(exc: BaseException) -> str:
    """Describe an exception without ever including its message.

    Exception messages can embed credentials or user content. For example, a
    ``requests`` error raised by a Telegram call contains the full request URL,
    including the bot token. Only the exception type names along the cause chain
    and the innermost source location are reported, which is enough to diagnose
    a failure without exposing message text.
    """
    names: list[str] = []
    seen: set[int] = set()
    innermost = exc
    current: BaseException | None = exc
    while current is not None and id(current) not in seen and len(names) < _MAX_CHAIN_DEPTH:
        seen.add(id(current))
        names.append(type(current).__name__)
        innermost = current
        if current.__cause__ is not None:
            current = current.__cause__
        elif current.__suppress_context__:
            current = None
        else:
            current = current.__context__

    description = " <- ".join(names)
    location = _innermost_location(innermost)
    if location is None:
        return description
    return f"{description} at {location}"


def _innermost_location(exc: BaseException) -> str | None:
    traceback = exc.__traceback__
    if traceback is None:
        return None
    while traceback.tb_next is not None:
        traceback = traceback.tb_next
    filename = os.path.basename(traceback.tb_frame.f_code.co_filename)
    function = traceback.tb_frame.f_code.co_name
    return f"{filename}:{traceback.tb_lineno} in {function}"
