"""Tests for credential-safe exception descriptions."""

from __future__ import annotations

import re

import pytest

from error_reporting import describe_exception

SECRET = "123456789:AAH-s3cretTokenValue_0123456789abcdefghi"


def _raise(exc: BaseException) -> None:
    raise exc


def _caught(exc: BaseException) -> BaseException:
    try:
        _raise(exc)
    except BaseException as caught:  # noqa: BLE001
        return caught
    raise AssertionError("unreachable")


def test_description_contains_type_but_never_the_message() -> None:
    description = describe_exception(_caught(RuntimeError(f"failed at /bot{SECRET}/getMe")))

    assert description.startswith("RuntimeError")
    assert SECRET not in description
    assert "getMe" not in description


def test_description_names_the_innermost_source_location() -> None:
    description = describe_exception(_caught(ValueError("boom")))

    assert re.fullmatch(r"ValueError at test_error_reporting\.py:\d+ in _raise", description)


def test_description_follows_the_explicit_cause_chain_by_type_only() -> None:
    try:
        try:
            raise ConnectionError(f"url with {SECRET}")
        except ConnectionError as cause:
            raise RuntimeError(f"wrapper {SECRET}") from cause
    except RuntimeError as exc:
        description = describe_exception(exc)

    assert description.startswith("RuntimeError <- ConnectionError")
    assert SECRET not in description


def test_description_follows_implicit_context_unless_suppressed() -> None:
    try:
        try:
            raise ConnectionError("inner")
        except ConnectionError:
            raise RuntimeError("outer")  # noqa: B904
    except RuntimeError as chained:
        assert describe_exception(chained).startswith("RuntimeError <- ConnectionError")

    try:
        try:
            raise ConnectionError("inner")
        except ConnectionError:
            raise RuntimeError("outer") from None
    except RuntimeError as suppressed:
        assert describe_exception(suppressed).startswith("RuntimeError")
        assert "ConnectionError" not in describe_exception(suppressed)


def test_description_without_traceback_has_no_location() -> None:
    assert describe_exception(RuntimeError("never raised")) == "RuntimeError"


def test_description_survives_cyclic_exception_chains() -> None:
    first = RuntimeError("first")
    second = ValueError("second")
    first.__cause__ = second
    second.__cause__ = first

    assert describe_exception(first) == "RuntimeError <- ValueError"


def test_description_bounds_very_long_chains() -> None:
    exc: BaseException = RuntimeError("root")
    for index in range(50):
        wrapper = ValueError(str(index))
        wrapper.__cause__ = exc
        exc = wrapper

    assert describe_exception(exc).count("<-") == 4


@pytest.mark.parametrize("exc", [KeyboardInterrupt(), SystemExit(1), GeneratorExit()])
def test_description_accepts_base_exceptions(exc: BaseException) -> None:
    assert describe_exception(exc) == type(exc).__name__
