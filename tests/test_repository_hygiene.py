"""Repository hygiene: import side effects, git ignore rules, offline test harness."""

from __future__ import annotations

import os
import socket
import subprocess
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Run in a fresh interpreter with no credentials in the environment and no network:
# importing the application must neither read a .env file, nor reach a service, nor
# create a bot, pipeline or store as a module-level global.
_IMPORT_PROBE = """
import importlib
import os
import socket
import sys


def refuse(*args, **kwargs):
    raise AssertionError("network access while importing")


socket.socket.connect = refuse
socket.getaddrinfo = refuse

environment_before = dict(os.environ)
modules = [importlib.import_module(name) for name in sys.argv[1:]]
assert dict(os.environ) == environment_before, "importing changed os.environ (.env loaded?)"

import telebot
from haystack import Pipeline
from haystack_integrations.document_stores.pinecone import PineconeDocumentStore

from session_store import InMemorySessionStore

forbidden = (telebot.TeleBot, Pipeline, PineconeDocumentStore, InMemorySessionStore)
for module in modules:
    for name, value in vars(module).items():
        assert not isinstance(value, forbidden), f"{module.__name__}.{name} is built at import time"
"""

_CREDENTIAL_VARIABLES = (
    "TELEGRAM_BOT_TOKEN",
    "OPENAI_API_KEY",
    "OPENAI_BASE_URL",
    "OPENAI_MODEL",
    "EMBEDDING_MODEL",
    "PINECONE_API_KEY",
    "PINECONE_INDEX_NAME",
)


def _application_modules() -> list[str]:
    return sorted(path.stem for path in PROJECT_ROOT.glob("*.py"))


def test_importing_the_application_has_no_side_effects() -> None:
    modules = _application_modules()
    assert {"bot", "runtime", "config", "telegram_handlers"} <= set(modules)
    environment = {k: v for k, v in os.environ.items() if k not in _CREDENTIAL_VARIABLES}

    result = subprocess.run(
        [sys.executable, "-c", _IMPORT_PROBE, *modules],
        cwd=PROJECT_ROOT,  # a developer's local .env sits here
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr


def _git_ignores(path: str) -> bool:
    try:
        result = subprocess.run(
            ["git", "check-ignore", "-q", "--no-index", path],
            cwd=PROJECT_ROOT,
            capture_output=True,
            check=False,
        )
    except FileNotFoundError:
        pytest.skip("git is not installed")
    if result.returncode == 128:
        pytest.skip("not a git checkout")
    return result.returncode == 0


@pytest.mark.parametrize(
    "path",
    [
        ".env",
        ".venv/pyvenv.cfg",
        "__pycache__/bot.cpython-312.pyc",
        "tests/__pycache__/conftest.cpython-312.pyc",
        "module.pyc",
        ".pytest_cache/README.md",
        ".ruff_cache/CACHEDIR.TAG",
    ],
)
def test_secrets_and_local_artifacts_are_git_ignored(path: str) -> None:
    assert _git_ignores(path), f"{path} must be ignored by git"


def test_env_example_is_not_git_ignored() -> None:
    assert (PROJECT_ROOT / ".env.example").is_file()
    assert not _git_ignores(".env.example"), ".env.example must stay committable"


def test_haystack_telemetry_is_disabled_for_automated_runs() -> None:
    from haystack.telemetry import _telemetry

    assert os.environ["HAYSTACK_TELEMETRY_ENABLED"].lower() == "false"
    assert _telemetry.telemetry is None  # Haystack leaves no telemetry client when opted out


def test_offline_guard_refuses_remote_hosts() -> None:
    # The suite must never reach OpenAI, Pinecone or Telegram: see the autouse guard.
    with pytest.raises(RuntimeError, match="offline tests"):
        socket.getaddrinfo("api.openai.com", 443)
    with pytest.raises(RuntimeError, match="offline tests"):
        socket.create_connection(("203.0.113.1", 443), timeout=1)


def test_offline_guard_still_allows_loopback() -> None:
    with socket.socket() as server:
        server.bind(("127.0.0.1", 0))
        server.listen()
        socket.create_connection(server.getsockname(), timeout=2).close()
