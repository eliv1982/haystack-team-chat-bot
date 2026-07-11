"""Tests for project structure, imports, and repository conventions."""

from __future__ import annotations

import importlib
import subprocess
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
NOTEBOOK_NAME = "Pipeline example.ipynb"
PYTHON_MODULES = (
    "config",
    "models",
    "documents",
    "document_store",
    "pinecone_preflight",
    "retrieval_filters",
    "pipelines",
    "indexing_service",
    "retrieval_service",
    "summarization_prompt",
    "summarization_service",
    "session_store",
    "telegram_adapter",
    "telegram_bot",
    "bot",
)


def test_notebook_exists_in_project_root() -> None:
    notebook_path = PROJECT_ROOT / NOTEBOOK_NAME
    assert notebook_path.is_file()
    assert notebook_path.name == NOTEBOOK_NAME


@pytest.mark.parametrize("module_name", PYTHON_MODULES)
def test_module_import_without_env_or_network(module_name: str) -> None:
    module = importlib.import_module(module_name)
    assert module is not None


def test_gitignore_rules() -> None:
    gitignore_path = PROJECT_ROOT / ".gitignore"
    assert gitignore_path.is_file()

    patterns = {
        line.strip()
        for line in gitignore_path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.strip().startswith("#")
    }

    assert ".env" in patterns
    assert "!.env.example" in patterns
    assert ".venv/" in patterns
    assert "__pycache__/" in patterns
    assert ".pytest_cache/" in patterns
    assert ".ruff_cache/" in patterns
    assert "*.pyc" in patterns


def test_env_example_is_not_ignored_by_git() -> None:
    env_example = PROJECT_ROOT / ".env.example"
    assert env_example.is_file()

    result = subprocess.run(
        [
            "git",
            "ls-files",
            "--others",
            "--ignored",
            "--exclude-standard",
            "--",
            ".env.example",
        ],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert not result.stdout.strip(), ".env.example must not be ignored by git"


def test_env_file_is_ignored_by_git() -> None:
    result = subprocess.run(
        ["git", "check-ignore", "-v", ".env"],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, ".env must be ignored by git"


def test_stage_3a_runtime_modules_exist() -> None:
    assert (PROJECT_ROOT / "retrieval_filters.py").is_file()
    assert (PROJECT_ROOT / "retrieval_service.py").is_file()


def test_stage_4a_runtime_modules_exist() -> None:
    assert (PROJECT_ROOT / "summarization_prompt.py").is_file()
    assert (PROJECT_ROOT / "summarization_service.py").is_file()


def test_stage_5a_runtime_modules_exist() -> None:
    assert (PROJECT_ROOT / "telegram_adapter.py").is_file()
    assert (PROJECT_ROOT / "telegram_bot.py").is_file()


def test_stage_5a_imports_do_not_create_global_bot_or_store() -> None:
    import telegram_adapter
    import telegram_bot

    assert not hasattr(telegram_bot, "bot")
    assert not hasattr(telegram_bot, "store")
    assert not hasattr(telegram_adapter, "bot")
    assert not hasattr(telegram_adapter, "store")
    assert not hasattr(telegram_adapter, "session_store")


def test_bot_does_not_start_polling() -> None:
    import bot

    with pytest.raises(NotImplementedError, match="polling"):
        bot.main()
