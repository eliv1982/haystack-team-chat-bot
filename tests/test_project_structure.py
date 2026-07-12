"""Tests for project structure, imports, and repository conventions."""

from __future__ import annotations

import ast
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
    "telegram_application",
    "telegram_summary_application",
    "telegram_handlers",
    "telegram_bot",
    "telegram_commands",
    "runtime",
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


def test_stage_5b_runtime_modules_exist() -> None:
    assert (PROJECT_ROOT / "telegram_application.py").is_file()
    assert (PROJECT_ROOT / "telegram_handlers.py").is_file()


def test_stage_5b_imports_do_not_create_global_bot_store_or_service() -> None:
    import telegram_application
    import telegram_handlers

    assert not hasattr(telegram_application, "bot")
    assert not hasattr(telegram_application, "store")
    assert not hasattr(telegram_application, "application_service")
    assert not hasattr(telegram_handlers, "bot")
    assert not hasattr(telegram_handlers, "application_service")

    application_source = (PROJECT_ROOT / "telegram_application.py").read_text(encoding="utf-8")
    assert "summarization_service" not in application_source
    assert "SummarizationService" not in application_source


def test_stage_5c_runtime_modules_exist() -> None:
    assert (PROJECT_ROOT / "telegram_summary_application.py").is_file()


def test_stage_5c_imports_do_not_create_global_dependencies() -> None:
    import telegram_summary_application

    assert not hasattr(telegram_summary_application, "bot")
    assert not hasattr(telegram_summary_application, "store")
    assert not hasattr(telegram_summary_application, "application_service")

    source = (PROJECT_ROOT / "telegram_summary_application.py").read_text(encoding="utf-8")
    assert "Pipeline(" not in source
    assert "Pinecone" not in source
    assert "OpenAI" not in source
    assert "IndexingService" not in source


def test_stage_6a_runtime_module_exists() -> None:
    assert (PROJECT_ROOT / "runtime.py").is_file()


def test_stage_6b1_modules_exist() -> None:
    assert (PROJECT_ROOT / "telegram_commands.py").is_file()
    assert (PROJECT_ROOT / "docs" / "project_roadmap.md").is_file()


def test_bot_has_main_and_entry_guard() -> None:
    bot_source = (PROJECT_ROOT / "bot.py").read_text(encoding="utf-8")
    bot_tree = ast.parse(bot_source)

    function_names = {
        node.name
        for node in bot_tree.body
        if isinstance(node, ast.FunctionDef)
    }
    assert "main" in function_names

    main_guard_found = False
    for node in bot_tree.body:
        if not isinstance(node, ast.If):
            continue
        test = node.test
        if (
            isinstance(test, ast.Compare)
            and isinstance(test.left, ast.Name)
            and test.left.id == "__name__"
            and len(test.comparators) == 1
            and isinstance(test.comparators[0], ast.Constant)
            and test.comparators[0].value == "__main__"
        ):
            main_guard_found = True
            break
    assert main_guard_found


def test_stage_6a_imports_do_not_create_global_dependencies() -> None:
    import bot
    import runtime

    for module in (runtime, bot):
        assert not hasattr(module, "bot")
        assert not hasattr(module, "store")
        assert not hasattr(module, "session_store")
        assert not hasattr(module, "runtime")
        assert not hasattr(module, "pipeline")


def test_polling_occurs_only_in_run_polling() -> None:
    runtime_source = (PROJECT_ROOT / "runtime.py").read_text(encoding="utf-8")
    runtime_tree = ast.parse(runtime_source)

    polling_call_sites: list[str] = []

    class PollingVisitor(ast.NodeVisitor):
        def __init__(self) -> None:
            self.current_function: str | None = None

        def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
            previous = self.current_function
            self.current_function = node.name
            self.generic_visit(node)
            self.current_function = previous

        def visit_Call(self, node: ast.Call) -> None:
            if isinstance(node.func, ast.Attribute) and node.func.attr in {
                "infinity_polling",
                "polling",
            }:
                polling_call_sites.append(self.current_function or "<module>")
            self.generic_visit(node)

    PollingVisitor().visit(runtime_tree)

    assert polling_call_sites == ["run_polling"]


def test_bot_does_not_call_polling_directly() -> None:
    bot_source = (PROJECT_ROOT / "bot.py").read_text(encoding="utf-8")
    bot_tree = ast.parse(bot_source)

    polling_attrs: list[str] = []

    class PollingVisitor(ast.NodeVisitor):
        def visit_Call(self, node: ast.Call) -> None:
            if isinstance(node.func, ast.Attribute) and node.func.attr in {
                "infinity_polling",
                "polling",
            }:
                polling_attrs.append(node.func.attr)
            self.generic_visit(node)

    PollingVisitor().visit(bot_tree)
    assert polling_attrs == []
