"""Tests for configuration loading and validation."""

from __future__ import annotations

import pytest

from config import (
    DEFAULT_PINECONE_DIMENSION,
    DEFAULT_PINECONE_METRIC,
    DEFAULT_PINECONE_NAMESPACE,
    DEFAULT_RETRIEVAL_TOP_K,
    ConfigurationError,
    Settings,
    load_settings,
)

REQUIRED_ENV = {
    "TELEGRAM_BOT_TOKEN": "test-telegram-token",
    "OPENAI_API_KEY": "test-openai-key",
    "OPENAI_MODEL": "test-chat-model",
    "EMBEDDING_MODEL": "test-embedding-model",
    "PINECONE_API_KEY": "test-pinecone-key",
    "PINECONE_INDEX_NAME": "test-index",
}

CONFIG_ENV_KEYS = {
    "TELEGRAM_BOT_TOKEN",
    "OPENAI_API_KEY",
    "OPENAI_BASE_URL",
    "OPENAI_MODEL",
    "EMBEDDING_MODEL",
    "PINECONE_API_KEY",
    "PINECONE_INDEX_NAME",
    "PINECONE_NAMESPACE",
    "PINECONE_DIMENSION",
    "PINECONE_METRIC",
    "RETRIEVAL_TOP_K",
}


@pytest.fixture
def clean_config_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for key in CONFIG_ENV_KEYS:
        monkeypatch.delenv(key, raising=False)


def _set_env(monkeypatch: pytest.MonkeyPatch, values: dict[str, str]) -> None:
    for key, value in values.items():
        monkeypatch.setenv(key, value)


def test_missing_required_variable(clean_config_env: None, monkeypatch: pytest.MonkeyPatch) -> None:
    _set_env(monkeypatch, {k: v for k, v in REQUIRED_ENV.items() if k != "OPENAI_API_KEY"})

    with pytest.raises(ConfigurationError, match="OPENAI_API_KEY"):
        load_settings(dotenv_path=None)


def test_empty_required_variable(clean_config_env: None, monkeypatch: pytest.MonkeyPatch) -> None:
    _set_env(monkeypatch, REQUIRED_ENV)
    monkeypatch.setenv("OPENAI_MODEL", "   ")

    with pytest.raises(ConfigurationError, match="OPENAI_MODEL"):
        load_settings(dotenv_path=None)


def test_load_full_settings(clean_config_env: None, monkeypatch: pytest.MonkeyPatch) -> None:
    _set_env(monkeypatch, REQUIRED_ENV)
    monkeypatch.setenv("OPENAI_BASE_URL", "https://api.example.com/v1")
    monkeypatch.setenv("PINECONE_NAMESPACE", "custom-namespace")
    monkeypatch.setenv("PINECONE_DIMENSION", "2048")
    monkeypatch.setenv("PINECONE_METRIC", "dotproduct")
    monkeypatch.setenv("RETRIEVAL_TOP_K", "25")

    settings = load_settings(dotenv_path=None)

    assert isinstance(settings, Settings)
    assert settings.telegram_bot_token == "test-telegram-token"
    assert settings.openai_api_key == "test-openai-key"
    assert settings.api_base_url == "https://api.example.com/v1"
    assert settings.openai_model == "test-chat-model"
    assert settings.embedding_model == "test-embedding-model"
    assert settings.pinecone_api_key == "test-pinecone-key"
    assert settings.pinecone_index_name == "test-index"
    assert settings.pinecone_namespace == "custom-namespace"
    assert settings.pinecone_dimension == 2048
    assert settings.pinecone_metric == "dotproduct"
    assert settings.retrieval_top_k == 25


def test_defaults_for_optional_values(clean_config_env: None, monkeypatch: pytest.MonkeyPatch) -> None:
    _set_env(monkeypatch, REQUIRED_ENV)

    settings = load_settings(dotenv_path=None)

    assert settings.api_base_url is None
    assert settings.pinecone_namespace == DEFAULT_PINECONE_NAMESPACE
    assert settings.pinecone_dimension == DEFAULT_PINECONE_DIMENSION
    assert settings.pinecone_metric == DEFAULT_PINECONE_METRIC
    assert settings.retrieval_top_k == DEFAULT_RETRIEVAL_TOP_K


@pytest.mark.parametrize("raw_value", ["not-a-number", "12.5"])
def test_invalid_integer(
    clean_config_env: None,
    monkeypatch: pytest.MonkeyPatch,
    raw_value: str,
) -> None:
    _set_env(monkeypatch, REQUIRED_ENV)
    monkeypatch.setenv("PINECONE_DIMENSION", raw_value)

    with pytest.raises(ConfigurationError, match="PINECONE_DIMENSION"):
        load_settings(dotenv_path=None)


@pytest.mark.parametrize(
    ("variable", "raw_value"),
    [
        ("PINECONE_DIMENSION", "0"),
        ("PINECONE_DIMENSION", "-1"),
        ("RETRIEVAL_TOP_K", "0"),
        ("RETRIEVAL_TOP_K", "-5"),
    ],
)
def test_zero_and_negative_integer(
    clean_config_env: None,
    monkeypatch: pytest.MonkeyPatch,
    variable: str,
    raw_value: str,
) -> None:
    _set_env(monkeypatch, REQUIRED_ENV)
    monkeypatch.setenv(variable, raw_value)

    with pytest.raises(ConfigurationError, match=variable):
        load_settings(dotenv_path=None)


def test_invalid_metric(clean_config_env: None, monkeypatch: pytest.MonkeyPatch) -> None:
    _set_env(monkeypatch, REQUIRED_ENV)
    monkeypatch.setenv("PINECONE_METRIC", "invalid-metric")

    with pytest.raises(ConfigurationError, match="PINECONE_METRIC"):
        load_settings(dotenv_path=None)


def test_error_messages_do_not_reveal_secret_values(
    clean_config_env: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    secret_value = "super-secret-token-value-12345"
    _set_env(monkeypatch, REQUIRED_ENV)
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", secret_value)

    with pytest.raises(ConfigurationError) as exc_info:
        monkeypatch.setenv("OPENAI_API_KEY", "")
        load_settings(dotenv_path=None)

    message = str(exc_info.value)
    assert "OPENAI_API_KEY" in message
    assert secret_value not in message


@pytest.mark.parametrize("raw_value", ["", "   "])
def test_empty_or_whitespace_openai_base_url_normalizes_to_none(
    clean_config_env: None,
    monkeypatch: pytest.MonkeyPatch,
    raw_value: str,
) -> None:
    _set_env(monkeypatch, REQUIRED_ENV)
    monkeypatch.setenv("OPENAI_BASE_URL", raw_value)

    settings = load_settings(dotenv_path=None)

    assert settings.api_base_url is None


def test_trimmed_custom_openai_base_url_is_preserved(
    clean_config_env: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _set_env(monkeypatch, REQUIRED_ENV)
    monkeypatch.setenv("OPENAI_BASE_URL", "  https://api.example.com/v1  ")

    settings = load_settings(dotenv_path=None)

    assert settings.api_base_url == "https://api.example.com/v1"


def test_settings_repr_and_str_hide_credentials(
    clean_config_env: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    secrets = {
        "TELEGRAM_BOT_TOKEN": "123456789:AAH-s3cretTokenValue_0123456789abcdefghi",
        "OPENAI_API_KEY": "sk-test-s3cret-openai-key",
        "PINECONE_API_KEY": "pcsk-s3cret-pinecone-key",
    }
    _set_env(monkeypatch, {**REQUIRED_ENV, **secrets})

    settings = load_settings(dotenv_path=None)

    for rendered in (repr(settings), str(settings), f"{settings!r}", f"{settings}"):
        for secret in secrets.values():
            assert secret not in rendered
        for secret_part in ("s3cret", "AAH-"):
            assert secret_part not in rendered
        # Non-secret fields stay visible so the representation remains useful.
        assert "test-index" in rendered
        assert "test-chat-model" in rendered


def test_settings_secret_fields_remain_usable_and_comparable(
    clean_config_env: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _set_env(monkeypatch, REQUIRED_ENV)

    first = load_settings(dotenv_path=None)
    second = load_settings(dotenv_path=None)

    assert first.telegram_bot_token == "test-telegram-token"
    assert first.openai_api_key == "test-openai-key"
    assert first.pinecone_api_key == "test-pinecone-key"
    assert first == second
