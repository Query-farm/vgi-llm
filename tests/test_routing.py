"""Provider routing precedence and prefix resolution (no network)."""

from __future__ import annotations

import pytest

from vgi_llm.providers import (
    AnthropicProvider,
    OllamaProvider,
    OpenAIProvider,
    OpenRouterProvider,
    resolve,
)

_ALL_KEY_ENVS = ["ANTHROPIC_API_KEY", "OPENAI_API_KEY", "OPENROUTER_API_KEY", "OLLAMA_API_KEY"]


@pytest.fixture(autouse=True)
def _clear_key_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Ensure no ambient provider key env vars leak into routing tests."""
    for name in _ALL_KEY_ENVS:
        monkeypatch.delenv(name, raising=False)


class TestPrefixRouting:
    def test_anthropic_prefix(self) -> None:
        provider, model_id = resolve("anthropic/claude-opus-4-8")
        assert isinstance(provider, AnthropicProvider)
        assert model_id == "claude-opus-4-8"

    def test_openrouter_prefix_keeps_remainder(self) -> None:
        provider, model_id = resolve("openrouter/anthropic/claude-sonnet-5")
        assert isinstance(provider, OpenRouterProvider)
        assert model_id == "anthropic/claude-sonnet-5"

    def test_openai_prefix(self) -> None:
        provider, model_id = resolve("openai/gpt-4o")
        assert isinstance(provider, OpenAIProvider)
        assert model_id == "gpt-4o"

    def test_ollama_prefix(self) -> None:
        provider, model_id = resolve("ollama/llama3.2")
        assert isinstance(provider, OllamaProvider)
        assert model_id == "llama3.2"

    def test_bare_prefix_uses_provider_default_model(self) -> None:
        # ``anthropic/`` with no model id falls back to the provider default.
        provider, model_id = resolve("anthropic/")
        assert isinstance(provider, AnthropicProvider)
        assert model_id == provider.default_model

    def test_ollama_host_secret_sets_base_url(self) -> None:
        secrets = {"llm": {"ollama_host": "http://remote:11434/v1"}}
        provider, _ = resolve("ollama/llama3.2", secrets=secrets)
        assert provider.base_url == "http://remote:11434/v1"


class TestDefaultPrecedence:
    def test_no_keys_defaults_to_ollama(self) -> None:
        provider, model_id = resolve("some-model")
        assert isinstance(provider, OllamaProvider)
        assert model_id == "some-model"

    def test_openrouter_wins_over_anthropic(self) -> None:
        secrets = {"llm": {"openrouter_api_key": "or", "anthropic_api_key": "an"}}
        provider, _ = resolve("m", secrets=secrets)
        assert isinstance(provider, OpenRouterProvider)

    def test_anthropic_when_only_anthropic_key(self) -> None:
        secrets = {"llm": {"anthropic_api_key": "an"}}
        provider, _ = resolve("m", secrets=secrets)
        assert isinstance(provider, AnthropicProvider)

    def test_openai_when_only_openai_key(self) -> None:
        secrets = {"llm": {"openai_api_key": "oa"}}
        provider, _ = resolve("m", secrets=secrets)
        assert isinstance(provider, OpenAIProvider)

    def test_env_var_key_selects_provider(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("OPENAI_API_KEY", "oa")
        provider, _ = resolve("m")
        assert isinstance(provider, OpenAIProvider)

    def test_empty_model_uses_default_provider_default_model(self) -> None:
        provider, model_id = resolve("")
        assert isinstance(provider, OllamaProvider)
        assert model_id == provider.default_model
