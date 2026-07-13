# Copyright 2026 Query Farm LLC - https://query.farm

"""Provider registry and model routing.

A ``model`` string routes to a backend by its leading path segment:

* ``anthropic/claude-opus-4-8``            → Anthropic direct
* ``openrouter/anthropic/claude-sonnet-5`` → OpenRouter (remainder kept intact)
* ``openai/gpt-4o``                        → OpenAI direct
* ``ollama/llama3.2``                      → local Ollama
* ``claude-...`` (no known prefix)         → the default provider, chosen by
  which key is configured (precedence: openrouter → anthropic → openai →
  ollama), with the bare string used as the model id.

Keys resolve from a VGI ``aisql`` secret first, then a provider-named secret,
then a ``<PROVIDER>_API_KEY`` environment variable (local/CI fallback).
"""

from __future__ import annotations

import os
from typing import Any

from vgi_aisql.providers.anthropic_provider import AnthropicProvider
from vgi_aisql.providers.base import (
    BaseProvider,
    ChatProvider,
    Completion,
    CompletionParams,
    ImagePart,
    Message,
    MissingKeyError,
    ProviderError,
    ResponseFormat,
    Usage,
)
from vgi_aisql.providers.openai_compat import (
    OllamaProvider,
    OpenAIProvider,
    OpenRouterProvider,
)
from vgi_aisql.secrets import host_from_secrets, key_from_secrets

__all__ = [
    "AnthropicProvider",
    "BaseProvider",
    "ChatProvider",
    "Completion",
    "CompletionParams",
    "ImagePart",
    "Message",
    "MissingKeyError",
    "OllamaProvider",
    "OpenAIProvider",
    "OpenRouterProvider",
    "ProviderError",
    "ResponseFormat",
    "Usage",
    "available_providers",
    "build_provider",
    "resolve",
]

_REGISTRY: dict[str, type[BaseProvider]] = {
    cls.name: cls for cls in (AnthropicProvider, OpenRouterProvider, OpenAIProvider, OllamaProvider)
}

#: Adapter names usable as a ``model`` prefix.
ADAPTER_NAMES = frozenset(_REGISTRY)

#: Order in which a bare (unprefixed) model picks its provider by key presence.
_DEFAULT_ORDER = ("openrouter", "anthropic", "openai", "ollama")


def available_providers() -> list[str]:
    """Return the registered provider names."""
    return list(_REGISTRY)


def _resolve_key(name: str, secrets: dict[str, Any] | None) -> str | None:
    """Resolve a provider key from secrets, then a ``<NAME>_API_KEY`` env var."""
    return key_from_secrets(secrets, name) or os.environ.get(f"{name.upper()}_API_KEY")


def _resolve_base_url(name: str, secrets: dict[str, Any] | None) -> str | None:
    """Resolve an override base URL from secrets, then a ``<NAME>_HOST`` env var."""
    return host_from_secrets(secrets, name) or os.environ.get(f"{name.upper()}_HOST")


def build_provider(
    name: str,
    *,
    secrets: dict[str, Any] | None = None,
    client: Any | None = None,
    timeout: float | None = None,
) -> BaseProvider:
    """Instantiate the named provider with its resolved key.

    Args:
        name: A registered provider name.
        secrets: Resolved VGI secrets mapping (``params.secrets``).
        client: Injectable SDK client (tests).
        timeout: Per-request timeout (seconds); None keeps the provider default.

    Returns:
        The instantiated provider with its resolved key.

    Raises:
        ProviderError: If ``name`` is not a registered provider.
    """
    cls = _REGISTRY.get(name)
    if cls is None:
        raise ProviderError(f"unknown provider {name!r}; available: {sorted(_REGISTRY)}")
    extra: dict[str, Any] = {} if timeout is None else {"timeout": timeout}
    return cls(
        api_key=_resolve_key(name, secrets),
        base_url=_resolve_base_url(name, secrets),
        client=client,
        **extra,
    )


def _pick_default(secrets: dict[str, Any] | None) -> str:
    """Choose a default provider by which key is configured."""
    for name in _DEFAULT_ORDER:
        if name == "ollama" or _resolve_key(name, secrets):
            return name
    return "ollama"


def resolve(
    model: str | None,
    *,
    secrets: dict[str, Any] | None = None,
    client: Any | None = None,
    timeout: float | None = None,
) -> tuple[BaseProvider, str]:
    """Route a ``model`` string to a (provider, model_id) pair.

    Args:
        model: The user-supplied model string, or None/empty for the default.
        secrets: Resolved VGI secrets mapping.
        client: Injectable SDK client (tests).
        timeout: Per-request timeout (seconds); None keeps the provider default.

    Returns:
        The provider instance and the model id to send to it (empty string means
        "use the provider default").
    """
    spec = (model or "").strip()
    head, _, rest = spec.partition("/")
    if head in ADAPTER_NAMES:
        provider = build_provider(head, secrets=secrets, client=client, timeout=timeout)
        return provider, rest or provider.default_model
    name = _pick_default(secrets)
    provider = build_provider(name, secrets=secrets, client=client, timeout=timeout)
    return provider, spec or provider.default_model
