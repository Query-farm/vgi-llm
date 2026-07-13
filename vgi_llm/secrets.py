# Copyright 2026 Query Farm LLC - https://query.farm

"""Extract provider API keys from VGI-resolved secrets.

The framework hands functions a mapping ``secret_type -> {field -> pa.Scalar}``.
A single ``llm`` secret type carries one field per provider
(``anthropic_api_key``, ``openrouter_api_key``, ``openai_api_key``,
``ollama_host``), so one secret configures every backend. Keys can also be
read from conventional single-field secrets named after the provider, matching
the vgi-search convention.
"""

from __future__ import annotations

from typing import Any

# Fields checked, in order, on a provider-named secret entry.
_KEY_FIELDS = ("api_key", "key", "token", "value", "secret", "secret_string")


def _as_py(scalar: Any) -> Any:
    """Unwrap a ``pa.Scalar`` (or plain value) to Python."""
    return scalar.as_py() if hasattr(scalar, "as_py") else scalar


def key_from_secrets(secrets: dict[str, dict[str, Any]] | None, provider: str) -> str | None:
    """Resolve the API key for ``provider`` from resolved VGI secrets.

    Looks first in a unified ``llm`` secret for a ``{provider}_api_key`` (or
    ``{provider}_host``) field, then falls back to a provider-named secret with
    any conventional key field.

    Args:
        secrets: The resolved-secrets mapping from ``params.secrets``, or None.
        provider: Provider name, e.g. ``"anthropic"``.

    Returns:
        The key string, or None when not configured.
    """
    if not secrets:
        return None

    unified = secrets.get("llm")
    if unified:
        for field_name in (f"{provider}_api_key", f"{provider}_key"):
            value = _as_py(unified.get(field_name))
            if value:
                return str(value)

    entry = secrets.get(provider)
    if entry:
        for field_name in _KEY_FIELDS:
            value = _as_py(entry.get(field_name))
            if value:
                return str(value)

    return None


def host_from_secrets(secrets: dict[str, dict[str, Any]] | None, provider: str) -> str | None:
    """Resolve an override base URL for ``provider`` (e.g. a non-default Ollama daemon).

    Reads a ``{provider}_host`` field from the unified ``llm`` secret. The value
    is the OpenAI-compatible base URL, including any ``/v1`` path segment.

    Args:
        secrets: The resolved-secrets mapping from ``params.secrets``, or None.
        provider: Provider name, e.g. ``"ollama"``.

    Returns:
        The base-URL string, or None when not configured.
    """
    if not secrets:
        return None
    unified = secrets.get("llm")
    if unified:
        value = _as_py(unified.get(f"{provider}_host"))
        if value:
            return str(value)
    return None
