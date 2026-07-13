# Copyright 2026 Query Farm LLC - https://query.farm

"""Provider abstraction for chat/completion backends.

Defines the wire-neutral request/response dataclasses, the :class:`ChatProvider`
protocol, and :class:`BaseProvider` — the small shared surface (key handling,
an injectable SDK client) that concrete providers build on. Each adapter uses
the best official SDK for its API: the Anthropic adapter uses the ``anthropic``
SDK; the OpenAI, OpenRouter, and Ollama adapters use the ``openai`` SDK (they
are all OpenAI-compatible, differing only by ``base_url`` and key). Providers
never crash the worker: recoverable failures raise :class:`ProviderError` (or
:class:`MissingKeyError`), which the function layer turns into a clean DuckDB
error or a NULL row.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

DEFAULT_TIMEOUT = 60.0
DEFAULT_MAX_TOKENS = 4096


class ProviderError(RuntimeError):
    """A recoverable provider failure surfaced as a clean DuckDB error."""


class MissingKeyError(ProviderError):
    """The selected provider requires an API key that was not configured."""


@dataclass(frozen=True, slots=True)
class ImagePart:
    """An inline image supplied to a multimodal model."""

    data: bytes
    media_type: str = "image/png"


@dataclass(frozen=True, slots=True)
class Message:
    """A single chat message.

    ``content`` is either plain text or a sequence mixing ``str`` and
    :class:`ImagePart` for multimodal prompts.
    """

    role: str  # "system" | "user" | "assistant"
    content: str | Sequence[str | ImagePart]


@dataclass(frozen=True, slots=True)
class CompletionParams:
    """Model hyperparameters. Mirrors the AISQL ``options`` object."""

    temperature: float | None = None
    max_tokens: int = DEFAULT_MAX_TOKENS
    top_p: float | None = None
    stop: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class ResponseFormat:
    """Structured-output request: a JSON Schema the model must satisfy."""

    json_schema: dict[str, Any]
    name: str = "response"


@dataclass(slots=True)
class Usage:
    """Token accounting returned alongside a completion."""

    input_tokens: int | None = None
    output_tokens: int | None = None


@dataclass(slots=True)
class Completion:
    """A model completion plus metadata for the ``*_details`` envelope."""

    text: str
    model: str
    usage: Usage = field(default_factory=Usage)
    finish_reason: str | None = None
    raw: dict[str, Any] = field(default_factory=dict)


@runtime_checkable
class ChatProvider(Protocol):
    """Typing protocol every chat backend satisfies."""

    name: str
    requires_key: bool
    supports_images: bool
    default_model: str

    def complete(
        self,
        messages: Sequence[Message],
        *,
        model: str,
        params: CompletionParams,
        response_format: ResponseFormat | None = None,
    ) -> Completion:
        """Generate a single completion for ``messages``."""
        ...


class BaseProvider:
    """Shared surface for chat providers.

    Subclasses set the class attributes, resolve an SDK client (injectable for
    tests), and implement :meth:`complete`.
    """

    name: str = "base"
    requires_key: bool = True
    supports_images: bool = False
    default_model: str = ""
    default_base_url: str = ""

    def __init__(
        self,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        timeout: float = DEFAULT_TIMEOUT,
        client: Any | None = None,
    ) -> None:
        """Build a provider instance.

        Args:
            api_key: Resolved API key (from a VGI secret or env var), or None.
            base_url: Override for the provider's API base URL.
            timeout: Per-request timeout in seconds.
            client: Injectable SDK client for tests; the concrete provider
                constructs a real one lazily when omitted.
        """
        self.api_key = api_key
        self.base_url = (base_url or self.default_base_url).rstrip("/") or None
        self.timeout = timeout
        self._client = client

    def require_key(self) -> str:
        """Return the API key or raise :class:`MissingKeyError`."""
        if not self.api_key:
            raise MissingKeyError(
                f"provider '{self.name}' requires an API key; configure it via a VGI secret "
                f"(TYPE llm, field {self.name}_api_key) or the {self.name.upper()}_API_KEY env var"
            )
        return self.api_key

    def complete(
        self,
        messages: Sequence[Message],
        *,
        model: str,
        params: CompletionParams,
        response_format: ResponseFormat | None = None,
    ) -> Completion:
        """Generate a completion. Overridden by concrete providers."""
        raise NotImplementedError
