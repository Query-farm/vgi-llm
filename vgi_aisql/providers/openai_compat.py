# Copyright 2026 Query Farm LLC - https://query.farm

"""OpenAI-compatible chat providers: OpenAI, OpenRouter, and Ollama.

All three speak the OpenAI Chat Completions API, so they share one adapter that
differs only by ``default_base_url``, key handling, and default model. The
``openai`` SDK client is injectable for offline tests.
"""

from __future__ import annotations

import base64
from collections.abc import Sequence
from typing import Any

from vgi_aisql.providers.base import (
    BaseProvider,
    Completion,
    CompletionParams,
    ImagePart,
    Message,
    ProviderError,
    ResponseFormat,
    Usage,
)


def _content(content: str | Sequence[str | ImagePart]) -> Any:
    """Render neutral content into OpenAI chat message content."""
    if isinstance(content, str):
        return content
    parts: list[dict[str, Any]] = []
    for part in content:
        if isinstance(part, ImagePart):
            b64 = base64.standard_b64encode(part.data).decode("ascii")
            parts.append({"type": "image_url", "image_url": {"url": f"data:{part.media_type};base64,{b64}"}})
        else:
            parts.append({"type": "text", "text": part})
    return parts


class OpenAICompatProvider(BaseProvider):
    """Shared adapter for OpenAI Chat Completions-compatible backends."""

    #: Extra HTTP headers merged into every request (OpenRouter attribution).
    extra_headers: dict[str, str] = {}

    @property
    def client(self) -> Any:
        """Return the (lazily created) OpenAI-SDK client."""
        if self._client is None:
            import openai

            key = self.api_key or ("nokey" if not self.requires_key else self.require_key())
            self._client = openai.OpenAI(
                api_key=key,
                base_url=self.base_url,
                timeout=self.timeout,
                default_headers=self.extra_headers or None,
            )
        return self._client

    def complete(
        self,
        messages: Sequence[Message],
        *,
        model: str,
        params: CompletionParams,
        response_format: ResponseFormat | None = None,
    ) -> Completion:
        """Generate a single completion for ``messages``."""
        turns = [{"role": m.role, "content": _content(m.content)} for m in messages]
        kwargs: dict[str, Any] = {
            "model": model or self.default_model,
            "messages": turns,
            "max_tokens": params.max_tokens,
        }
        if params.temperature is not None:
            kwargs["temperature"] = params.temperature
        if params.top_p is not None:
            kwargs["top_p"] = params.top_p
        if params.stop:
            kwargs["stop"] = list(params.stop)
        if response_format is not None:
            kwargs["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "name": response_format.name,
                    "schema": response_format.json_schema,
                    "strict": True,
                },
            }

        try:
            resp = self.client.chat.completions.create(**kwargs)
        except Exception as exc:  # noqa: BLE001 — normalize every SDK error
            raise ProviderError(f"{self.name} request failed: {exc}") from exc

        choice = resp.choices[0]
        usage = Usage(
            input_tokens=getattr(resp.usage, "prompt_tokens", None) if resp.usage else None,
            output_tokens=getattr(resp.usage, "completion_tokens", None) if resp.usage else None,
        )
        return Completion(
            text=choice.message.content or "",
            model=getattr(resp, "model", model),
            usage=usage,
            finish_reason=getattr(choice, "finish_reason", None),
        )


class OpenAIProvider(OpenAICompatProvider):
    """Direct OpenAI API."""

    name = "openai"
    requires_key = True
    supports_images = True
    default_model = "gpt-4o"
    default_base_url = ""  # SDK default (api.openai.com)


class OpenRouterProvider(OpenAICompatProvider):
    """OpenRouter — one key, hundreds of models behind a provider-prefixed id."""

    name = "openrouter"
    requires_key = True
    supports_images = True
    default_model = "anthropic/claude-sonnet-5"
    default_base_url = "https://openrouter.ai/api/v1"
    extra_headers = {
        "HTTP-Referer": "https://github.com/Query-farm/vgi-aisql",
        "X-Title": "vgi-aisql",
    }


class OllamaProvider(OpenAICompatProvider):
    """Local Ollama daemon via its OpenAI-compatible endpoint. Keyless."""

    name = "ollama"
    requires_key = False
    supports_images = True
    default_model = "llama3.2"
    default_base_url = "http://localhost:11434/v1"
