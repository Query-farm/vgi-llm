# Copyright 2026 Query Farm LLC - https://query.farm

"""Anthropic (Claude) chat provider, via the official ``anthropic`` SDK."""

from __future__ import annotations

import base64
from collections.abc import Sequence
from typing import Any

from vgi_llm.providers.base import (
    BaseProvider,
    Completion,
    CompletionParams,
    ImagePart,
    Message,
    ProviderError,
    ResponseFormat,
    Usage,
)

# Default to the most capable Opus tier; callers pick cheaper models per-call
# via the ``model`` argument for bulk workloads.
DEFAULT_MODEL = "claude-opus-4-8"


def _content_blocks(content: str | Sequence[str | ImagePart]) -> Any:
    """Render our neutral content into Anthropic message content."""
    if isinstance(content, str):
        return content
    blocks: list[dict[str, Any]] = []
    for part in content:
        if isinstance(part, ImagePart):
            blocks.append(
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": part.media_type,
                        "data": base64.standard_b64encode(part.data).decode("ascii"),
                    },
                }
            )
        else:
            blocks.append({"type": "text", "text": part})
    return blocks


class AnthropicProvider(BaseProvider):
    """Claude via ``anthropic.Anthropic``.

    System-role messages are hoisted to the top-level ``system`` parameter (the
    Anthropic Messages API keeps system separate from the turn list). Sampling
    parameters are forwarded only when explicitly set, because current Claude
    models reject ``temperature``/``top_p``.
    """

    name = "anthropic"
    requires_key = True
    supports_images = True
    default_model = DEFAULT_MODEL
    default_base_url = ""

    @property
    def client(self) -> Any:
        """Return the (lazily created) Anthropic client."""
        if self._client is None:
            import anthropic

            self._client = anthropic.Anthropic(api_key=self.require_key(), timeout=self.timeout)
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
        system_parts: list[str] = []
        turns: list[dict[str, Any]] = []
        for msg in messages:
            if msg.role == "system":
                # System content is text-only when hoisted.
                system_parts.append(msg.content if isinstance(msg.content, str) else "")
            else:
                turns.append({"role": msg.role, "content": _content_blocks(msg.content)})

        kwargs: dict[str, Any] = {
            "model": model or self.default_model,
            "max_tokens": params.max_tokens,
            "messages": turns,
        }
        if system_parts:
            kwargs["system"] = "\n\n".join(p for p in system_parts if p)
        if params.temperature is not None:
            kwargs["temperature"] = params.temperature
        if params.top_p is not None:
            kwargs["top_p"] = params.top_p
        if params.stop:
            kwargs["stop_sequences"] = list(params.stop)
        if response_format is not None:
            kwargs["output_config"] = {"format": {"type": "json_schema", "schema": response_format.json_schema}}

        try:
            resp = self.client.messages.create(**kwargs)
        except Exception as exc:  # noqa: BLE001 — normalize every SDK error
            raise ProviderError(f"anthropic request failed: {exc}") from exc

        text = "".join(block.text for block in resp.content if getattr(block, "type", None) == "text")
        usage = Usage(
            input_tokens=getattr(resp.usage, "input_tokens", None),
            output_tokens=getattr(resp.usage, "output_tokens", None),
        )
        return Completion(
            text=text,
            model=getattr(resp, "model", model),
            usage=usage,
            finish_reason=getattr(resp, "stop_reason", None),
        )
