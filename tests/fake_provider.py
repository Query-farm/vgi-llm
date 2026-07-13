"""An offline fake provider + install helper for deterministic unit tests.

``FakeProvider.complete`` records every call and returns a canned reply produced
by a ``responder`` callable (so a test can shape the JSON a structured function
sees, or raise a per-row error). :func:`install` swaps the module-level
``engine.resolve_provider`` seam so every scalar/aggregate routes to the fake
without touching a real SDK or the network.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

import pytest

from vgi_aisql import engine
from vgi_aisql.providers import Completion, ImagePart, Message, ProviderError, Usage

Responder = Callable[[Sequence[Message], str, Any], str | None]


class FakeProvider:
    """A recording, deterministic stand-in for a real chat provider."""

    def __init__(self, responder: Responder | None = None, *, default_model: str = "fake-model") -> None:
        self.calls: list[tuple[Sequence[Message], str, Any]] = []
        self.default_model = default_model
        self._responder: Responder = responder or (lambda messages, model, rf: "FAKE_REPLY")

    @property
    def prompts(self) -> list[str]:
        """The user-message text of every recorded call."""
        out: list[str] = []
        for messages, _model, _rf in self.calls:
            user = next((m for m in messages if m.role == "user"), None)
            out.append(_text_of(user.content) if user is not None else "")
        return out

    def complete(
        self,
        messages: Sequence[Message],
        *,
        model: str,
        params: Any,
        response_format: Any = None,
    ) -> Completion:
        """Record the call and return the responder's canned completion."""
        self.calls.append((messages, model, response_format))
        text = self._responder(messages, model, response_format)
        if text is None:
            raise ProviderError("fake provider error")
        return Completion(
            text=text,
            model=model or self.default_model,
            usage=Usage(input_tokens=7, output_tokens=11),
            finish_reason="stop",
        )


def _text_of(content: str | Sequence[str | ImagePart]) -> str:
    """Flatten message content to its text part(s)."""
    if isinstance(content, str):
        return content
    return " ".join(p for p in content if isinstance(p, str))


def install(monkeypatch: pytest.MonkeyPatch, provider: FakeProvider) -> None:
    """Point ``engine.resolve_provider`` at ``provider`` for the test's duration."""

    def _resolver(
        model: str | None, *, secrets: Any = None, client: Any = None, timeout: Any = None
    ) -> tuple[FakeProvider, str]:
        return provider, (model or provider.default_model)

    monkeypatch.setattr(engine, "resolve_provider", _resolver)
