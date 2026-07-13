"""Engine helpers: JSON/boolean parsing, batched completion, dedup, settings."""

from __future__ import annotations

import pytest

from tests.fake_provider import FakeProvider, install
from vgi_aisql import engine
from vgi_aisql.providers import CompletionParams, Message, ProviderError


class TestParseJsonObject:
    def test_plain_object(self) -> None:
        assert engine.parse_json_object('{"a": 1}') == {"a": 1}

    def test_fenced_block(self) -> None:
        assert engine.parse_json_object('```json\n{"a": 1}\n```') == {"a": 1}

    def test_object_embedded_in_prose(self) -> None:
        assert engine.parse_json_object('Sure! {"labels": ["x"]} done.') == {"labels": ["x"]}

    def test_non_object_and_garbage_and_none(self) -> None:
        assert engine.parse_json_object("[1, 2, 3]") is None
        assert engine.parse_json_object("not json") is None
        assert engine.parse_json_object(None) is None


class TestCoerceBool:
    @pytest.mark.parametrize(
        ("text", "expected"),
        [
            ("true", True),
            ("Yes, definitely", True),
            ("false", False),
            ("No.", False),
            # A leading digit must not flip the answer (the inversion bug).
            ("Item 0: yes", True),
            ("Row 1: false", False),
            # Bare digits / single letters are no longer treated as yes/no.
            ("1", None),
            ("0", None),
            ("maybe so", None),
            (None, None),
        ],
    )
    def test_coercions(self, text: str | None, expected: bool | None) -> None:
        assert engine.coerce_bool(text) is expected


class TestBuildSecrets:
    def test_unified_and_extra(self) -> None:
        out = engine.build_secrets({"anthropic_api_key": "k"}, {"openai": {"api_key": "o"}})
        assert out == {"aisql": {"anthropic_api_key": "k"}, "openai": {"api_key": "o"}}

    def test_empty(self) -> None:
        assert engine.build_secrets(None) == {}


class _Scalar:
    """Minimal ``pa.Scalar``-like wrapper for settings tests."""

    def __init__(self, value: object) -> None:
        self._value = value

    def as_py(self) -> object:
        return self._value


class TestRuntimeSettings:
    def test_defaults_send_nothing(self) -> None:
        params = engine.RuntimeSettings().completion_params()
        assert params.temperature is None
        assert params.top_p is None

    def test_read_settings_maps_scalars(self) -> None:
        s = engine.read_settings(
            aisql_max_tokens=_Scalar(8192),
            aisql_temperature=_Scalar(0.5),
            aisql_top_p=_Scalar(0.9),
            aisql_model=_Scalar("ollama/llama3.2"),
            aisql_max_workers=_Scalar(4),
            aisql_timeout=_Scalar(30.0),
        )
        params = s.completion_params()
        assert params.max_tokens == 8192
        assert params.temperature == 0.5
        assert params.top_p == 0.9
        assert s.effective_model("") == "ollama/llama3.2"
        assert s.effective_model("anthropic/claude-opus-4-8") == "anthropic/claude-opus-4-8"
        assert s.workers() == 4
        assert s.timeout_value() == 30.0

    def test_effective_model_and_workers_default(self) -> None:
        assert engine.RuntimeSettings().effective_model("") == ""
        assert engine.RuntimeSettings().workers() == engine.DEFAULT_MAX_WORKERS
        assert engine.RuntimeSettings().timeout_value() is None


class TestMapComplete:
    def test_masks_null_and_empty(self, monkeypatch: pytest.MonkeyPatch) -> None:
        install(monkeypatch, FakeProvider(lambda m, model, rf: "R"))
        out = engine.map_complete(
            [None, "", "  ", "hello"],
            build_messages=engine.user_message,
            model="",
        )
        assert [c.text if c else None for c in out] == [None, None, None, "R"]

    def test_provider_error_propagates(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # Errors THROW now (no silent NULL). The FakeProvider raises on a None reply.
        install(monkeypatch, FakeProvider(lambda m, model, rf: None))
        with pytest.raises(ProviderError):
            engine.map_complete(["boom"], build_messages=engine.user_message, model="")

    def test_unexpected_error_is_wrapped(self, monkeypatch: pytest.MonkeyPatch) -> None:
        def boom(messages: list[Message], model: str, rf: object) -> str:
            raise RuntimeError("kaboom")

        install(monkeypatch, FakeProvider(boom))
        with pytest.raises(ProviderError, match="LLM call failed"):
            engine.map_complete(["x"], build_messages=engine.user_message, model="")

    def test_preserves_order(self, monkeypatch: pytest.MonkeyPatch) -> None:
        install(monkeypatch, FakeProvider(lambda m, model, rf: _text(m).upper()))
        out = engine.map_complete(["a", "b", "c", "d"], build_messages=engine.user_message, model="", max_workers=2)
        assert [c.text if c else None for c in out] == ["A", "B", "C", "D"]

    def test_dedup_one_call_per_distinct_prompt(self, monkeypatch: pytest.MonkeyPatch) -> None:
        fake = FakeProvider(lambda m, model, rf: _text(m).upper())
        install(monkeypatch, fake)
        out = engine.map_complete(["x", "x", "y", "x", "y"], build_messages=engine.user_message, model="")
        # Every row is filled correctly...
        assert [c.text if c else None for c in out] == ["X", "X", "Y", "X", "Y"]
        # ...but the provider was called only once per DISTINCT prompt.
        assert len(fake.calls) == 2

    def test_timeout_threaded_to_resolve(self, monkeypatch: pytest.MonkeyPatch) -> None:
        seen: dict[str, object] = {}

        def resolver(model: str | None, *, secrets: object = None, client: object = None, timeout: object = None):
            seen["timeout"] = timeout
            return FakeProvider(lambda m, mo, rf: "ok"), model or "fake"

        monkeypatch.setattr(engine, "resolve_provider", resolver)
        engine.map_complete(
            ["hi"],
            build_messages=engine.user_message,
            model="",
            params=CompletionParams(temperature=0.3),
            timeout=12.5,
        )
        assert seen["timeout"] == 12.5


def _text(messages: list[Message]) -> str:
    user = next(m for m in messages if m.role == "user")
    assert isinstance(user.content, str)
    return user.content
