"""Aggregate functions: update/combine/finalize, chunked reduce, NULL groups."""

from __future__ import annotations

import types

import pyarrow as pa
import pytest

from tests.fake_provider import FakeProvider, install
from vgi_llm import aggregates, engine
from vgi_llm.aggregates import AggState, AiAgg, AiSummarizeAgg, _chunk, _collapse
from vgi_llm.providers import ProviderError


def _params(task: str = "") -> types.SimpleNamespace:
    positional = (pa.scalar(task),) if task else ()
    return types.SimpleNamespace(args=types.SimpleNamespace(positional=positional))


class TestState:
    def test_collapse_bounds_item_count_losslessly(self) -> None:
        texts = [f"t{i}" for i in range(aggregates._MAX_BUFFER_ITEMS + 5)]
        collapsed = _collapse(texts)
        assert len(collapsed) == 1
        # Every original piece is still present (concatenation is lossless).
        for t in texts:
            assert t in collapsed[0]

    def test_collapse_noop_when_small(self) -> None:
        assert _collapse(["a", "b"]) == ["a", "b"]

    def test_chunk_splits_on_char_budget(self) -> None:
        big = "x" * 5000
        chunks = _chunk([big, big, big])
        assert len(chunks) == 3


class TestUpdateCombine:
    def test_update_buffers_per_group(self) -> None:
        states: dict[int, AggState] = {0: AggState(), 1: AggState()}
        AiAgg.update(states, pa.array([0, 0, 1], type=pa.int64()), pa.array(["a", "b", "c"]))
        assert states[0].texts == ["a", "b"]
        assert states[1].texts == ["c"]

    def test_update_skips_null_and_empty(self) -> None:
        states: dict[int, AggState] = {0: AggState()}
        AiAgg.update(states, pa.array([0, 0, 0], type=pa.int64()), pa.array(["x", None, "  "]))
        assert states[0].texts == ["x"]

    def test_combine_merges_buffers(self) -> None:
        merged = AiAgg.combine(AggState(texts=["b"]), AggState(texts=["a"]), _params())
        assert merged.texts == ["a", "b"]


class TestFinalize:
    def test_single_group_single_call(self, monkeypatch: pytest.MonkeyPatch) -> None:
        fake = FakeProvider(lambda m, model, rf: "SUMMARY")
        install(monkeypatch, fake)
        states = {0: AggState(texts=["one", "two"])}
        out = AiAgg.finalize(pa.array([0], type=pa.int64()), states, _params("list themes"))
        assert out.column("result").to_pylist() == ["SUMMARY"]
        assert len(fake.calls) == 1
        # The task travels into the system prompt.
        assert "list themes" in fake.calls[0][0][0].content

    def test_empty_group_is_null(self, monkeypatch: pytest.MonkeyPatch) -> None:
        install(monkeypatch, FakeProvider(lambda m, model, rf: "x"))
        states: dict[int, AggState | None] = {0: None, 1: AggState(texts=[])}
        out = AiAgg.finalize(pa.array([0, 1], type=pa.int64()), states, _params("t"))
        assert out.column("result").to_pylist() == [None, None]

    def test_chunked_map_reduce_makes_multiple_calls(self, monkeypatch: pytest.MonkeyPatch) -> None:
        fake = FakeProvider(lambda m, model, rf: "R")
        install(monkeypatch, fake)
        big = "x" * 5000
        states = {0: AggState(texts=[big, big, big])}
        out = AiAgg.finalize(pa.array([0], type=pa.int64()), states, _params("reduce"))
        assert out.column("result").to_pylist() == ["R"]
        # 3 chunks mapped to partials + 1 reduce over the partials.
        assert len(fake.calls) == 4

    def test_provider_failure_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # A provider failure during the reduce surfaces as an error, not NULL.
        install(monkeypatch, FakeProvider(lambda m, model, rf: None))
        states = {0: AggState(texts=["a"])}
        with pytest.raises(ProviderError):
            AiAgg.finalize(pa.array([0], type=pa.int64()), states, _params("t"))


class TestSummarizeAgg:
    def test_fixed_task_reduce(self, monkeypatch: pytest.MonkeyPatch) -> None:
        fake = FakeProvider(lambda m, model, rf: "GROUP SUMMARY")
        install(monkeypatch, fake)
        states = {0: AggState(texts=["note one", "note two"])}
        out = AiSummarizeAgg.finalize(pa.array([0], type=pa.int64()), states, _params())
        assert out.column("result").to_pylist() == ["GROUP SUMMARY"]
        assert "Summarize" in fake.calls[0][0][0].content


class TestSecretCapture:
    def test_finalize_resolves_via_captured_bind_secret(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # on_bind captures the resolved llm secret; finalize must resolve with it.
        captured: dict[str, object] = {}

        def resolver(
            model: str | None, *, secrets: object = None, client: object = None, timeout: object = None
        ) -> tuple[FakeProvider, str]:
            captured["secrets"] = secrets
            return FakeProvider(lambda m, mo, rf: "OK"), model or "fake"

        monkeypatch.setattr(engine, "resolve_provider", resolver)

        args = types.SimpleNamespace(positional=(pa.scalar("secret-capture-task"),))
        secrets_accessor = types.SimpleNamespace(_unscoped={"llm": {"anthropic_api_key": pa.scalar("sk-x")}})
        bind_params = types.SimpleNamespace(secrets=secrets_accessor, args=args, settings=None)
        AiAgg.on_bind(bind_params)  # type: ignore[arg-type]

        states = {0: AggState(texts=["hello"])}
        out = AiAgg.finalize(
            pa.array([0], type=pa.int64()),
            states,
            types.SimpleNamespace(args=args),  # type: ignore[arg-type]
        )
        assert out.column("result").to_pylist() == ["OK"]
        assert captured["secrets"] == {"llm": {"anthropic_api_key": "sk-x"}}
