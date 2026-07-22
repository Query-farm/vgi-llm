"""Aggregate functions: update/combine/finalize, chunked reduce, NULL groups."""

from __future__ import annotations

import types

import pyarrow as pa
import pytest
from vgi.table_function import SecretsAccessor

from tests.fake_provider import FakeProvider, install
from vgi_llm import aggregates, engine
from vgi_llm.aggregates import AggState, AiAgg, AiSummarizeAgg, _chunk, _collapse
from vgi_llm.providers import ProviderError


def _params(task: str = "") -> types.SimpleNamespace:
    positional = (pa.scalar(task),) if task else ()
    return types.SimpleNamespace(args=types.SimpleNamespace(positional=positional))


def _secrets_accessor(llm_fields: dict[str, str] | None) -> SecretsAccessor:
    """Build a real framework SecretsAccessor from a first-pass secrets batch.

    Mirrors what ``aggregate_bind`` constructs: ``is_retry`` is False (aggregates
    get no two-phase retry), and an unconfigured secret means *no column at all*.
    """
    if llm_fields is None:
        return SecretsAccessor(None)
    struct = pa.struct([(k, pa.string()) for k in llm_fields])
    return SecretsAccessor(pa.record_batch({"llm": pa.array([llm_fields], type=struct)}))


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

        # Use the real framework SecretsAccessor, built from a real secrets
        # RecordBatch exactly as the framework builds it at bind.
        secrets_accessor = _secrets_accessor({"anthropic_api_key": "sk-x"})

        args = types.SimpleNamespace(positional=(pa.scalar("secret-capture-task"),))
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
        # Reading a *present* secret must not request resolution either.
        assert secrets_accessor.needs_resolution is False

    @pytest.mark.parametrize("agg", [AiAgg, AiSummarizeAgg])
    def test_bind_without_secret_requests_no_resolution(self, agg: type) -> None:
        # No CREATE SECRET (TYPE llm, ...) is the normal case: keys come from
        # provider env vars, or Ollama runs keyless. on_bind must NOT register a
        # pending secret lookup -- aggregate_bind cannot do the two-phase retry
        # and raises NotImplementedError, which used to break every ai_agg call.
        secrets_accessor = _secrets_accessor(None)
        args = types.SimpleNamespace(positional=(pa.scalar("no-secret-task"),))
        bind_params = types.SimpleNamespace(secrets=secrets_accessor, args=args, settings=None)

        agg.on_bind(bind_params)

        assert secrets_accessor.needs_resolution is False
        assert secrets_accessor.pending_lookups == []
        # ...and finalize falls back to env vars / keyless (no captured secret).
        assert aggregates._BIND_CONFIG[aggregates._bind_key(args)][0] is None
