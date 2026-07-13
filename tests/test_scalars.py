"""Scalar functions with an injected FakeProvider (offline, deterministic)."""

from __future__ import annotations

import json
from collections.abc import Sequence

import pyarrow as pa
import pytest

from tests.fake_provider import FakeProvider, install
from tests.harness import model_available
from vgi_llm.providers import ImagePart, Message, ProviderError
from vgi_llm.scalars import (
    AiClassify,
    AiComplete,
    AiCompleteDetails,
    AiCompleteImage,
    AiCompleteModel,
    AiCountTokens,
    AiCountTokensModel,
    AiExtract,
    AiFilter,
    AiSentiment,
    AiSimilarityText,
    AiSummarize,
    Prompt,
)


def _user_text(messages: Sequence[Message]) -> str:
    user = next(m for m in messages if m.role == "user")
    if isinstance(user.content, str):
        return user.content
    return " ".join(p for p in user.content if isinstance(p, str))


# --- ai_complete ------------------------------------------------------------


class TestAiComplete:
    def test_returns_reply_and_masks(self, monkeypatch: pytest.MonkeyPatch) -> None:
        install(monkeypatch, FakeProvider(lambda m, model, rf: _user_text(m)[::-1]))
        out = AiComplete.compute(pa.array(["abc", None, "", "xyz"])).to_pylist()
        assert out == ["cba", None, None, "zyx"]

    def test_explicit_model_is_routed(self, monkeypatch: pytest.MonkeyPatch) -> None:
        fake = FakeProvider(lambda m, model, rf: model)
        install(monkeypatch, fake)
        out = AiCompleteModel.compute(pa.array(["hi"]), "ollama/llama3.2").to_pylist()
        assert out == ["ollama/llama3.2"]

    def test_provider_error_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # A provider failure now surfaces as an error, not a silent NULL.
        install(monkeypatch, FakeProvider(lambda m, model, rf: None))
        with pytest.raises(ProviderError):
            AiComplete.compute(pa.array(["hi"]))


# --- ai_complete_details ----------------------------------------------------


class TestAiCompleteDetails:
    def test_struct_envelope(self, monkeypatch: pytest.MonkeyPatch) -> None:
        install(monkeypatch, FakeProvider(lambda m, model, rf: "answer", default_model="fake-model"))
        row = AiCompleteDetails.compute(pa.array(["q"])).to_pylist()[0]
        assert row["text"] == "answer"
        assert row["model"] == "fake-model"
        assert row["input_tokens"] == 7
        assert row["output_tokens"] == 11
        assert row["finish_reason"] == "stop"

    def test_null_row(self, monkeypatch: pytest.MonkeyPatch) -> None:
        install(monkeypatch, FakeProvider(lambda m, model, rf: "x"))
        assert AiCompleteDetails.compute(pa.array([None])).to_pylist() == [None]


# --- ai_classify ------------------------------------------------------------


class TestAiClassify:
    def test_parses_and_filters_to_allowed(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # Model returns a bogus extra label; only allowed categories survive.
        install(monkeypatch, FakeProvider(lambda m, model, rf: '{"labels": ["billing", "spam"]}'))
        cats = pa.array([["billing", "bug", "feature"]], type=pa.list_(pa.string()))
        row = AiClassify.compute(pa.array(["my card was declined"]), cats).to_pylist()[0]
        assert row == {"labels": ["billing"]}

    def test_requests_response_format(self, monkeypatch: pytest.MonkeyPatch) -> None:
        fake = FakeProvider(lambda m, model, rf: '{"labels": []}')
        install(monkeypatch, fake)
        cats = pa.array([["a"]], type=pa.list_(pa.string()))
        AiClassify.compute(pa.array(["x"]), cats)
        assert fake.calls[0][2] is not None  # a ResponseFormat was passed

    def test_parse_failure_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # Non-JSON output is a real failure now -> raise (no silent all-NULL column).
        install(monkeypatch, FakeProvider(lambda m, model, rf: "not json"))
        cats = pa.array([["a"]], type=pa.list_(pa.string()))
        with pytest.raises(ProviderError, match="ai_classify"):
            AiClassify.compute(pa.array(["x"]), cats)

    def test_blank_input_is_null_not_error(self, monkeypatch: pytest.MonkeyPatch) -> None:
        install(monkeypatch, FakeProvider(lambda m, model, rf: "not json"))
        cats = pa.array([["a"]], type=pa.list_(pa.string()))
        # Blank input never calls the model -> NULL row, no error.
        assert AiClassify.compute(pa.array([None]), cats).to_pylist() == [None]


# --- ai_filter --------------------------------------------------------------


class TestAiFilter:
    def test_boolean_coercion(self, monkeypatch: pytest.MonkeyPatch) -> None:
        replies = {"a": "yes", "b": "no", "c": "banana"}
        install(monkeypatch, FakeProvider(lambda m, model, rf: replies[_user_text(m)]))
        out = AiFilter.compute("is it a question", pa.array(["a", "b", "c"])).to_pylist()
        assert out == [True, False, None]

    def test_empty_input_is_null(self, monkeypatch: pytest.MonkeyPatch) -> None:
        install(monkeypatch, FakeProvider(lambda m, model, rf: "true"))
        assert AiFilter.compute("p", pa.array([None, ""])).to_pylist() == [None, None]


# --- ai_extract -------------------------------------------------------------


class TestAiExtract:
    def test_returns_json_string(self, monkeypatch: pytest.MonkeyPatch) -> None:
        install(monkeypatch, FakeProvider(lambda m, model, rf: '{"age": 42}'))
        schema = '{"type":"object","properties":{"age":{"type":"integer"}}}'
        out = AiExtract.compute(pa.array(["Bob is 42"]), schema).to_pylist()
        assert json.loads(out[0]) == {"age": 42}

    def test_unparseable_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        install(monkeypatch, FakeProvider(lambda m, model, rf: "sorry"))
        with pytest.raises(ProviderError, match="ai_extract"):
            AiExtract.compute(pa.array(["x"]), "{}")


# --- ai_sentiment -----------------------------------------------------------


class TestAiSentiment:
    def test_parses_overall_and_categories(self, monkeypatch: pytest.MonkeyPatch) -> None:
        reply = json.dumps({"overall": "mixed", "categories": [{"name": "food", "sentiment": "positive"}]})
        install(monkeypatch, FakeProvider(lambda m, model, rf: reply))
        row = AiSentiment.compute(pa.array(["great food, slow service"])).to_pylist()[0]
        assert row["overall"] == "mixed"
        assert row["categories"] == [{"name": "food", "sentiment": "positive"}]

    def test_missing_overall_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        install(monkeypatch, FakeProvider(lambda m, model, rf: '{"categories": []}'))
        with pytest.raises(ProviderError, match="ai_sentiment"):
            AiSentiment.compute(pa.array(["x"]))


# --- ai_summarize -----------------------------------------------------------


class TestAiSummarize:
    def test_summary(self, monkeypatch: pytest.MonkeyPatch) -> None:
        install(monkeypatch, FakeProvider(lambda m, model, rf: "short"))
        assert AiSummarize.compute(pa.array(["long text", None])).to_pylist() == ["short", None]


# --- ai_complete_image ------------------------------------------------------

_PNG = b"\x89PNG\r\n\x1a\n" + b"\x00" * 16


class TestAiCompleteImage:
    def test_image_part_is_sent(self, monkeypatch: pytest.MonkeyPatch) -> None:
        seen: list[bool] = []

        def responder(messages: list[Message], model: str, rf: object) -> str:
            user = next(m for m in messages if m.role == "user")
            has_image = not isinstance(user.content, str) and any(isinstance(p, ImagePart) for p in user.content)
            seen.append(has_image)
            return "a cat"

        install(monkeypatch, FakeProvider(responder))
        out = AiCompleteImage.compute(pa.array(["describe"]), pa.array([_PNG], type=pa.binary())).to_pylist()
        assert out == ["a cat"]
        assert seen == [True]

    def test_null_image_is_null(self, monkeypatch: pytest.MonkeyPatch) -> None:
        install(monkeypatch, FakeProvider(lambda m, model, rf: "x"))
        out = AiCompleteImage.compute(
            pa.array(["describe", "describe"]),
            pa.array([None, _PNG], type=pa.binary()),
        ).to_pylist()
        assert out[0] is None
        assert out[1] == "x"


# --- prompt (pure) ----------------------------------------------------------


class TestPrompt:
    def test_sequential_substitution(self) -> None:
        out = Prompt.compute(pa.array(["Hi {} from {}"]), [pa.array(["Sam"]), pa.array(["NYC"])]).to_pylist()
        assert out == ["Hi Sam from NYC"]

    def test_explicit_index_and_escapes(self) -> None:
        out = Prompt.compute(
            pa.array(["{1} then {0}", "literal {{brace}}"]),
            [pa.array(["a", "x"]), pa.array(["b", "y"])],
        ).to_pylist()
        assert out == ["b then a", "literal {brace}"]

    def test_null_template_and_out_of_range(self) -> None:
        out = Prompt.compute(
            pa.array([None, "need {} and {}"]),
            [pa.array(["only-one", "a"])],
        ).to_pylist()
        assert out == [None, None]

    def test_attribute_traversal_is_rejected_not_executed(self) -> None:
        # A str.format template like {0.__class__} would traverse attributes;
        # the safe substitutor rejects it -> NULL, and never evaluates it.
        out = Prompt.compute(pa.array(["{0.__class__}"]), [pa.array(["x"])]).to_pylist()
        assert out == [None]

    def test_format_spec_is_rejected(self) -> None:
        # A giant-width format spec ({:>9999999999}) must not allocate; rejected -> NULL.
        out = Prompt.compute(pa.array(["{:>9999999999}"]), [pa.array(["x"])]).to_pylist()
        assert out == [None]

    def test_index_access_is_rejected(self) -> None:
        out = Prompt.compute(pa.array(["{0[0]}"]), [pa.array(["abc"])]).to_pylist()
        assert out == [None]


# --- ai_count_tokens (tiktoken, local) --------------------------------------


class TestAiCountTokens:
    def test_counts_tokens_and_masks_nulls(self) -> None:
        text = "hello world, this is tokenization"
        out = AiCountTokens.compute(pa.array(["", None, text])).to_pylist()
        assert out[0] is None
        assert out[1] is None
        # A real tiktoken count (7), distinct from the old ~len/4 heuristic (8).
        assert out[2] == 7
        assert out[2] != len(text) // 4

    def test_model_selects_tokenizer(self) -> None:
        text = "hello world, this is tokenization"
        default = AiCountTokens.compute(pa.array([text])).to_pylist()[0]
        gpt4o = AiCountTokensModel.compute(pa.array([text]), "openai/gpt-4o").to_pylist()[0]
        gpt35 = AiCountTokensModel.compute(pa.array([text]), "openai/gpt-3.5-turbo").to_pylist()[0]
        assert default is not None and gpt4o is not None and gpt35 is not None
        # gpt-4o uses o200k_base (the default); gpt-3.5 uses cl100k_base -> may differ.
        assert gpt4o == default


# --- ai_similarity text form (keyless, local ONNX) --------------------------

needs_model = pytest.mark.skipif(
    not model_available(), reason="fastembed default model not available (offline / cold cache)"
)


@needs_model
class TestAiSimilarityText:
    def test_related_beats_unrelated(self) -> None:
        related = AiSimilarityText.compute(pa.array(["cat"]), pa.array(["kitten"])).to_pylist()[0]
        unrelated = AiSimilarityText.compute(pa.array(["cat"]), pa.array(["airplane"])).to_pylist()[0]
        assert related > unrelated

    def test_null_input_is_null(self) -> None:
        out = AiSimilarityText.compute(pa.array([None, "cat"]), pa.array(["dog", ""])).to_pylist()
        assert out == [None, None]
