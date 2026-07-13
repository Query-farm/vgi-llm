"""Local embedding + similarity scalars.

Two tiers: pure logic (similarity math, NULL masking, token estimate, warm-up)
always runs; the model-backed embedding assertions are gated on the default
fastembed model being loadable. The model assertions are structural / relative
(exact length, self-similarity ~ 1.0, related > unrelated) -- never exact floats.
"""

from __future__ import annotations

import math

import pyarrow as pa
import pytest

from tests.harness import model_available
from vgi_aisql import models
from vgi_aisql.scalars import AiEmbed, AiEmbedModel, AiSimilarity

needs_model = pytest.mark.skipif(
    not model_available(), reason="fastembed default model not available (offline / cold cache)"
)

_DIM = 384
_LT = pa.list_(pa.float32())


class TestWarmUp:
    def test_warm_up_is_idempotent_and_never_raises(self) -> None:
        models.warm_up()
        models.warm_up()


class TestSimilarityMath:
    def test_identical_vectors_score_one(self) -> None:
        v = pa.array([[1.0, 2.0, 3.0]], type=_LT)
        assert math.isclose(AiSimilarity.compute(v, v).to_pylist()[0], 1.0, abs_tol=1e-6)

    def test_orthogonal_scores_zero(self) -> None:
        a = pa.array([[1.0, 0.0]], type=_LT)
        b = pa.array([[0.0, 1.0]], type=_LT)
        assert math.isclose(AiSimilarity.compute(a, b).to_pylist()[0], 0.0, abs_tol=1e-6)

    def test_null_mismatch_zero_yield_null(self) -> None:
        a = pa.array([None, [1.0, 2.0], [0.0, 0.0]], type=_LT)
        b = pa.array([[1.0, 2.0], [1.0, 2.0, 3.0], [1.0, 2.0]], type=_LT)
        assert AiSimilarity.compute(a, b).to_pylist() == [None, None, None]


class TestEmbedNullMasking:
    def test_all_null_empty_without_loading_model(self) -> None:
        # Every row is NULL/empty -> all NULL, and the model is never invoked.
        assert AiEmbed.compute(pa.array([None, "", "   "])).to_pylist() == [None, None, None]


@needs_model
class TestEmbed:
    def test_fixed_length_vector(self) -> None:
        out = AiEmbed.compute(pa.array(["hello"])).to_pylist()
        assert len(out[0]) == _DIM
        assert all(isinstance(x, float) for x in out[0])

    def test_self_similarity_is_one(self) -> None:
        v = AiEmbed.compute(pa.array(["a quick brown fox"]))
        assert math.isclose(AiSimilarity.compute(v, v).to_pylist()[0], 1.0, abs_tol=1e-5)

    def test_related_beats_unrelated(self) -> None:
        vecs = AiEmbed.compute(pa.array(["dog", "puppy", "airplane"]))
        dog, puppy, airplane = vecs[0], vecs[1], vecs[2]
        related = AiSimilarity.compute(pa.array([dog], type=_LT), pa.array([puppy], type=_LT)).to_pylist()[0]
        unrelated = AiSimilarity.compute(pa.array([dog], type=_LT), pa.array([airplane], type=_LT)).to_pylist()[0]
        assert related > unrelated

    def test_null_rows_interleave(self) -> None:
        out = AiEmbed.compute(pa.array(["hello", None, "", "world"])).to_pylist()
        assert len(out[0]) == _DIM
        assert out[1] is None and out[2] is None
        assert len(out[3]) == _DIM

    def test_explicit_model_overload(self) -> None:
        out = AiEmbedModel.compute(pa.array(["hello"]), "BAAI/bge-small-en-v1.5").to_pylist()
        assert len(out[0]) == _DIM

    def test_unknown_model_raises(self) -> None:
        with pytest.raises(models.ModelNotAvailableError, match="Unknown embedding model"):
            AiEmbedModel.compute(pa.array(["hi"]), "no/such-model").to_pylist()
