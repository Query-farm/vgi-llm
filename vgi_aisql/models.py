# Copyright 2026 Query Farm LLC - https://query.farm

"""Local embedding-model lifecycle: load a fastembed ONNX model once, cache it.

VGI keeps the worker process alive across queries, so the expensive thing a
local-embedding worker does -- loading the ONNX model (and, on first use ever,
*downloading* it) -- happens once and is amortised over every row of every
query. This module centralises that caching: callers ask for "the embedder for
model X" and get a ready ``fastembed.TextEmbedding`` back.

Why fastembed
-------------
``fastembed`` (Qdrant, Apache-2.0) runs sentence-transformer models through ONNX
Runtime -- **no torch**. On first use it downloads a small, quantised ONNX model
to a local cache and reuses it forever after. That makes the worker light to
install and fast to start once the model is cached, and lets ``ai_embed`` /
``ai_similarity`` run **keyless** -- no LLM provider required.

Default model
-------------
``BAAI/bge-small-en-v1.5`` -- 384-dim, MIT licensed, strong general-purpose
English retrieval/semantic-search embeddings. Downloaded on first use to the
fastembed cache dir (``~/.cache/...`` by default, or ``VGI_AISQL_CACHE_DIR`` /
``FASTEMBED_CACHE_PATH`` -- see :func:`_cache_dir`). The cache is gitignored.

Everything here is lazy: importing this module is cheap; nothing is loaded or
downloaded until the first row needs it (or :func:`warm_up` is called at
startup). A model that cannot be loaded raises a clear, actionable error rather
than a deep library traceback.
"""

from __future__ import annotations

import contextlib
import math
import os
import threading
from functools import cache
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover - typing only
    from fastembed import TextEmbedding

# ---------------------------------------------------------------------------
# Supported models. Keyed by the name users pass to ai_embed(text, model).
# Each entry is the output dimension (FLOAT[] length) the model produces. All
# are fastembed-supported ONNX models with permissive licenses.
# ---------------------------------------------------------------------------

_SUPPORTED_MODELS: dict[str, int] = {
    "BAAI/bge-small-en-v1.5": 384,
    "BAAI/bge-base-en-v1.5": 768,
    "BAAI/bge-small-en": 384,
    "sentence-transformers/all-MiniLM-L6-v2": 384,
}

DEFAULT_MODEL = "BAAI/bge-small-en-v1.5"

_CACHE_DIR_ENV = "VGI_AISQL_CACHE_DIR"

_lock = threading.Lock()


class ModelNotAvailableError(RuntimeError):
    """A requested embedding model is unknown or could not be loaded/downloaded.

    Carries an actionable hint (the supported model list, or that the first use
    needs network access to download) so the DuckDB-side error tells the user
    how to fix it.
    """


def supported_models() -> list[tuple[str, int]]:
    """Every ``(model, dim)`` the worker can produce, sorted by model name.

    Returns:
        The ``(model, dim)`` pairs, sorted by model name.
    """
    return sorted(_SUPPORTED_MODELS.items())


def resolve_model(model: str | None) -> str:
    """Normalise a requested model name, defaulting empty/None to the default.

    Args:
        model: The requested model name, or None/empty for the default.

    Returns:
        A supported model name.

    Raises:
        ModelNotAvailableError: If ``model`` is not a supported model.
    """
    name = (model or "").strip() or DEFAULT_MODEL
    if name not in _SUPPORTED_MODELS:
        raise ModelNotAvailableError(
            f"Unknown embedding model {name!r}. Supported models: {', '.join(sorted(_SUPPORTED_MODELS))}."
        )
    return name


def embedding_dim(model: str | None) -> int:
    """Output dimension for ``model`` (defaulting empty/None to the default).

    Args:
        model: The model name, or None/empty for the default.

    Returns:
        The vector length the model produces.
    """
    return _SUPPORTED_MODELS[resolve_model(model)]


def _cache_dir() -> str | None:
    """Where fastembed should cache downloaded ONNX models.

    ``VGI_AISQL_CACHE_DIR`` wins; otherwise we honour fastembed's own
    ``FASTEMBED_CACHE_PATH``; otherwise ``None`` lets fastembed pick its default
    (a cache dir under the user's home). The dir is created on demand.

    Returns:
        The cache directory path, or None to let fastembed choose.
    """
    explicit = os.environ.get(_CACHE_DIR_ENV) or os.environ.get("FASTEMBED_CACHE_PATH")
    if explicit:
        os.makedirs(explicit, exist_ok=True)
        return explicit
    return None


@cache
def _load_model(model_name: str) -> TextEmbedding:
    """Load (and cache) a fastembed ``TextEmbedding`` by name.

    First-ever use downloads the quantised ONNX model to the fastembed cache;
    all later worker processes that share the cache load it from disk.

    Args:
        model_name: A supported fastembed model name.

    Returns:
        The loaded embedder.

    Raises:
        ModelNotAvailableError: If fastembed is missing or the model cannot be
            loaded (e.g. offline on a cold cache).
    """
    try:
        from fastembed import TextEmbedding
    except ImportError as exc:  # pragma: no cover - dependency present in prod
        raise ModelNotAvailableError("fastembed is not installed. Install it with: uv pip install fastembed") from exc

    try:
        return TextEmbedding(model_name=model_name, cache_dir=_cache_dir())
    except Exception as exc:  # noqa: BLE001 - turn any backend failure into an actionable error
        raise ModelNotAvailableError(
            f"Could not load embedding model {model_name!r}. The model is downloaded "
            f"on first use, so this needs network access on a cold cache; afterwards it "
            f"is served from the fastembed cache (override with {_CACHE_DIR_ENV}). "
            f"Original error: {exc}"
        ) from exc


def get_model(model: str | None) -> TextEmbedding:
    """Get the cached fastembed embedder for ``model`` (thread-safe first load).

    Args:
        model: The model name, or None/empty for the default.

    Returns:
        The cached embedder.
    """
    name = resolve_model(model)
    with _lock:
        return _load_model(name)


def embed_texts(texts: list[str], *, model: str | None) -> list[list[float]]:
    """Embed a list of (already non-empty) strings, returning one vector each.

    Order is preserved. The caller is responsible for masking out NULL/empty
    rows and re-inserting NULLs.

    Args:
        texts: Non-empty strings to embed.
        model: The model name, or None/empty for the default.

    Returns:
        One embedding vector (list of floats) per input string.
    """
    if not texts:
        return []
    embedder = get_model(model)
    # fastembed yields numpy arrays; convert to plain Python float lists for Arrow.
    return [vec.tolist() for vec in embedder.embed(texts)]


def cosine_similarity(a: list[float] | None, b: list[float] | None) -> float | None:
    """Cosine similarity of two vectors in [-1, 1]; None on NULL/empty/mismatch.

    Pure arithmetic -- never touches a model. Returns None (rather than raising)
    for NULL inputs, empty vectors, length mismatches, or a zero-magnitude
    vector, so it is robust to odd input straight out of SQL.

    Args:
        a: The first vector, or None.
        b: The second vector, or None.

    Returns:
        The cosine similarity, or None when undefined.
    """
    if a is None or b is None:
        return None
    if len(a) == 0 or len(b) == 0 or len(a) != len(b):
        return None
    dot = 0.0
    na = 0.0
    nb = 0.0
    for x, y in zip(a, b, strict=False):
        if x is None or y is None:
            return None
        dot += x * y
        na += x * x
        nb += y * y
    if na == 0.0 or nb == 0.0:
        return None
    return dot / (math.sqrt(na) * math.sqrt(nb))


def warm_up() -> None:
    """Load (and, if needed, download) the default model once at worker startup.

    Everything in this module is lazy by design, so the *first* query of every
    ATTACH otherwise pays the model load -- and on a cold cache the multi-second
    *download* -- inline. Warming here moves that one-time cost to process spawn
    (before any query). It only populates the existing cache -- it never changes
    any output. Best-effort: if the model can't be loaded (e.g. offline on a
    cold cache) it is not fatal here -- the function that needs it will raise its
    own actionable error if actually invoked, so a worker still starts cleanly.
    """
    with contextlib.suppress(Exception):
        embedder = _load_model(DEFAULT_MODEL)
        next(iter(embedder.embed(["warm up"])), None)
