# Copyright 2026 Query Farm LLC - https://query.farm

"""AISQL aggregate functions: LLM map-reduce over a whole group.

``ai_agg(input, task)`` and ``ai_summarize_agg(input)`` fold a column of text
across a ``GROUP BY`` into a single LLM answer. They are **hierarchical chunked
map-reduce** so a group larger than the model's context window still reduces:

- ``update``   buffers each group's row texts into serializable state, collapsing
  an over-long buffer into a single running string (lossless concatenation, no
  model call) to bound item count.
- ``combine``  merges two partial buffers (parallel workers).
- ``finalize`` chunks the buffer by a character budget, maps each chunk to a
  partial answer with the LLM, then reduces the partials with the ``task`` --
  recursing until a single answer remains.

The provider is resolved through :func:`vgi_llm.engine.resolve_provider` (the
test seam). Aggregates honor ``CREATE SECRET (TYPE llm, ...)``: the resolved
key and the ``llm_*`` settings are captured in ``on_bind`` (the only aggregate
phase with secrets/settings in scope) and reused in ``finalize``, falling back to
provider env vars (``ANTHROPIC_API_KEY`` etc.) or keyless Ollama. Consistent with
the rest of the surface, a provider/runtime failure is **raised** as a DuckDB
error, not swallowed to NULL; an empty group still yields NULL.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Annotated, Any

import pyarrow as pa
from vgi.aggregate_function import AggregateBindParams, AggregateFunction
from vgi.arguments import ConstParam, Param, Returns
from vgi.invocation import BindResponse
from vgi.metadata import FunctionExample, NullHandling
from vgi.schema_utils import schema
from vgi.table_function import ProcessParams
from vgi_rpc import ArrowSerializableDataclass, ArrowType

from vgi_llm import engine, meta

#: Character budget per map chunk before a group is split for reduction.
_CHUNK_CHARS = 6000
#: Item-count cap on a group's buffer before update/combine collapse it.
_MAX_BUFFER_ITEMS = 64
#: Fixed task used by ``ai_summarize_agg``.
_SUMMARIZE_TASK = "Summarize the following texts into a single, coherent summary."

#: Bind-scoped provider config (resolved ``llm`` secret + ``llm_*`` settings)
#: captured in ``on_bind`` -- the only aggregate phase with real secrets/settings
#: in scope -- and looked back up in ``finalize`` by the const-args key. Provider
#: env vars remain the fallback when no ``llm`` secret is configured.
_BIND_CONFIG: dict[bytes, tuple[dict[str, Any] | None, engine.RuntimeSettings]] = {}


def _bind_key(args: Any) -> bytes:
    """Build a stable key from an aggregate's const arguments (task/model).

    Both ``on_bind`` (via ``request.arguments``) and ``finalize`` (via
    ``params.args``) see the same const args, so this keys the captured provider
    config so finalize can retrieve it.

    Args:
        args: The bound :class:`Arguments`, or None.

    Returns:
        A bytes key derived from the positional const-arg values.
    """
    positional = getattr(args, "positional", None)
    if not positional:
        return b""
    parts: list[str] = []
    for scalar in positional:
        value = scalar.as_py() if scalar is not None and hasattr(scalar, "as_py") else scalar
        parts.append("" if value is None else str(value))
    return "\x1f".join(parts).encode()


def _settings_from_bind(settings: dict[str, Any] | None) -> engine.RuntimeSettings:
    """Read the ``llm_*`` settings scalars from an aggregate bind's settings dict.

    Args:
        settings: The bind's ``{setting_name: scalar}`` mapping, or None.

    Returns:
        The resolved runtime settings.
    """
    get = settings.get if settings else (lambda _k: None)
    return engine.read_settings(
        llm_max_tokens=get("llm_max_tokens"),
        llm_temperature=get("llm_temperature"),
        llm_top_p=get("llm_top_p"),
        llm_model=get("llm_model"),
        llm_max_workers=get("llm_max_workers"),
        llm_timeout=get("llm_timeout"),
    )


@dataclass(kw_only=True)
class AggState(ArrowSerializableDataclass):
    """Per-group accumulation state: a serializable buffer of row texts."""

    texts: Annotated[list[str], ArrowType(pa.list_(pa.string()))] = field(default_factory=list)


def _collapse(texts: list[str]) -> list[str]:
    """Bound a buffer's item count by joining it into one running string.

    Concatenation is lossless (it keeps every character), so this bounds memory
    churn and serialization size without dropping content or calling a model.

    Args:
        texts: The current buffer.

    Returns:
        The buffer, collapsed to a single joined string when it grew too long.
    """
    if len(texts) <= _MAX_BUFFER_ITEMS:
        return texts
    return ["\n\n".join(texts)]


def _chunk(texts: list[str]) -> list[str]:
    """Group consecutive texts into chunks under the character budget.

    Args:
        texts: The buffered texts.

    Returns:
        A list of chunk strings, each roughly within ``_CHUNK_CHARS``.
    """
    chunks: list[str] = []
    current: list[str] = []
    size = 0
    for t in texts:
        piece = t or ""
        if current and size + len(piece) > _CHUNK_CHARS:
            chunks.append("\n\n".join(current))
            current, size = [], 0
        current.append(piece)
        size += len(piece)
    if current:
        chunks.append("\n\n".join(current))
    return chunks


def _map_reduce(
    texts: list[str],
    task: str,
    *,
    secrets: dict[str, Any] | None,
    settings: engine.RuntimeSettings,
) -> str | None:
    """Reduce a group's texts to a single answer via chunked LLM map-reduce.

    An empty group (no live texts) returns None. A provider-resolution or
    per-call failure is **raised** (surfacing as a DuckDB error) -- it is never
    swallowed to NULL.

    Args:
        texts: The group's buffered texts.
        task: The instruction describing what to produce.
        secrets: Resolved provider secrets (from a captured ``llm`` secret), or
            None to fall back to provider env vars.
        settings: The resolved ``llm_*`` settings (model / sampling / timeout).

    Returns:
        The reduced answer, or None for an empty group.

    Raises:
        ProviderError: On provider resolution or a per-call failure.
    """
    live = [t for t in texts if t and t.strip()]
    if not live:
        return None

    provider, model_id = engine.resolve_provider(
        settings.effective_model(""),
        secrets=secrets,
        timeout=settings.timeout_value(),
    )
    call_params = settings.completion_params()

    def _one(system: str, body: str) -> str:
        completion = provider.complete(
            engine.system_user_messages(system, body),
            model=model_id,
            params=call_params,
        )
        text: str = completion.text
        return text

    system = f"Apply this task to the text and return only the result. Task: {task}"

    # Reduce until a single chunk remains.
    chunks = _chunk(live)
    guard = 0
    while len(chunks) > 1 and guard < 8:
        guard += 1
        chunks = _chunk([_one(system, ch) for ch in chunks])

    return _one(system, chunks[0])


class _AiAggBase(AggregateFunction[AggState]):
    """Shared update/combine/finalize plumbing for the AISQL text aggregates."""

    @classmethod
    def on_bind(cls, params: AggregateBindParams, **kwargs: Any) -> BindResponse:
        """Capture the resolved ``llm`` secret + ``llm_*`` settings at bind.

        Aggregates only have real secrets/settings in scope at bind, so we stash
        them (keyed by the const args) for ``finalize`` to reuse. The already-
        resolved unscoped secret is read directly -- without registering a
        two-phase lookup, which aggregates do not support.

        Args:
            params: The aggregate bind parameters.
            **kwargs: Forwarded to the base ``on_bind``.

        Returns:
            The bind response with the single ``result`` output column.
        """
        secret: dict[str, Any] | None = None
        unscoped = getattr(params.secrets, "_unscoped", {})
        entry = unscoped.get("llm")
        if entry:
            secret = {k: (v.as_py() if hasattr(v, "as_py") else v) for k, v in entry.items()}
        _BIND_CONFIG[_bind_key(params.args)] = (secret, _settings_from_bind(params.settings))
        return BindResponse(output_schema=schema(result=pa.string()))

    @classmethod
    def initial_state(cls, params: ProcessParams[Any]) -> AggState:
        """Return a fresh, empty per-group buffer."""
        return AggState()

    @classmethod
    def combine(cls, source: AggState, target: AggState, params: ProcessParams[Any]) -> AggState:
        """Merge two partial buffers for the same group."""
        return AggState(texts=_collapse(target.texts + source.texts))

    @classmethod
    def _finalize_with_task(
        cls,
        group_ids: pa.Int64Array,
        states: dict[int, AggState | None],
        task: str,
        params: ProcessParams[Any],
    ) -> pa.RecordBatch:
        """Reduce each group's buffer to one answer row.

        Args:
            group_ids: The requested group ids, in output order.
            states: Per-group state (None for a group that saw no rows).
            task: The instruction to apply during reduction.
            params: The finalize process params (used to look up the bind config).

        Returns:
            A one-column ``result`` RecordBatch, one row per group id.
        """
        secret, settings = _BIND_CONFIG.get(_bind_key(params.args), (None, engine.RuntimeSettings()))
        secrets = engine.build_secrets(secret) if secret else None
        results: list[str | None] = []
        for gid in group_ids:
            state = states[gid.as_py()]
            if state is None or not state.texts:
                results.append(None)
            else:
                results.append(_map_reduce(state.texts, task, secrets=secrets, settings=settings))
        return pa.record_batch({"result": pa.array(results, type=pa.string())})


_AI_AGG_TAGS = meta.object_tags(
    title="AI Aggregate Over A Group",
    doc_llm=(
        "## ai_agg(input, task)\n\n"
        "Aggregate a whole column of text within each `GROUP BY` group by applying "
        "a natural-language `task` to all the rows at once, returning one `VARCHAR` "
        "answer per group. It is a hierarchical **chunked map-reduce**: rows are "
        "buffered, split into context-sized chunks, each chunk mapped to a partial "
        "answer, then the partials reduced with your task -- so a group larger than "
        "the model's context window still produces a single answer.\n\n"
        "**When to use.** One summary, theme, or answer per category over all its "
        "rows -- grouping a text column and applying a task such as listing the top "
        "complaints to each group. For a per-row answer use `ai_complete`; for a "
        "plain summary-per-group use `ai_summarize_agg`. See the attached example "
        "queries.\n\n"
        "**Input/output.** Inputs: a VARCHAR column and a constant `task` string. "
        "Output: one VARCHAR per group. A group with no rows yields NULL; a "
        "provider failure is raised as a DuckDB error. Keys come from a CREATE "
        "SECRET (TYPE llm) or provider env vars (or keyless Ollama)."
    ),
    doc_md=(
        "# ai_agg\n\n"
        "LLM map-reduce over the rows of a group, returning one `VARCHAR` per group.\n\n"
        "## Notes\n\n"
        "- Chunked map-reduce handles groups larger than the context window.\n"
        "- `task` is a constant instruction; NULL for an empty group.\n"
        "- Keys come from a CREATE SECRET (TYPE llm) or env vars (or keyless Ollama)."
    ),
    keywords=[
        "ai",
        "llm",
        "aggregate",
        "group by",
        "map reduce",
        "reduce",
        "summarize group",
        "combine",
        "roll up",
        "per group",
    ],
    category="aggregate",
)


class AiAgg(_AiAggBase):
    """``ai_agg(input, task)`` -- LLM map-reduce over a group's rows."""

    class Meta:
        """Declarative metadata for ``ai_agg(input, task)``."""

        name = "ai_agg"
        description = "Aggregate a group's text rows by applying a task via chunked LLM map-reduce; VARCHAR per group."
        categories = ["aggregate"]
        null_handling = NullHandling.DEFAULT
        required_secrets = ["llm"]
        tags = _AI_AGG_TAGS
        examples = [
            FunctionExample(
                sql=(
                    "SELECT llm.main.ai_agg(comment, 'List the top complaints') "
                    "FROM (VALUES ('too slow'), ('buggy UI')) AS t(comment)"
                ),
                description="Reduce a group's rows to one answer with a task",
            )
        ]

    @classmethod
    def update(
        cls,
        states: dict[int, AggState],
        group_ids: pa.Int64Array,
        value: Annotated[pa.StringArray, Param(doc="Text column to aggregate over the group.")],
        task: Annotated[str, ConstParam("Natural-language task to apply across the group.", phase="finalize")] = "",
    ) -> None:
        """Buffer each group's row texts into serializable state."""
        by_group: dict[int, list[str]] = {}
        gids = group_ids.to_pylist()
        for gid, val in zip(gids, value.to_pylist(), strict=True):
            if gid is not None and val is not None and val.strip():
                by_group.setdefault(gid, []).append(val)
        for gid, new_texts in by_group.items():
            current = states[gid]
            states[gid] = AggState(texts=_collapse(current.texts + new_texts))

    @classmethod
    def finalize(
        cls,
        group_ids: pa.Int64Array,
        states: dict[int, AggState | None],
        params: ProcessParams[Any],
    ) -> Annotated[pa.RecordBatch, Returns(pa.string())]:
        """Reduce each group's buffer with the constant ``task``."""
        task = ""
        if params.args and params.args.positional:
            scalar = params.args.positional[0]
            task = scalar.as_py() if scalar is not None else ""
        return cls._finalize_with_task(group_ids, states, task or "", params)


_AI_SUMMARIZE_AGG_TAGS = meta.object_tags(
    title="AI Summarize Over A Group",
    doc_llm=(
        "## ai_summarize_agg(input)\n\n"
        "Summarize a whole column of text within each `GROUP BY` group into one "
        "coherent `VARCHAR` summary per group. It is `ai_agg` with a fixed "
        "summarize task: rows are buffered and reduced via chunked map-reduce, so "
        "a group larger than the model's context window still yields one summary.\n\n"
        '**When to use.** "One summary per category over all its rows" -- grouping a '
        "text column and summarizing each group (see the attached example queries). For a custom "
        "instruction use `ai_agg`; for a per-row summary use `ai_summarize`.\n\n"
        "**Input/output.** Input: a VARCHAR column. Output: one VARCHAR summary per "
        "group. An empty group yields NULL; a provider failure is raised. Keys come "
        "from a CREATE SECRET (TYPE llm) or env vars (or keyless Ollama)."
    ),
    doc_md=(
        "# ai_summarize_agg\n\n"
        "LLM summarization across the rows of a group, one `VARCHAR` per group.\n\n"
        "## Notes\n\n"
        "- Chunked map-reduce handles groups larger than the context window.\n"
        "- NULL for an empty group; a provider failure raises an error.\n"
        "- Keys come from a CREATE SECRET (TYPE llm) or env vars (or keyless Ollama)."
    ),
    keywords=[
        "ai",
        "llm",
        "summarize",
        "summary",
        "aggregate",
        "group by",
        "map reduce",
        "summarize group",
        "roll up",
        "per group",
    ],
    category="aggregate",
)


class AiSummarizeAgg(_AiAggBase):
    """``ai_summarize_agg(input)`` -- LLM summary across a group's rows."""

    class Meta:
        """Declarative metadata for ``ai_summarize_agg(input)``."""

        name = "ai_summarize_agg"
        description = "Summarize all of a group's text rows into one summary via chunked map-reduce; VARCHAR per group."
        categories = ["aggregate"]
        null_handling = NullHandling.DEFAULT
        required_secrets = ["llm"]
        tags = _AI_SUMMARIZE_AGG_TAGS
        examples = [
            FunctionExample(
                sql=("SELECT llm.main.ai_summarize_agg(note) FROM (VALUES ('login failed'), ('disk full')) AS t(note)"),
                description="Summarize all of a group's rows into one summary",
            )
        ]

    @classmethod
    def update(
        cls,
        states: dict[int, AggState],
        group_ids: pa.Int64Array,
        value: Annotated[pa.StringArray, Param(doc="Text column to summarize over the group.")],
    ) -> None:
        """Buffer each group's row texts into serializable state."""
        by_group: dict[int, list[str]] = {}
        gids = group_ids.to_pylist()
        for gid, val in zip(gids, value.to_pylist(), strict=True):
            if gid is not None and val is not None and val.strip():
                by_group.setdefault(gid, []).append(val)
        for gid, new_texts in by_group.items():
            current = states[gid]
            states[gid] = AggState(texts=_collapse(current.texts + new_texts))

    @classmethod
    def finalize(
        cls,
        group_ids: pa.Int64Array,
        states: dict[int, AggState | None],
        params: ProcessParams[Any],
    ) -> Annotated[pa.RecordBatch, Returns(pa.string())]:
        """Reduce each group's buffer with the fixed summarize task."""
        return cls._finalize_with_task(group_ids, states, _SUMMARIZE_TASK, params)


AGGREGATE_FUNCTIONS: list[type] = [AiAgg, AiSummarizeAgg]
