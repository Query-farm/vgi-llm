# CLAUDE.md — vgi-llm

Guidance for AI agents (and humans) working in this repository.

## What this is

`vgi-llm` is a [VGI](https://github.com/Query-farm) worker exposing Snowflake
Cortex **AISQL-style** AI functions to DuckDB over a **pluggable LLM provider**,
plus **keyless** local embeddings. It is built on `vgi-python`.

## Layout

```
vgi_llm/
  providers/          # provider abstraction (DO NOT rewrite): base, anthropic, openai_compat, registry
  secrets.py          # key extraction from resolved VGI secrets (DO NOT rewrite)
  models.py           # fastembed ONNX registry + cached load + warm_up()
  engine.py           # shared LLM core: resolve_provider seam, map_complete, JSON/bool parsers
  scalars.py          # per-row scalar functions (ai_complete, ai_classify, ai_embed, ...)
  aggregates.py       # ai_agg / ai_summarize_agg (chunked map-reduce)
  catalog.py          # declarative catalog + vgi.* tag metadata + the `llm` secret type
  worker.py           # LlmWorker(Worker) + main(); warms the model in run()
tests/                # offline unit tests (FakeProvider injected at the engine seam)
test/sql/             # haybarn SQLLogic E2E (.test): bind-only schema + keyless + live-gated
```

## Conventions

- Python ≥ 3.13, `from __future__ import annotations`, Google docstrings.
- Copyright header on every module: `# Copyright 2026 Query Farm LLC - https://query.farm`.
- **Errors throw; only empty input is NULL.** A provider/runtime failure
  (`ProviderError`, missing key, unparseable structured reply) must **propagate**
  and surface as a DuckDB error — do NOT swallow it to NULL. Only an empty/NULL
  *input* row maps to a NULL output (with no provider call). `engine.map_complete`
  and `aggregates._map_reduce` enforce this; don't reintroduce `except ...: return None`.
- Scalars are **positional-only**: an optional `model` is a second/third *arity
  overload* (two classes sharing `Meta.name`), never a named argument.
- Structured functions (`ai_classify`/`ai_sentiment`/`ai_extract`) add a JSON
  system instruction **and** request `response_format`; a reply that won't parse
  **raises** (via `_parse_or_raise`), it does not yield NULL.
- STRUCT outputs are fixed at bind via a module-level `pa.struct([...])` and
  `Returns(arrow_type=...)`.
- Generic pyarrow annotations need a type arg under mypy strict, but the runtime
  can't subscript them — use the `_ListArray`/`_Array` `TYPE_CHECKING` aliases.
- The `llm_*` DuckDB settings are declared on `LlmWorker.Settings` (all with a
  default so DuckDB always delivers them) and read via `Setting()` compute params
  → `engine.read_settings`. `llm_temperature`/`llm_top_p` use a negative
  sentinel for "unset"; `llm_model` uses `''`.

## Testing seam

Everything provider-backed routes through `engine.resolve_provider`. Offline
tests monkeypatch that symbol (`tests/fake_provider.install`) to return a
`FakeProvider`, so no SDK or network is touched. Embedding tests use the real
local ONNX model and are gated on `tests.harness.model_available`.

## Gates (run from the repo root)

```sh
uv sync --all-extras
uv run ruff check --fix . && uv run ruff format .
uv run mypy vgi_llm/
uv run pydoclint --config pyproject.toml vgi_llm/
uv run pytest -n auto
./run_tests.sh                 # haybarn SQLLogic E2E (needs haybarn-unittest + vgi extension)
```

Iterate until ruff, mypy `--strict`, pydoclint, and pytest are green. The SQL
E2E and `vgi-lint` need external tooling (the community `vgi` DuckDB extension /
`vgi-lint-check`); the bind-only + keyless SQL legs run without any key.

## Gotchas

- Aggregates have secrets/settings **only in `on_bind`** (not in
  `update`/`finalize`). `_AiAggBase.on_bind` reads the resolved `llm` secret and
  the settings, and stashes them in the process-local `_BIND_CONFIG` keyed by the
  const-args; `finalize` reads them back. Env vars remain the fallback.
  **Read secrets with `params.secrets.to_dict()`, never `params.secrets.get(...)`**
  — `get()` registers a pending two-phase lookup on a miss, and aggregates get no
  bind retry, so `aggregate_bind` raises `NotImplementedError`. With no
  `CREATE SECRET (TYPE llm, …)` (the normal case: env-var keys, keyless Ollama)
  that broke *every* `ai_agg`/`ai_summarize_agg` call at bind. `test/sql/schema.test`
  DESCRIBEs both aggregates with no secret configured to guard it.
- `ai_count_tokens` uses **tiktoken** locally (no network): exact for OpenAI
  models, `o200k_base` estimate elsewhere. Do not add a provider call to it.
- `ai_extract` returns a JSON `VARCHAR` (parse with DuckDB JSON functions), not a
  dynamic STRUCT.
- `prompt()` uses a **safe** substitutor (`_safe_format`), not `str.format` — do
  not reintroduce `str.format` (attribute-traversal / format-spec DoS).
- `map_complete` de-dups identical prompts (one call per distinct prompt); a
  FakeProvider call counter asserts it. Provider `timeout` threads through
  `resolve`/`build_provider` → `BaseProvider(timeout=)`.
- The embedding cache dir is `VGI_LLM_CACHE_DIR` (falls back to
  `FASTEMBED_CACHE_PATH`).
