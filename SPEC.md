# vgi-aisql — Build Specification

A VGI (Vector Gateway Interface) Python worker exposing Snowflake Cortex
AISQL-style AI functions to DuckDB, over a **pluggable LLM provider** abstraction
plus **local ONNX embeddings**. Built on `vgi-python` (~/Development/vgi-python).

Repo: `~/Development/vgi-aisql`. Catalog: `aisql`. License: MIT (Query Farm LLC).
Worker class/console-script: `vgi_aisql.worker:main` → `vgi-aisql-worker`.

## Design decisions (fixed — do not relitigate)

- **Pluggable providers, auto-routing.** A `model` string routes by leading path
  segment (`anthropic/…`, `openrouter/…`, `openai/…`, `ollama/…`); a bare model
  picks the default provider by which key is present (precedence: openrouter →
  anthropic → openai → ollama). See `vgi_aisql/providers/__init__.py::resolve`.
- **Adoption-first / keyless-first.** `ai_embed`/`ai_similarity` run on local
  ONNX (fastembed) with **no key**. Ollama gives keyless local completions.
  OpenRouter is the recommended cloud key (one key → many models).
- **Anthropic via the official `anthropic` SDK; OpenAI/OpenRouter/Ollama via the
  official `openai` SDK** (they are OpenAI-compatible). Clients are **injectable**
  for offline tests. Providers never crash the worker — they raise `ProviderError`.
- **Idiomatic surface** (not drop-in Snowflake): our own arg names + model strings.
- **STRUCT outputs are fixed at BIND** — declare a module-level `pa.struct([...])`
  and `Returns(arrow_type=...)`. Structured extraction uses provider
  `response_format` (JSON Schema) where the model supports it.

## Already implemented (do not rewrite)

- `pyproject.toml` (hatchling; ruff E,F,I,UP,B,SIM,D + google + double; mypy strict;
  console-script; `serve` extra; deps: vgi-python, anthropic, openai, fastembed, pyarrow).
- `vgi_aisql/providers/{base,anthropic_provider,openai_compat,__init__}.py` — the
  provider abstraction: `Message`, `ImagePart`, `CompletionParams`, `ResponseFormat`,
  `Usage`, `Completion`, `ChatProvider`, `BaseProvider`; adapters; `resolve()`,
  `build_provider()`, `available_providers()`.
- `vgi_aisql/secrets.py` — `key_from_secrets(secrets, provider)`; unified `aisql`
  secret with `<provider>_api_key` fields, or provider-named secrets.
- `LICENSE`, `.gitignore`.

## To build

### 1. Scalar functions (`vgi_aisql/scalars.py`)

Base classes from `vgi` / `vgi.arguments` / `vgi.scalar_function` (see reference
patterns below). Each `compute()` classmethod batches per-row LLM calls with a
bounded `ThreadPoolExecutor` (respect a `max_workers`), masks NULL/empty input →
NULL output, and catches per-row `ProviderError` → NULL (never crash). Read the
resolved secrets via a `Secret("aisql")` (+ per-provider `Secret(...)`) parameter;
read the model via a `ConstParam`. Functions:

| Function | Signature (idiomatic) | Returns |
|---|---|---|
| `ai_complete` | `(prompt, model := '', options := NULL)` | VARCHAR |
| `ai_complete_details` | `(prompt, model := '')` | STRUCT{text VARCHAR, model VARCHAR, input_tokens BIGINT, output_tokens BIGINT, finish_reason VARCHAR} |
| `ai_complete_image` | `(prompt, image BLOB, model := '')` | VARCHAR (multimodal) |
| `ai_classify` | `(input, categories LIST<VARCHAR>, model := '')` | STRUCT{labels LIST<VARCHAR>} |
| `ai_filter` | `(predicate, input, model := '')` | BOOLEAN (for WHERE) |
| `ai_extract` | `(input, response_format JSON/STRUCT, model := '')` | STRUCT (schema-driven) or JSON(VARCHAR) |
| `ai_sentiment` | `(input, model := '')` | STRUCT{overall VARCHAR, categories LIST<STRUCT{name VARCHAR, sentiment VARCHAR}>} |
| `ai_summarize` | `(input, model := '')` | VARCHAR |
| `ai_count_tokens` | `(input, model := '')` | BIGINT (see note) |
| `prompt` | `(template, args...)` | VARCHAR — **pure, no model call**; simple `{}`/named substitution |

Notes:
- Reserved-keyword arg trap: DuckDB rejects a bare `<reserved> := value` and VGI
  table args are named-only. `model`/`input`/`options` should be fine as names but
  verify at E2E; rename if DuckDB objects (e.g. `model_name`).
- `ai_classify`/`ai_sentiment`/`ai_extract` build a system+user prompt instructing
  JSON output matching the STRUCT, request `response_format`, parse JSON, coerce to
  the STRUCT. On parse failure → NULL row.
- `ai_filter` maps a yes/no LLM answer to BOOLEAN; unparseable → NULL.
- `ai_count_tokens`: prefer a local tokenizer if trivially available; otherwise use
  the provider's token count (Anthropic `count_tokens`) — if that means a network
  call, gate it and document. A local heuristic is acceptable for v1; document it.

### 2. Aggregate functions (`vgi_aisql/aggregates.py`)

`AggregateFunction[State]` (see reference). Hierarchical chunked map-reduce so a
group larger than the context window still reduces:

- `ai_agg(input, task)` → VARCHAR — `update` buffers row texts into serializable
  state; `combine` merges buffers/partials; `finalize` chunk-reduces via the LLM
  (map each chunk to a partial answer, then reduce partials with the `task`).
- `ai_summarize_agg(input)` → VARCHAR — same shape with a fixed summarize task.

State is a `@dataclass(kw_only=True)` extending `ArrowSerializableDataclass` with
`ArrowType`-annotated fields (list of strings + the task). Keep a bounded buffer;
if it grows large, pre-reduce in `update`/`combine` to a running summary.

### 3. Local embeddings (`vgi_aisql/models.py` + embeddings in `scalars.py`)

Model registry + `@functools.cache`+lock `_load_model()` → `fastembed.TextEmbedding`,
cache dir via `VGI_AISQL_CACHE_DIR`→`FASTEMBED_CACHE_PATH`. `DEFAULT_MODEL =
"BAAI/bge-small-en-v1.5"` (384-dim). `warm_up()` called in a `Worker.run` override.

- `ai_embed(input, model := '')` → `FLOAT[]` (`pa.list_(pa.float32())`); NULL/empty
  → NULL vector.
- `ai_similarity(a, b)` → DOUBLE — cosine of two `ai_embed` vectors (pure math), OR
  `(text_a, text_b, model := '')` embedding both then cosine. Provide the
  vector-in form as the primary; a text-in convenience overload is optional.
- Optional API-embeddings path via OpenAI is a later enhancement; local is default.

### 4. Worker + declarative catalog (`vgi_aisql/worker.py`, `vgi_aisql/catalog.py`)

`class AiSqlWorker(Worker): catalog = make_catalog(...)` with `main()` → `.run()`.
Override `run()` to call `models.warm_up()` (best-effort). Declarative `Catalog`/
`Schema` with full `vgi.*` tag metadata (title/doc_llm/doc_md/keywords/categories/
executable_examples/agent_test_tasks on catalog+schema; per-fn doc_llm/doc_md/
category/argDocs + FunctionExample examples). Register all scalars + aggregates.
Declare the `aisql` secret type usage via `Meta.required_secrets` /
`SecretLookupEntry` and `Secret(...)` params.

### 5. Tests (`tests/`)

- **Unit, offline, deterministic:** a `FakeProvider` (records prompts, returns
  canned completions/embeddings) injected via `client=`/`resolve(..., client=)` or a
  fake provider seam. Cover: routing (`resolve`) incl. prefix vs default precedence;
  prompt assembly; `response_format`→STRUCT parse; `_details` envelope; classify/
  sentiment/extract JSON parse; `ai_filter` boolean coercion; aggregate update/
  combine/finalize incl. chunked reduce; NULL/error-per-row. Embeddings: run real
  local ONNX (skipif model unavailable) — assert vector length 384, self-similarity
  ≈ 1.0, related > unrelated; never exact floats.
- **haybarn SQLLogic E2E (`test/sql/*.test`):** header `# name/description/group`,
  `require-env VGI_TEST_WORKER`, `ATTACH 'aisql' AS a (TYPE vgi, LOCATION
  '${VGI_TEST_WORKER}')`. DESCRIBE-schema asserts (bind-only, no network) as the
  deterministic backbone. A keyless **Ollama** leg gated on reachability (probe with
  the worker's own driver; skip when down). Anthropic/OpenAI/OpenRouter live asserts
  gated on `ANTHROPIC_API_KEY`/etc. via `require-env`, skipped when absent → CI green
  anywhere. `prompt()` and `ai_embed`/`ai_similarity` run for real offline.
- Gates: `ruff check --fix && ruff format`, `mypy --strict vgi_aisql/`, `pydoclint`,
  `pytest -n auto`, `vgi-lint` 100/100 with offline-executable examples.

### 6. Docs/infra

`README.md` (lead with the keyless embed/similarity demo, then one-key OpenRouter
upgrade, then the full function table), `CLAUDE.md`, `run_tests.sh`,
`vgi-lint.toml` (`select=["ALL"] fail_on="info"`), `ci/check-version.sh`,
`.github/workflows/ci.yml` (pytest matrix 3.13/3.14 × ubuntu/macos; ruff; mypy;
haybarn e2e; vgi-lint), `Dockerfile` + `docker-entrypoint.sh` (dual transport),
modeled on `~/Development/vgi-etf-schwab`.

## Reference patterns (from the fleet — copy these idioms)

**Scalar function** (`~/Development/vgi-python/vgi/_test_fixtures/scalar/*.py`,
`~/Development/vgi-search/vgi_search/scalars.py`):
```python
from typing import Annotated, Any
import pyarrow as pa
from vgi.arguments import ConstParam, Param, Returns, Secret
from vgi.scalar_function import ScalarFunction
from vgi.metadata import FunctionExample

class AiComplete(ScalarFunction):
    class Meta:
        name = "ai_complete"
        description = "..."
        required_secrets = ["aisql"]  # + provider secrets as needed
        examples = [FunctionExample(sql="SELECT ai_complete('hi')", description="...")]

    @classmethod
    def compute(
        cls,
        prompt: Annotated[pa.StringArray, Param(doc="Prompt text")],
        model: Annotated[str, ConstParam("provider/model, or '' for default")] = "",
        aisql_secret: Annotated[dict[str, pa.Scalar[Any]] | None, Secret("aisql")] = None,
    ) -> Annotated[pa.StringArray, Returns()]:
        ...
```
Output-length helper: `OutputLength()` param gives row count when there's no input
array to size against. STRUCT return: `Returns(arrow_type=_STRUCT)` with
`pa.StructArray.from_arrays([...], names=[...])`.

**STRUCT output** (`.../scalar/geo.py::GeoCentroidStructFunction`): module-level
`_T = pa.struct([("a", pa.float64()), ("b", pa.float64())])`, `Returns(arrow_type=_T)`.

**Aggregate** (`~/Development/vgi-python/examples/sum_worker.py`,
`~/Development/vgi-python/docs/aggregate-functions.md`): `AggregateFunction[State]`
with `initial_state`/`update(states, group_ids, value)`/`combine(source, target,
params)`/`finalize(group_ids, states, params)`; `State(ArrowSerializableDataclass)`
fields annotated `Annotated[T, ArrowType(pa.type())]`.

**Local ONNX embeddings** (`~/Development/vgi-embed/vgi_embed/{models,scalars}.py`,
`embed_worker.py`): fastembed `TextEmbedding(model_name=, cache_dir=)`,
`@functools.cache`+`threading.Lock`, `warm_up()` in `Worker.run` override,
`_VECTOR = pa.list_(pa.float32())`, NULL-mask + splice, model-gated tests via a
`model_available()` skipif.

**Provider pattern** already realized here — model the function-layer secret reads
on `vgi-search`'s `key_from_secret` + `Secret(...)`/`required_secrets` usage.

**Scaffold** (`~/Development/vgi-etf-schwab/`): `worker.py` (`class XWorker(Worker):
catalog = ...`, `main()`), `catalog.py` (`Catalog`/`Schema` + `vgi.*` `tags`),
`Dockerfile`, `docker-entrypoint.sh` (`case "${1:-http}"` → `vgi-serve
vgi_aisql.worker:AiSqlWorker --http` / console-script), `run_tests.sh`
(`VGI_TEST_WORKER=... VGI_WORKER_CATALOG_NAME=aisql unittest test/sql/*`),
`vgi-lint.toml`, `ci/check-version.sh`, `.github/workflows/*`.

## Verification (must pass before "done")

```
cd ~/Development/vgi-aisql && uv sync --all-extras
uv run ruff check --fix . && uv run ruff format .
uv run mypy vgi_aisql/
uv run pytest -n auto
# haybarn E2E against the community vgi extension (see run_tests.sh / vgi-embed ci)
```
Report which gates are green and any that need the C++ vgi extension / live keys.
