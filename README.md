# vgi-aisql

**Call LLMs and embed text directly in DuckDB SQL.**

`vgi-aisql` is a [VGI](https://github.com/Query-farm) worker that brings
[Snowflake Cortex **AISQL**](https://docs.snowflake.com/en/user-guide/snowflake-cortex/aisql)-style
AI functions to DuckDB — completion,
classification, boolean filtering, extraction, sentiment, summarization, and
group-level LLM map-reduce — over a **pluggable provider** (Anthropic,
OpenRouter, OpenAI, or a local Ollama), plus **keyless** local text embeddings
and cosine similarity (fastembed / ONNX, no torch, no key).

Empty/NULL **input** yields a `NULL` output row (never an error). A
**provider/runtime failure** — a missing key, an unreachable endpoint, an
unparseable structured reply — **raises a DuckDB error** rather than silently
returning `NULL`, so a broken configuration is loud, not a column of quiet
`NULL`s.

## Contents

- [Requirements](#requirements)
- [Install and attach](#install-and-attach)
- [Keyless in 10 seconds](#keyless-in-10-seconds)
- [Add a cloud provider](#add-a-cloud-provider)
- [Functions](#functions)
- [Examples](#examples)
- [Snowflake Cortex compatibility](#snowflake-cortex-compatibility)
- [Embedding models](#embedding-models)
- [Performance and cost](#performance-and-cost)
- [Running as a service (HTTP / Docker)](#running-as-a-service-http--docker)
- [Design notes](#design-notes)
- [Development](#development)
- [License](#license)

## Requirements

- **Python 3.13+**
- The DuckDB **`vgi`** community extension (`INSTALL vgi FROM community;`)
- No API key to start (embeddings, similarity, `prompt`, and `ai_count_tokens`
  are fully local). A provider key unlocks the LLM functions.

## Install and attach

`vgi-aisql` is a standard Python package whose worker is launched by the
`vgi-aisql-worker` console script. Point DuckDB's `ATTACH ... LOCATION` at that
command.

**From source (no publish required):**

```sh
git clone https://github.com/Query-farm/vgi-aisql
cd vgi-aisql
uv sync
```

```sql
INSTALL vgi FROM community;
LOAD vgi;
-- uv reads the project and runs the worker; run DuckDB from the repo root
ATTACH 'aisql' (TYPE vgi, LOCATION 'uv run vgi-aisql-worker');
```

**Installed on PATH** (`pip install .` or `uv tool install .`) — then the
console script resolves directly:

```sql
ATTACH 'aisql' (TYPE vgi, LOCATION 'vgi-aisql-worker');
```

Functions live in the `aisql.main` schema. Qualify calls as `aisql.<fn>(...)`,
or set the search path once to call them unqualified:

```sql
SET search_path = 'aisql.main';
```

## Keyless in 10 seconds

Embeddings and similarity run **entirely in-process** with **no API key** — the
default model (`BAAI/bge-small-en-v1.5`, 384-dim, MIT) downloads once on first
use and is cached thereafter:

```sql
-- A phrase is more similar to itself / a related word than to an unrelated one
SELECT aisql.ai_similarity(aisql.ai_embed('cat'), aisql.ai_embed('kitten')) AS score;

-- Semantic search / RAG ranking, all in SQL (pairs with the DuckDB VSS extension)
SELECT id
FROM docs
ORDER BY aisql.ai_similarity(aisql.ai_embed(body), aisql.ai_embed('reset password')) DESC
LIMIT 5;
```

`prompt()` (template substitution) and `ai_count_tokens()` (a local token
estimate) are also pure and keyless.

## Add a cloud provider

Keys live in a DuckDB **secret**, never in SQL text. One unified `aisql` secret
carries a field per backend, so a single `CREATE SECRET` configures everything:

```sql
CREATE SECRET (
  TYPE aisql,
  anthropic_api_key  'sk-ant-...',
  openrouter_api_key 'sk-or-...',
  openai_api_key     'sk-...'
  -- ollama is keyless; set ollama_host (e.g. 'http://host:11434/v1') for a remote daemon
);

SELECT aisql.ai_complete('Write a haiku about DuckDB');
SELECT aisql.ai_complete('Summarize this', 'openrouter/anthropic/claude-sonnet-5');
```

The **model** argument routes to a backend by its leading path segment:

| Prefix | Backend | Secret field | Default model |
|---|---|---|---|
| `anthropic/…` | Anthropic (official SDK) | `anthropic_api_key` | `claude-opus-4-8` |
| `openrouter/…` | OpenRouter (one key, many models) | `openrouter_api_key` | `anthropic/claude-sonnet-5` |
| `openai/…` | OpenAI | `openai_api_key` | `gpt-4o` |
| `ollama/…` | Local Ollama daemon | *(keyless)* | `llama3.2` |

- A **bare** model id (no prefix) picks the default provider by which key is
  configured, in precedence order **OpenRouter → Anthropic → OpenAI → Ollama**.
- OpenRouter model ids are themselves provider-prefixed, so the segment after
  `openrouter/` is passed through intact — e.g.
  `openrouter/meta-llama/llama-3.1-70b-instruct`.
- Keys may instead come from per-provider secrets (e.g. `TYPE anthropic`) or the
  `ANTHROPIC_API_KEY` / `OPENAI_API_KEY` / `OPENROUTER_API_KEY` **environment
  variables** (handy for local/CI). Aggregates read keys from the environment
  only — see [Design notes](#design-notes).

Prefer keyless local completions? Run [Ollama](https://ollama.com) and use an
`ollama/…` model — no key at all.

## Tuning (settings)

Optional `aisql_*` DuckDB session settings tune the provider calls without
changing SQL. Set them per session; unset settings keep the library default.

```sql
SET aisql_max_tokens = 8192;                 -- raise the output-token cap
SET aisql_model = 'openrouter/anthropic/claude-sonnet-5';  -- default when a call's model arg is ''
SET aisql_max_workers = 16;                  -- more concurrency per batch
SET aisql_timeout = 120;                     -- per-request timeout (seconds)
```

| Setting | Type | Default | Effect |
|---|---|---|---|
| `aisql_max_tokens` | `BIGINT` | 4096 | Max output tokens per completion |
| `aisql_temperature` | `DOUBLE` | unset | Sampling temperature (set ≥ 0 to send) |
| `aisql_top_p` | `DOUBLE` | unset | Nucleus-sampling `top_p` (set ≥ 0 to send) |
| `aisql_model` | `VARCHAR` | unset | Default model when a call's `model` arg is empty |
| `aisql_max_workers` | `BIGINT` | 8 | Concurrency cap for per-row calls in a batch |
| `aisql_timeout` | `DOUBLE` | 60 | Per-request provider timeout, in seconds |

> Current Anthropic models reject `temperature` / `top_p`; leave them unset when
> routing to Anthropic (setting them will error — see [Design notes](#design-notes)).

## Functions

| Function | Signature | Returns |
|---|---|---|
| `ai_complete` | `(prompt[, model])` | `VARCHAR` — text completion |
| `ai_complete_details` | `(prompt[, model])` | `STRUCT{text, model, input_tokens, output_tokens, finish_reason}` |
| `ai_complete_image` | `(prompt, image BLOB[, model])` | `VARCHAR` — multimodal (vision) |
| `ai_classify` | `(input, categories LIST<VARCHAR>[, model])` | `STRUCT{labels LIST<VARCHAR>}` |
| `ai_filter` | `(predicate, input[, model])` | `BOOLEAN` — for `WHERE` |
| `ai_extract` | `(input, response_format[, model])` | `VARCHAR` (JSON) — schema-driven |
| `ai_sentiment` | `(input[, model])` | `STRUCT{overall, categories LIST<STRUCT{name, sentiment}>}` |
| `ai_summarize` | `(input[, model])` | `VARCHAR` — per-row summary |
| `ai_count_tokens` | `(input[, model])` | `BIGINT` — local **tiktoken** count (no model call) |
| `prompt` | `(template, args...)` | `VARCHAR` — pure, safe template substitution |
| `ai_embed` | `(input[, model])` | `FLOAT[]` — **keyless** local embedding |
| `ai_similarity` | `(a FLOAT[], b FLOAT[])` | `DOUBLE` — **keyless** cosine similarity of two vectors |
| `ai_similarity` | `(a VARCHAR, b VARCHAR[, model])` | `DOUBLE` — **keyless** cosine similarity of two texts |
| `ai_agg` | `(input, task)` — aggregate | `VARCHAR` per group — LLM map-reduce |
| `ai_summarize_agg` | `(input)` — aggregate | `VARCHAR` per group — summary map-reduce |

The optional `model` argument is a **positional** second/third argument (scalar
functions are positional-only in DuckDB — there is no `model := …` for scalars).

## Examples

Assumes `SET search_path = 'aisql.main';` (else prefix each call with `aisql.`):

```sql
-- Classify support tickets against a fixed taxonomy
SELECT ai_classify(body, ['billing','bug','feature']).labels FROM tickets;

-- Semantic WHERE filter
SELECT * FROM reviews WHERE ai_filter('the customer is angry', body);

-- Schema-driven extraction to JSON, then pull a field with DuckDB's JSON ops
SELECT ai_extract(
  'Invoice 42 for $9.99 due 2026-01-01',
  '{"type":"object","properties":{"invoice":{"type":"integer"},"amount":{"type":"number"}}}'
)::JSON ->> 'amount';

-- Aspect-based sentiment
SELECT ai_sentiment(review).overall FROM reviews;

-- Token metadata alongside the answer
SELECT ai_complete_details('Explain MVCC in one sentence').output_tokens;

-- Vision (multimodal): pass an image BLOB column
SELECT ai_complete_image('What is in this image?', img) FROM photos;

-- One answer per group via LLM map-reduce
SELECT topic, ai_agg(comment, 'List the top three complaints')
FROM feedback GROUP BY topic;
```

## Snowflake Cortex compatibility

`vgi-aisql` mirrors the **capabilities** of the
[Snowflake Cortex AISQL](https://docs.snowflake.com/en/user-guide/snowflake-cortex/aisql)
functions with an idiomatic DuckDB surface (our own argument names and model
strings) — it is *not* a drop-in for Snowflake SQL. Rough mapping:

| Snowflake Cortex AISQL | vgi-aisql |
|---|---|
| `AI_COMPLETE` | `ai_complete`, `ai_complete_details`, `ai_complete_image` |
| `AI_CLASSIFY` | `ai_classify` |
| `AI_FILTER` | `ai_filter` |
| `AI_AGG` | `ai_agg` |
| `AI_SUMMARIZE_AGG` | `ai_summarize_agg` |
| `AI_EXTRACT` | `ai_extract` |
| `AI_SENTIMENT` | `ai_sentiment` |
| `SUMMARIZE` | `ai_summarize` |
| `AI_EMBED` | `ai_embed` |
| `AI_SIMILARITY` | `ai_similarity` |
| `AI_COUNT_TOKENS` | `ai_count_tokens` |
| `PROMPT` | `prompt` |

Some Cortex functions are intentionally **out of scope** here because a
dedicated VGI worker already does that job well (translation, PII redaction,
document parsing, transcription). Reach for these instead:

| Snowflake Cortex | Dedicated VGI worker |
|---|---|
| `AI_TRANSLATE` | [vgi-translate](https://github.com/Query-farm/vgi-translate) — offline neural machine translation |
| `AI_REDACT` | [vgi-pii](https://github.com/Query-farm/vgi-pii) — PII detection / redaction (Presidio) |
| `AI_PARSE_DOCUMENT` | [vgi-tika](https://github.com/Query-farm/vgi-tika) / [vgi-pdf](https://github.com/Query-farm/vgi-pdf) — document text + layout extraction |
| `AI_TRANSCRIBE` | [vgi-audio](https://github.com/Query-farm/vgi-audio) — audio/speech features |

Related workers worth pairing with `vgi-aisql`:
[vgi-embed](https://github.com/Query-farm/vgi-embed) (more local embedding
models + DuckDB VSS helpers), [vgi-rerank](https://github.com/Query-farm/vgi-rerank)
(cross-encoder RAG reranking), and
[vgi-tiktoken](https://github.com/Query-farm/vgi-tiktoken) (dedicated LLM token
counting; `ai_count_tokens` here is exact for OpenAI models and a close estimate
elsewhere).

## Embedding models

`ai_embed` / `ai_similarity` run locally via [fastembed](https://github.com/qdrant/fastembed)
(ONNX Runtime, no torch). Pass a second argument to `ai_embed` to select a model:

| Model | Dim | Notes |
|---|---|---|
| `BAAI/bge-small-en-v1.5` | 384 | **default** — MIT, strong general-purpose |
| `BAAI/bge-base-en-v1.5` | 768 | higher quality, larger |
| `BAAI/bge-small-en` | 384 | earlier bge-small |
| `sentence-transformers/all-MiniLM-L6-v2` | 384 | classic MiniLM baseline |

Only embed and compare vectors produced by the **same** model — dimensions and
vector spaces are not interchangeable. The download cache location is overridable
via `VGI_AISQL_CACHE_DIR` (or `FASTEMBED_CACHE_PATH`).

## Performance and cost

- The LLM scalar functions make **one provider call per row**. That is billable
  and network-bound — favor a cheap model (or keyless Ollama) for bulk scans,
  and push `WHERE`/`LIMIT` filtering *before* the AI call where you can.
- Each call batches its live rows across up to **8 concurrent** provider requests,
  so a batch of rows overlaps its network latency rather than running serially.
- `ai_embed` / `ai_similarity` / `prompt` / `ai_count_tokens` are local and free
  — no per-row cost.
- `ai_count_tokens` is a fast ~4-characters-per-token **estimate**, not an exact
  provider tokenizer count.

## Running as a service (HTTP / Docker)

The worker also runs as an HTTP service (the VGI RPC transport plus a `/health`
endpoint), which is what the bundled `Dockerfile` serves.

```sh
# Local HTTP server (requires the `serve` extra: vgi-python[http])
uv run --extra serve vgi-serve vgi_aisql.worker:AiSqlWorker --http --port 8000

# Container image — serves HTTP by default; `stdio` for on-host worker mode
docker build -t vgi-aisql .
docker run -p 8000:8000 vgi-aisql            # http (default)
docker run -i vgi-aisql stdio                 # stdio transport
```

Attach a running HTTP worker from DuckDB by its URL instead of a command:

```sql
ATTACH 'aisql' (TYPE vgi, LOCATION 'http://localhost:8000');
```

## Design notes

- **Errors throw; only empty input is NULL.** A provider/runtime failure — a
  missing key, an unreachable endpoint, or a structured function whose reply
  will not parse — raises a `ProviderError` that surfaces as a DuckDB error. Only
  an empty/NULL *input* row maps to a `NULL` output (with no provider call). This
  is deliberate: a misconfiguration is loud, not a silent column of `NULL`s.
- **Structured functions are model-agnostic.** `ai_classify` / `ai_sentiment` /
  `ai_extract` both request the provider's JSON-Schema `response_format` *and*
  add a system instruction to reply with JSON only, so they also work on models
  that don't support `response_format` (most OpenRouter models, Ollama). A reply
  that still won't parse raises (with a snippet) rather than yielding NULL.
- **STRUCT outputs are fixed at bind** (declared Arrow struct types).
- **`ai_extract`** returns a JSON `VARCHAR`; parse it with DuckDB's JSON
  functions. The `response_format` argument is a JSON-Schema string.
- **`prompt()` is safe.** It uses a purpose-built positional substitutor (`{}` /
  `{n}` with `{{`/`}}` escapes), **not** `str.format` — so a template can never
  traverse object attributes (`{0.__class__}`) or trigger a huge-width
  allocation (`{:>9999999999}`). A malformed template yields `NULL` (it is a pure
  function, so it does not raise).
- **`ai_count_tokens` uses `tiktoken`** locally (no network): exact for OpenAI
  models, and a strong `o200k_base` estimate for other models. For exact
  Anthropic counts, use the provider's own token-count API.
- **Aggregates honor `CREATE SECRET`.** `ai_agg` / `ai_summarize_agg` capture the
  resolved `aisql` secret and the `aisql_*` settings at bind time and reuse them
  during finalize, falling back to `*_API_KEY` environment variables (or keyless
  Ollama).
- **`ai_agg` / `ai_summarize_agg` use hierarchical map-reduce** — a group larger
  than the model's context is reduced in chunks, so it scales past a single
  prompt.
- **Within-batch dedup.** Each per-row scalar calls the provider **once per
  distinct prompt** in a batch and fans the result to every matching row — a big
  saving on repeated column values. Cross-*query* caching is a separate concern
  (see `vgi-cache` / `vgi-proxy`).
- **Usage accounting** (token counts) is carried only by `ai_complete_details`
  (the STRUCT-returning path); the `VARCHAR`/`BOOLEAN`/`STRUCT` returns of the
  other functions can't carry usage without changing their types. For those, the
  within-batch dedup above is the cost lever.
- **Anthropic sampling caveat.** Current Claude models reject `temperature` /
  `top_p` (HTTP 400). They are left **unset** by default; if you `SET
  aisql_temperature`/`aisql_top_p` while routing to Anthropic, the call now
  errors loudly (a direct consequence of errors throwing) — set them only for
  providers that accept them.

## Development

```sh
uv sync --all-extras
uv run ruff check --fix . && uv run ruff format .
uv run mypy vgi_aisql/
uv run pytest -n auto
```

The SQL end-to-end suite (`test/sql/*.test`) runs against the community `vgi`
DuckDB extension via `haybarn-unittest`; see `run_tests.sh`. The DESCRIBE-schema
legs are bind-only (no network) and always run; live legs are gated on
`ANTHROPIC_API_KEY` / Ollama reachability and skip cleanly when absent.

## License

MIT — Copyright 2026 Query Farm LLC.
