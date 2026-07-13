# vgi-aisql — Fix Spec (top-11 review findings)

Implement all 11 fixes below. **Guiding principle (explicit user directive):
provider/runtime errors must THROW (surface as a DuckDB error with an actionable
message), not silently become NULL.** Empty/NULL *input* still maps to a NULL
output row — that is not an error.

Keep the fleet bar: `ruff`, `ruff format`, `mypy --strict`, `pytest -n auto`,
and `vgi-lint --no-execute` (the CI gate) all green. See "Gates" at the end.

## 1. Errors throw instead of returning NULL  *(engine.py, aggregates.py, scalars.py)*

- `engine.map_complete`: **remove** the error-swallowing. Provider-resolution
  failure (`ProviderError`/`MissingKeyError`) → **raise** (do not `return out`).
  Per-row: delete `except ProviderError: return None` and
  `except Exception: return None` in `_one` — let exceptions propagate (they
  surface when the `pool.map` iterator is consumed → out of `compute()` → DuckDB
  error). Wrap the raised error with function/provider context and a clear
  message. Blank/NULL input rows still map to `None`.
- `aggregates._map_reduce`: same — the resolve `except ProviderError: return None`
  and both `_one` `except` clauses must **raise**, not swallow. An empty group
  (no live texts) still returns `None`.
- Every scalar `compute` keeps emitting `None` for blank rows; nothing else may
  swallow provider errors.
- Update unit tests that asserted error→NULL to assert `pytest.raises(...)`.

## 2. Structured functions: robust across models, loud on real failure  *(scalars.py)*

`ai_classify` / `ai_sentiment` / `ai_extract`:
- Always add a **system instruction** telling the model to reply with ONLY a
  JSON object of the required shape — so models that don't support
  `response_format` (most OpenRouter models, Ollama) still comply.
- Keep passing `response_format` (providers that support it use it).
- Tolerant-parse with `parse_json_object`; if it does not parse to the expected
  object, **raise** a clear error naming the function + a snippet of the output.
  (No more silent all-NULL columns.)

## 3. Aggregates honor `CREATE SECRET`  *(aggregates.py)*

Aggregates *can* read secrets — `AggregateBindParams.secrets` exists and
`on_bind` supports `Secret(...)` annotations (`vgi/aggregate_function.py`).
- Add/extend `on_bind` on `_AiAggBase` to declare `Secret("aisql")`, capture the
  resolved fields into `AggState` as a serializable JSON-string field
  (`Annotated[str, ArrowType(pa.string())]`), and have `finalize` pass
  `engine.build_secrets(captured)` into `_map_reduce` →
  `resolve_provider(model, secrets=...)`. Keep the `*_API_KEY` env var as a
  fallback. Remove the "aggregates read env only" caveat from README/CLAUDE.

## 4 + 10. Sampling / output / routing knobs as DuckDB settings  *(scalars.py, aggregates.py, engine.py, providers)*

Expose optional settings (read via `Setting()` params; unset → current default):

| Setting | Type | Default | Effect |
|---|---|---|---|
| `aisql_max_tokens` | BIGINT | 4096 | output cap (fixes silent 4096 truncation) |
| `aisql_temperature` | DOUBLE | unset (not sent) | sampling |
| `aisql_top_p` | DOUBLE | unset | sampling |
| `aisql_model` | VARCHAR | unset | global default model when the per-call `model` arg is empty |
| `aisql_max_workers` | BIGINT | 8 | concurrency cap in `map_complete` |
| `aisql_timeout` | DOUBLE | 60 | provider request timeout (thread through `resolve`/`build_provider` → `BaseProvider(timeout=...)`) |

Add a shared `_params_from_settings(...)` → `CompletionParams`, and honor
`aisql_model` when the call's model is `''`. Wire into every LLM scalar and both
aggregates. **Document the Anthropic caveat:** current Claude models reject
`temperature`/`top_p` (400) — with errors now throwing, setting those while
routing to Anthropic will error loudly; leave them unset for Anthropic. This is
intentional/honest.

## 5. `prompt()` safe substitution (no format-string injection / DoS)  *(scalars.py)*

Replace `tmpl.format(*args)` with a **safe** positional substitutor: support
`{}` (sequential) and `{n}` (explicit index) plus `{{`/`}}` escapes; **reject
format specs and attribute/index access** (`{0.attr}`, `{0[k]}`, `{:...}`) — no
`str.format`, so no attribute traversal and no `{:>9999999999}` allocation
attack. A malformed template or out-of-range index → `None` for that row (pure
function, no provider — keep NULL, don't raise here). Implement with a regex,
not `str.format`.

## 6. `ai_similarity` text overload  *(scalars.py)*

Add `ai_similarity(a VARCHAR, b VARCHAR)` and
`ai_similarity(a VARCHAR, b VARCHAR, model VARCHAR)` → embed both locally
(`models.embed_texts`) and return cosine. Keep the existing FLOAT[]/FLOAT[]
forms. (Snowflake's `AI_SIMILARITY` is text-in.)

## 7. Accurate `ai_count_tokens`  *(scalars.py, pyproject.toml)*

Add `tiktoken` as a dependency. Map the `model` arg to an encoding
(OpenAI/GPT-family → `o200k_base`/`cl100k_base`; unknown/others → `o200k_base`
as a strong cross-model default), encode, and count — local, no network.
Replace the ~4-chars heuristic; honor the `model` arg. Document: exact for
OpenAI, a good estimate elsewhere; for exact Anthropic counts see `vgi-tiktoken`
/ the provider API.

## 8. Fix `coerce_bool` inversion + strict filter prompt  *(engine.py, scalars.py)*

- `coerce_bool`: drop digit tokens (`"0"`,`"1"`,`"t"`,`"f"`,`"y"`,`"n"` that
  cause misfires) — match only the first whole **word** in {`yes`,`no`,`true`,
  `false`} (case-insensitive). "Item 0: yes" must read **True**, not False.
- `ai_filter`: system prompt must demand a single word — exactly `true` or
  `false`.

## 9. Token accounting  *(docs; structurally bounded)*

VARCHAR/BOOLEAN/STRUCT returns cannot carry usage without changing their types.
`ai_complete_details` is the usage-bearing path (keep it). Document in
README/CLAUDE that per-call usage is available via `ai_complete_details`, and
that the within-batch dedup (#11) is the cost lever for the others. Do **not**
change existing return types. (Report this as "addressed within type
constraints" — do not silently claim full coverage.)

## 11. Within-batch prompt dedup  *(engine.py)*

In `map_complete`, dedup identical prompt strings: call the provider **once per
distinct prompt**, then fan the result to every row with that prompt (preserve
order and NULL rows). Big cost win for repeated column values. (Cross-query
caching remains the `vgi-cache` / `vgi-proxy` story — note it in README.)

## Docs

Update README.md + CLAUDE.md for: errors now throw (not NULL); the new
`aisql_*` settings table; aggregates now honor `CREATE SECRET`; `prompt()`
safe; `ai_similarity` text form; accurate `ai_count_tokens` (tiktoken);
within-batch dedup; the Anthropic sampling-param caveat.

## Tests

Update/add: error→raises (was →NULL); settings→params mapping; `prompt()` safety
(attribute traversal like `{0.__class__}` yields a literal/NULL, never executes;
`{{`/`}}` escapes; `{1}` index; out-of-range→NULL); `ai_similarity` text form;
`ai_count_tokens` via tiktoken is model-sensitive and ≠ len/4; `coerce_bool`
("Item 0: yes"→True); dedup (one provider call for duplicated prompts — assert
via a FakeProvider call counter); aggregate resolves via a captured `aisql`
secret. Keep the real-ONNX embedding tests model-gated.

## Gates (must verify, report actual results)

```
cd ~/Development/vgi-aisql && uv sync --all-extras
uv run ruff check --fix . && uv run ruff format .
uv run mypy vgi_aisql/
uv run pytest -n auto
uv run --project ~/Development/vgi-lint-check vgi-lint "$PWD/bin/vgi-aisql-worker" --fail-on info --no-execute
```
`--no-execute` (the CI gate) must stay **100/100**. Note explicitly: with errors
now throwing, `vgi-lint --execute` will error on the LLM per-function examples
when **no provider key** is configured (a direct consequence of fix #1) — the
keyless examples (embed/similarity/prompt/count_tokens) still execute. Report
that behavior; do not try to re-swallow errors to make `--execute` pass keyless.
