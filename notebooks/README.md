# LLM notebooks

[01 · Rank documents, then budget the rendered context](01_rank_then_budget_the_context.ipynb) uses real local embeddings to order synthetic support notes, then selects a complete ranking prefix using the token count of its fully rendered prompt text.

## Run

Install [uv](https://docs.astral.sh/uv/) and Git. Python 3.13–3.14 is required; uv can install it. First use needs internet for packages, the worker, extension, tokenizer resources and approximately **67 MB of pinned embedding assets**. No provider account or API key is required. Text inference runs locally; the notebook does not generate a completion.

```sh
cd notebooks
uv sync --locked
uv run --locked python execute.py --timeout 300
```

The notebook runs Haybarn 2.0.0rc11. The visible literal ATTACH uses `uvx --from git+https://github.com/Query-farm/vgi-llm@v0.4.0 ... vgi-llm-worker`; the tag resolves to `49a1f9e35edae6cb247ee0a07205c48b41833588`. Explicit worker pins are vgi-python 0.51.0, vgi-rpc 0.53.1, fastembed 0.9.1, onnxruntime 1.31.1, tiktoken 0.14.0, pyarrow 25.0.1 and numpy 2.5.4. uv installs and caches this Git package independently of any worker checkout. Upgrade the source/runtime pins deliberately and rerun both budgets.

Optional editor kernel:

```sh
uv run --locked python -m ipykernel install --user --name vgi-llm-notebooks --display-name "Python (vgi-llm notebooks)"
```

Use this directory as working directory and Restart Kernel → Run All. The runner creates a temporary kernel, saves only after all cells succeed, and preserves previous successful defaults on failure.

## Experiment

```sh
uv run --locked python execute.py --parameters '{"TOKEN_BUDGET":220}' --output-dir artifacts/budget-220
```

Restore 120 for the default. Overrides require a separate destination. Documentation, branding, lock, asset helper and model manifest accompany the executed copy. Generated model caches, charts, environments and experiments are ignored. A fresh detached copy downloads its own verified model assets; it does not require the original directory.

## Model provenance

The worker's `BAAI/bge-small-en-v1.5` embedding path uses the public [Qdrant ONNX model repository](https://huggingface.co/Qdrant/bge-small-en-v1.5-onnx-Q/tree/aa8f8b060edb00e03bfdd08813a2949946c8ba55). `model-manifest.json` pins immutable revision `aa8f8b060edb00e03bfdd08813a2949946c8ba55`, file sizes and SHA-256 hashes for all five required ONNX/config/tokenizer assets, totaling 67,179,163 bytes. The model produces 384-dimensional vectors. No model weights are committed here.

`embedding_assets.py` downloads exact immutable URLs anonymously into `artifacts/embedding-cache`, verifies size and digest before atomically publishing each file, and publishes the cache ref only after all files are verified. Existing cache files are hash-checked before reuse. Concurrent preparation uses distinct temporary files and publishes only verified complete files. It then sets the unchanged worker's `VGI_LLM_CACHE_DIR`/`FASTEMBED_CACHE_PATH` and enables Hugging Face offline mode so the worker cannot silently replace the pinned revision. Previous environment values are restored in cleanup. An unavailable or mismatched asset fails clearly; no fabricated embeddings or alternate model are supplied.

The same worker embedding path is used for both documents and query. No claim is made that this is an optimized retrieval setup or that scores are calibrated. The notebook validates dimensions, finite vectors, self-similarity and ranking consistency, and records a document-vector fingerprint for the budget comparison. Floating-point details may vary across inference platforms.

## Inputs and budget semantics

Ten SQL source rows are synthetic teaching notes: eight nonempty documents plus one NULL and one blank body. They do not describe a real service's security or recovery policy. Missing/blank inputs retain unavailable embeddings and are excluded from ranking rather than assigned a zero similarity.

The budget applies to the entire rendered string: instruction, fixed query, document IDs, bodies and separators. `prompt()` is pure template substitution. `ai_count_tokens(...,'openai/gpt-4o')` uses tiktoken's `o200k_base` encoding for that text; the model name selects a tokenizer, not a provider request. First use may download tokenizer data. Counting whole strings avoids assuming separate body-token counts are additive.

The selected set is the largest complete cosine-ranked prefix that fits. No document is truncated, and there is no alternative-subset optimization. Prefix size zero still contains instructions and the query; if even that text exceeds the budget, the result is explicitly unavailable. Increasing the budget changes selection, while the query, vector fingerprint, ranking and token-count curve remain fixed.

These are rendered-text counts, not a provider usage report. Message framing, tools, output allowance and other provider overhead are excluded. The notebook estimates neither billable usage nor the effective context limit of a completion API. Similarity and expanded context are not evidence of relevance probability, factual correctness or answer quality; no labeled benchmark or generation evaluation is performed. There is no replay/live mode because these are local computations.
