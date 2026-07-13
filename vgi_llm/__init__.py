# Copyright 2026 Query Farm LLC - https://query.farm

"""vgi-llm: Snowflake Cortex AISQL-style AI functions for DuckDB.

Exposes LLM completion, classification, filtering, extraction, sentiment,
summarization, and group-level map-reduce over a pluggable provider (Anthropic /
OpenRouter / OpenAI / Ollama), plus keyless local embeddings and cosine
similarity (fastembed/ONNX). See :mod:`vgi_llm.worker` for the worker entry
point and :mod:`vgi_llm.catalog` for the declarative catalog.
"""

from __future__ import annotations

__version__ = "0.1.0"

__all__ = ["__version__"]
