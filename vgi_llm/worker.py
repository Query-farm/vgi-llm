# Copyright 2026 Query Farm LLC - https://query.farm

"""VGI worker exposing the ``llm`` catalog (AI SQL functions) to DuckDB.

DuckDB spawns this over stdio and ATTACHes it::

    INSTALL vgi FROM community; LOAD vgi;
    ATTACH 'llm' (TYPE vgi, LOCATION 'vgi-llm-worker');

    -- Keyless: local ONNX embeddings + similarity
    SELECT llm.ai_similarity(llm.ai_embed('cat'), llm.ai_embed('kitten'));

    -- With a key (or keyless Ollama): completions and structured outputs
    CREATE SECRET (TYPE llm, openrouter_api_key 'sk-or-...');
    SELECT llm.ai_complete('Write a haiku about DuckDB');

The worker warms the local embedding model at startup (best-effort) so the first
``ai_embed`` query does not pay the one-time model load/download inline.
"""

from __future__ import annotations

from typing import Annotated, Any

from vgi import Worker
from vgi.catalog.setting import Setting

from vgi_llm import models
from vgi_llm.catalog import LLM_SECRET_TYPE, make_catalog


class LlmWorker(Worker):
    """Worker process hosting the ``llm`` catalog."""

    catalog = make_catalog()
    secret_types = [LLM_SECRET_TYPE]  # noqa: RUF012 - declarative worker configuration

    class Settings:
        """Optional ``llm_*`` DuckDB session settings (unset -> library default).

        Set them per session with ``SET llm_max_tokens = 8192`` etc. to tune the
        provider calls without changing SQL. Every setting carries a default so it
        is always delivered to the worker. ``llm_temperature`` / ``llm_top_p``
        default to a negative **sentinel** meaning "not sent" -- current Anthropic
        models reject those parameters, so they are omitted unless you set a real
        value (which, while routing to Anthropic, will then error loudly).
        """

        llm_max_tokens: Annotated[int, Setting(desc="Max output tokens per completion (default 4096).")] = 4096
        llm_temperature: Annotated[float, Setting(desc="Sampling temperature (0-2); < 0 means unset/not sent.")] = -1.0
        llm_top_p: Annotated[float, Setting(desc="Nucleus-sampling top_p (0-1); < 0 means unset/not sent.")] = -1.0
        llm_model: Annotated[
            str, Setting(desc="Global default model when a call's model arg is empty ('' = none).")
        ] = ""
        llm_max_workers: Annotated[int, Setting(desc="Max concurrent provider calls per batch (default 8).")] = 8
        llm_timeout: Annotated[float, Setting(desc="Per-request provider timeout in seconds (default 60).")] = 60.0

    def run(self, otel_config: Any = None) -> None:
        """Warm the local embedding model, then serve.

        Loading (and, on a cold cache, downloading) the ONNX model is lazy, so
        without this the first ``ai_embed`` query of every ATTACH pays that
        multi-second cost inline. Warming at spawn moves it ahead of any query.
        Best-effort; never fatal.

        Args:
            otel_config: Optional OpenTelemetry configuration passed through to
                the base worker.
        """
        models.warm_up()
        super().run(otel_config=otel_config)


def main() -> None:
    """Run the AISQL worker process (stdio or, via flags, HTTP)."""
    LlmWorker.main()


if __name__ == "__main__":
    main()
