# Copyright 2026 Query Farm LLC - https://query.farm

"""VGI worker exposing the ``aisql`` catalog (AI SQL functions) to DuckDB.

DuckDB spawns this over stdio and ATTACHes it::

    INSTALL vgi FROM community; LOAD vgi;
    ATTACH 'aisql' (TYPE vgi, LOCATION 'vgi-aisql-worker');

    -- Keyless: local ONNX embeddings + similarity
    SELECT aisql.ai_similarity(aisql.ai_embed('cat'), aisql.ai_embed('kitten'));

    -- With a key (or keyless Ollama): completions and structured outputs
    CREATE SECRET (TYPE aisql, openrouter_api_key 'sk-or-...');
    SELECT aisql.ai_complete('Write a haiku about DuckDB');

The worker warms the local embedding model at startup (best-effort) so the first
``ai_embed`` query does not pay the one-time model load/download inline.
"""

from __future__ import annotations

from typing import Annotated, Any

from vgi import Worker
from vgi.catalog.setting import Setting

from vgi_aisql import models
from vgi_aisql.catalog import AISQL_SECRET_TYPE, make_catalog


class AiSqlWorker(Worker):
    """Worker process hosting the ``aisql`` catalog."""

    catalog = make_catalog()
    secret_types = [AISQL_SECRET_TYPE]  # noqa: RUF012 - declarative worker configuration

    class Settings:
        """Optional ``aisql_*`` DuckDB session settings (unset -> library default).

        Set them per session with ``SET aisql_max_tokens = 8192`` etc. to tune the
        provider calls without changing SQL. Every setting carries a default so it
        is always delivered to the worker. ``aisql_temperature`` / ``aisql_top_p``
        default to a negative **sentinel** meaning "not sent" -- current Anthropic
        models reject those parameters, so they are omitted unless you set a real
        value (which, while routing to Anthropic, will then error loudly).
        """

        aisql_max_tokens: Annotated[int, Setting(desc="Max output tokens per completion (default 4096).")] = 4096
        aisql_temperature: Annotated[
            float, Setting(desc="Sampling temperature (0-2); < 0 means unset/not sent.")
        ] = -1.0
        aisql_top_p: Annotated[float, Setting(desc="Nucleus-sampling top_p (0-1); < 0 means unset/not sent.")] = -1.0
        aisql_model: Annotated[
            str, Setting(desc="Global default model when a call's model arg is empty ('' = none).")
        ] = ""
        aisql_max_workers: Annotated[int, Setting(desc="Max concurrent provider calls per batch (default 8).")] = 8
        aisql_timeout: Annotated[float, Setting(desc="Per-request provider timeout in seconds (default 60).")] = 60.0

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
    AiSqlWorker.main()


if __name__ == "__main__":
    main()
