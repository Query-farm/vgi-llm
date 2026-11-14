# Copyright 2026 Query Farm LLC - https://query.farm

"""Static checks on the published example SQL (no worker, no model)."""

from __future__ import annotations

import json

from vgi_llm import catalog, examples
from vgi_llm.aggregates import AGGREGATE_FUNCTIONS
from vgi_llm.scalars import SCALAR_FUNCTIONS

_ALL = [*SCALAR_FUNCTIONS, *AGGREGATE_FUNCTIONS]


def _all_example_sql() -> list[str]:
    sqls = [e.sql for cls in _ALL for e in cls.Meta.examples]
    sqls += [e["sql"] for e in json.loads(catalog._SCHEMA_EXAMPLE_QUERIES)]
    sqls += [e["sql"] for e in json.loads(catalog._EXECUTABLE_EXAMPLES)]
    return sqls


def test_example_sql_is_printable_ascii() -> None:
    """A Python ``'\\x89'`` escape once leaked a raw control char into a BLOB literal."""
    for sql in _all_example_sql():
        assert sql.isascii() and sql.isprintable(), repr(sql)


def test_image_example_uses_sql_hex_escape() -> None:
    sqls = [e.sql for cls in _ALL if cls.Meta.name == "ai_complete_image" for e in cls.Meta.examples]
    assert sqls and all("'\\x89PNG'::BLOB" in s for s in sqls)


def test_composed_examples_on_driving_function_and_schema() -> None:
    drivers = [
        (examples.SEMANTIC_SEARCH, "ai_similarity"),
        (examples.ROUTE_TICKETS, "ai_embed"),
        (examples.PROMPT_TOKEN_BUDGET, "ai_count_tokens"),
    ]
    schema_sql = {e["sql"] for e in json.loads(catalog._SCHEMA_EXAMPLE_QUERIES)}
    for ex, name in drivers:
        assert ex.sql in schema_sql
        tagged = [
            e["sql"] for cls in _ALL if cls.Meta.name == name for e in json.loads(cls.Meta.tags["vgi.example_queries"])
        ]
        assert ex.sql in tagged[:5], name
