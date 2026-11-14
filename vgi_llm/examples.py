# Copyright 2026 Query Farm LLC - https://query.farm

"""Composed example queries that combine several of the worker's functions.

Each example is defined once here and referenced from both the driving
function's ``Meta.examples`` (what agents and ``duckdb_functions()`` see) and the
schema-level ``vgi.example_queries`` list in :mod:`vgi_llm.catalog`. All three
are keyless and deterministic (local ONNX embeddings, local tiktoken, pure
template substitution), so they also run as ``vgi.executable_examples``.
"""

from __future__ import annotations

from vgi.metadata import FunctionExample

#: Semantic search: embed the query once, rank a document table against it.
SEMANTIC_SEARCH = FunctionExample(
    description=(
        "Semantic search (keyless): embed the search phrase once with ai_embed and rank a table of "
        "documents by ai_similarity to it, keeping the top 2. The two database sentences outrank the "
        "unrelated ones."
    ),
    sql=(
        "WITH docs(doc) AS (VALUES ('DuckDB is an in-process analytical SQL database'), "
        "('Paris is the capital of France'), ('Kittens are young cats'), "
        "('PostgreSQL is a client-server relational database')), "
        "q AS (SELECT llm.main.ai_embed('embedded database for analytics') AS v) "
        "SELECT doc, ROUND(llm.main.ai_similarity(llm.main.ai_embed(doc), q.v), 3) AS score "
        "FROM docs, q ORDER BY score DESC LIMIT 2"
    ),
)

#: Zero-shot routing: nearest label description per ticket, no LLM call.
ROUTE_TICKETS = FunctionExample(
    description=(
        "Keyless zero-shot routing: embed each support ticket and each label's description with "
        "ai_embed, then pick the label with the highest ai_similarity per ticket via arg_max "
        "(billing / bug / feature, one row per ticket with its best score)."
    ),
    sql=(
        "WITH tickets(ticket) AS (VALUES ('I was charged twice for my subscription this month'), "
        "('The app crashes every time I upload a photo'), ('Please add a dark mode to the dashboard')), "
        "labels(label, hint) AS (VALUES ('billing', 'payments, invoices, refunds and charges'), "
        "('bug', 'the software crashes or shows an error'), ('feature', 'a request for a new capability')), "
        "t AS (SELECT ticket, llm.main.ai_embed(ticket) AS v FROM tickets), "
        "l AS (SELECT label, llm.main.ai_embed(hint) AS v FROM labels) "
        "SELECT ticket, arg_max(label, llm.main.ai_similarity(t.v, l.v)) AS label, "
        "ROUND(max(llm.main.ai_similarity(t.v, l.v)), 3) AS score "
        "FROM t, l GROUP BY ticket ORDER BY ticket"
    ),
)

#: Pre-flight cost check: render prompts per row, count their tokens locally.
PROMPT_TOKEN_BUDGET = FunctionExample(
    description=(
        "Pre-flight cost estimate before calling a model: render one prompt per review row with "
        "prompt() and count its tokens locally with ai_count_tokens, plus the batch total "
        "(no model call, no key)."
    ),
    sql=(
        "WITH reviews(id, product, review) AS (VALUES "
        "(1, 'laptop', 'Battery barely lasts two hours and the fan is loud.'), "
        "(2, 'headphones', 'Great sound, but the ear cushions started peeling after a month.'), "
        "(3, 'blender', 'Works fine.')) "
        "SELECT id, llm.main.ai_count_tokens(llm.main.prompt("
        "'Summarize this {} review in one sentence: {}', product, review)) AS prompt_tokens, "
        "SUM(prompt_tokens) OVER () AS total_tokens "
        "FROM reviews ORDER BY id"
    ),
)

#: Every composed example, in the order the schema-level list shows them.
COMPOSED_EXAMPLES: list[FunctionExample] = [SEMANTIC_SEARCH, ROUTE_TICKETS, PROMPT_TOKEN_BUDGET]
