# Copyright 2026 Query Farm LLC - https://query.farm

"""The ``llm`` declarative catalog and its ``vgi.*`` metadata tags.

Assembles every scalar and aggregate into a single ``llm`` catalog with the
full per-object / schema / catalog tag surface the strict VGI lint profile
grades. All example SQL is catalog-qualified (``llm.main.<fn>(...)``) so it
binds and (for the keyless functions) runs against an attached worker. The
one secret type, ``llm``, carries one field per provider so a single
``CREATE SECRET`` configures every backend.
"""

from __future__ import annotations

import json

import pyarrow as pa
from vgi.catalog import Catalog, Schema
from vgi.catalog.secret_type import SecretTypeSpec

from vgi_llm import models
from vgi_llm.aggregates import AGGREGATE_FUNCTIONS
from vgi_llm.scalars import SCALAR_FUNCTIONS

REPO = "https://github.com/Query-farm/vgi-llm"
ISSUES = f"{REPO}/issues"

#: The unified ``llm`` secret type: one field per provider so a single
#: ``CREATE SECRET (TYPE llm, ...)`` configures every backend.
LLM_SECRET_TYPE = SecretTypeSpec(
    name="llm",
    description="Provider keys for vgi-llm (one field per backend: Anthropic / OpenRouter / OpenAI / Ollama).",
    schema=pa.schema(
        [
            pa.field("anthropic_api_key", pa.string(), metadata={"redact": "true"}),
            pa.field("openrouter_api_key", pa.string(), metadata={"redact": "true"}),
            pa.field("openai_api_key", pa.string(), metadata={"redact": "true"}),
            pa.field("ollama_host", pa.string()),
        ]
    ),
)


_CATALOG_DOC_LLM = (
    "Snowflake Cortex AISQL-style AI functions for DuckDB over a pluggable LLM provider, plus "
    "keyless local embeddings. Call large language models directly from SQL to complete, classify, "
    "filter, extract, summarize, and analyze sentiment over columns of text, and to reduce a whole "
    "GROUP BY group to one answer with LLM map-reduce. Embeddings and cosine similarity run entirely "
    "in-process (fastembed/ONNX) with NO API key, so semantic search / RAG works out of the box; add "
    "one OpenRouter key to unlock hundreds of cloud models, or point at a local Ollama daemon for "
    "keyless local completions. The model is chosen per call by a provider-prefixed string "
    "(anthropic/…, openrouter/…, openai/…, ollama/…) or a bare id for the default provider. Every "
    "function degrades to NULL rather than erroring, so it is safe inside a larger scan. List this "
    "catalog's schema to discover the completion, structured-output, embedding, utility, and "
    "aggregate functions it provides."
)

_CATALOG_DOC_MD = (
    "# AI SQL for DuckDB\n\n"
    "**Call LLMs and embed text directly in DuckDB SQL.** `vgi-llm` brings Snowflake Cortex "
    "AISQL-style functions -- completion, classification, filtering, extraction, sentiment, "
    "summarization, and group-level map-reduce -- to DuckDB over a pluggable provider (Anthropic, "
    "OpenRouter, OpenAI, or a local Ollama), plus **keyless** local embeddings and cosine "
    "similarity.\n\n"
    "## Keyless first\n\n"
    "```sql\n"
    "-- No API key required: local ONNX embeddings + similarity\n"
    "SELECT ai_similarity(ai_embed('cat'), ai_embed('kitten')) AS score;\n"
    "```\n\n"
    "Add one OpenRouter key (or run Ollama locally) to unlock cloud/local completions:\n\n"
    "```sql\n"
    "CREATE SECRET (TYPE llm, openrouter_api_key 'sk-or-...');\n"
    "SELECT ai_complete('Write a haiku about DuckDB', 'openrouter/anthropic/claude-sonnet-5');\n"
    "```\n\n"
    "## Notes\n\n"
    "- The model argument routes by prefix (`anthropic/…`, `openrouter/…`, `openai/…`, `ollama/…`).\n"
    "- Every function returns NULL rather than erroring on empty input or a provider failure.\n"
    "- Embeddings run locally (fastembed/ONNX); pair with the DuckDB VSS extension."
)

_CATEGORIES = json.dumps(
    [
        {
            "name": "completion",
            "title": "Completion",
            "description": "Free-form LLM text generation and summarization over columns of prompts.",
        },
        {
            "name": "structured",
            "title": "Structured Output",
            "description": "Classification, boolean filtering, extraction, and sentiment as typed results.",
        },
        {
            "name": "embedding",
            "title": "Embedding & Similarity",
            "description": "Keyless local text embeddings and cosine similarity for semantic search / RAG.",
        },
        {
            "name": "aggregate",
            "title": "Aggregate",
            "description": "Reduce a whole GROUP BY group to one answer with LLM map-reduce.",
        },
        {
            "name": "utility",
            "title": "Utility",
            "description": "Pure helpers: prompt templating and local token estimation (no model call).",
        },
    ]
)

_EXECUTABLE_EXAMPLES = json.dumps(
    [
        {
            "name": "embed_dimension",
            "description": "Embed a string into a fixed-length FLOAT[] with the local default model (keyless).",
            "sql": "SELECT len(llm.main.ai_embed('hello world')) AS dim",
        },
        {
            "name": "self_similarity",
            "description": "A phrase is maximally similar to itself (keyless).",
            "sql": (
                "SELECT ROUND(llm.main.ai_similarity("
                "llm.main.ai_embed('database'), llm.main.ai_embed('database')), 3) AS sim"
            ),
        },
        {
            "name": "prompt_template",
            "description": "Build a prompt string by positional substitution (pure, no model).",
            "sql": "SELECT llm.main.prompt('Translate {} into {}', 'hello', 'French') AS p",
        },
        {
            "name": "token_estimate",
            "description": "Estimate a text's token count locally (no model call).",
            "sql": "SELECT llm.main.ai_count_tokens('the quick brown fox') AS tokens",
        },
    ]
)

_AGENT_TEST_TASKS = json.dumps(
    [
        {
            "name": "embedding_dimension_is_384",
            "prompt": "Embed the word 'database' and confirm the vector has 384 dimensions. Return a single boolean.",
            "reference_sql": "SELECT len(llm.main.ai_embed('database')) = 384 AS is_384",
            "success_criteria": "Returns true; the default model produces 384-dim vectors.",
            "ignore_column_names": True,
        },
        {
            "name": "self_similarity_is_one",
            "prompt": (
                "Compute the cosine similarity of the embedding of 'database' with itself, rounded to 3 decimals."
            ),
            "reference_sql": (
                "SELECT ROUND(llm.main.ai_similarity(llm.main.ai_embed('database'), "
                "llm.main.ai_embed('database')), 3) AS sim"
            ),
            "success_criteria": "Returns 1.0; a vector is identical to itself.",
            "ignore_column_names": True,
        },
        {
            "name": "related_more_similar_than_unrelated",
            "prompt": "Is 'dog' more similar to 'puppy' than to 'airplane'? Return a single boolean.",
            "reference_sql": (
                "SELECT llm.main.ai_similarity(llm.main.ai_embed('dog'), llm.main.ai_embed('puppy')) "
                "> llm.main.ai_similarity(llm.main.ai_embed('dog'), llm.main.ai_embed('airplane')) AS related"
            ),
            "success_criteria": "Returns true; a related pair scores higher than an unrelated one.",
            "ignore_column_names": True,
        },
        {
            "name": "prompt_template_substitution",
            "prompt": "Use the prompt() function to render 'Hi {}' with the value 'Sam'.",
            "reference_sql": "SELECT llm.main.prompt('Hi {}', 'Sam') AS p",
            "success_criteria": "Returns 'Hi Sam' via positional template substitution.",
            "ignore_column_names": True,
        },
        {
            "name": "count_tokens_is_positive",
            "prompt": "Estimate the number of tokens in the text 'the quick brown fox' and confirm it is positive.",
            "reference_sql": "SELECT llm.main.ai_count_tokens('the quick brown fox') > 0 AS ok",
            "success_criteria": "Returns true; the local token estimate is a positive integer.",
            "ignore_column_names": True,
        },
        {
            "name": "complete_a_prompt",
            "prompt": "Ask the model to reply with the single word pong and confirm a non-empty answer comes back.",
            "reference_sql": "SELECT length(llm.main.ai_complete('Reply with the single word: pong')) > 0 AS ok",
            "success_criteria": "Returns true; ai_complete produced a non-empty completion (needs a provider key).",
            "ignore_column_names": True,
        },
        {
            "name": "completion_details_has_text",
            "prompt": "Get the completion details for a short prompt and confirm the text field is populated.",
            "reference_sql": "SELECT llm.main.ai_complete_details('Say hello').text IS NOT NULL AS ok",
            "success_criteria": "Returns true; the details struct carries the reply text (needs a provider key).",
            "ignore_column_names": True,
        },
        {
            "name": "describe_an_image",
            "prompt": "Describe an image BLOB with the multimodal completion function.",
            "reference_sql": (
                "SELECT llm.main.ai_complete_image('What is in this image?', '\\x89PNG'::BLOB) IS NOT NULL AS ok"
            ),
            "success_criteria": "Uses ai_complete_image over a prompt + image BLOB (needs a vision key).",
            "ignore_column_names": True,
        },
        {
            "name": "classify_text",
            "prompt": "Classify the text 'my card was declined' into one of billing, bug, or feature.",
            "reference_sql": (
                "SELECT llm.main.ai_classify('my card was declined', ['billing','bug','feature']).labels "
                "IS NOT NULL AS ok"
            ),
            "success_criteria": "Returns the chosen labels struct (needs a provider key).",
            "ignore_column_names": True,
        },
        {
            "name": "filter_text_predicate",
            "prompt": "Decide whether 'How do I reset my password?' is a question using ai_filter.",
            "reference_sql": (
                "SELECT llm.main.ai_filter('the text is a question', 'How do I reset my password?') IS NOT NULL AS ok"
            ),
            "success_criteria": "Returns a boolean for the predicate (needs a provider key).",
            "ignore_column_names": True,
        },
        {
            "name": "extract_structured_json",
            "prompt": "Extract the age from 'Bob is 42' as JSON with an integer age field.",
            "reference_sql": (
                "SELECT llm.main.ai_extract('Bob is 42', "
                '\'{"type":"object","properties":{"age":{"type":"integer"}}}\') IS NOT NULL AS ok'
            ),
            "success_criteria": "Returns a JSON string with the extracted field (needs a provider key).",
            "ignore_column_names": True,
        },
        {
            "name": "sentiment_overall",
            "prompt": "Analyze the sentiment of 'The food was great but service was slow' and read the overall label.",
            "reference_sql": (
                "SELECT llm.main.ai_sentiment('The food was great but service was slow').overall IS NOT NULL AS ok"
            ),
            "success_criteria": "Returns the overall sentiment label (needs a provider key).",
            "ignore_column_names": True,
        },
        {
            "name": "summarize_text",
            "prompt": "Summarize a short paragraph with ai_summarize.",
            "reference_sql": (
                "SELECT llm.main.ai_summarize('DuckDB is an in-process SQL OLAP database.') IS NOT NULL AS ok"
            ),
            "success_criteria": "Returns a summary string (needs a provider key).",
            "ignore_column_names": True,
        },
        {
            "name": "aggregate_group_with_task",
            "prompt": "Apply a task across a group's rows with ai_agg to get one answer per group.",
            "reference_sql": (
                "SELECT llm.main.ai_agg(comment, 'List the top complaints') AS answer "
                "FROM (VALUES ('too slow'), ('buggy UI')) AS t(comment)"
            ),
            "success_criteria": "Returns one aggregated answer for the group (needs a provider key).",
            "ignore_column_names": True,
        },
        {
            "name": "summarize_group",
            "prompt": "Summarize all of a group's rows into one summary with ai_summarize_agg.",
            "reference_sql": (
                "SELECT llm.main.ai_summarize_agg(note) AS summary "
                "FROM (VALUES ('login failed'), ('disk full')) AS t(note)"
            ),
            "success_criteria": "Returns one summary for the group (needs a provider key).",
            "ignore_column_names": True,
        },
    ]
)

_SCHEMA_EXAMPLE_QUERIES = json.dumps(
    [
        {
            "description": "Embed text into a FLOAT[] vector (keyless local ONNX).",
            "sql": "SELECT llm.main.ai_embed('hello world')",
        },
        {
            "description": "Cosine similarity of two embeddings (keyless).",
            "sql": "SELECT llm.main.ai_similarity(llm.main.ai_embed('cat'), llm.main.ai_embed('kitten'))",
        },
        {
            "description": "Build a prompt string by safe template substitution (pure, no model).",
            "sql": "SELECT llm.main.prompt('Summarize: {}', 'DuckDB is an in-process OLAP database')",
        },
        {
            "description": "Count tokens locally with tiktoken (no model call).",
            "sql": "SELECT llm.main.ai_count_tokens('the quick brown fox')",
        },
        {
            "description": "Complete a prompt with the default provider.",
            "sql": "SELECT llm.main.ai_complete('Write a haiku about DuckDB')",
        },
        {
            "description": "Classify text into a subset of the given categories.",
            "sql": "SELECT llm.main.ai_classify('my card was declined', ['billing','bug','feature']).labels",
        },
    ]
)

_CATALOG_TAGS = {
    "vgi.title": "AI SQL Functions (AISQL)",
    "vgi.doc_llm": _CATALOG_DOC_LLM,
    "vgi.doc_md": _CATALOG_DOC_MD,
    "vgi.keywords": json.dumps(
        [
            "ai",
            "llm",
            "aisql",
            "cortex",
            "complete",
            "classify",
            "filter",
            "extract",
            "sentiment",
            "summarize",
            "embeddings",
            "similarity",
            "anthropic",
            "openrouter",
            "openai",
            "ollama",
            "rag",
            "semantic search",
        ]
    ),
    "vgi.author": "Query.Farm",
    "vgi.copyright": "Copyright 2026 Query Farm LLC - https://query.farm",
    "vgi.license": "MIT",
    "vgi.support_contact": ISSUES,
    "vgi.support_policy_url": f"{REPO}/blob/main/README.md",
    "vgi.executable_examples": _EXECUTABLE_EXAMPLES,
    "vgi.agent_test_tasks": _AGENT_TEST_TASKS,
}

_SCHEMA_TAGS = {
    "vgi.title": "AISQL — main schema",
    "vgi.keywords": json.dumps(
        [
            "ai",
            "llm",
            "ai_complete",
            "ai_classify",
            "ai_filter",
            "ai_extract",
            "ai_sentiment",
            "ai_summarize",
            "ai_embed",
            "ai_similarity",
            "ai_agg",
            "embeddings",
            "similarity",
            "completion",
            "structured output",
            "semantic search",
            "rag",
        ]
    ),
    "vgi.doc_llm": (
        "## llm.main schema\n\n"
        "The single schema of the AI SQL worker. It groups the surface into a few concepts: free-form "
        "completion and summarization, structured outputs (classification, boolean filtering, "
        "extraction, sentiment), keyless local embeddings and cosine similarity for semantic "
        "search / RAG, group-level LLM map-reduce aggregates, and pure utilities (prompt templating "
        "and local token estimation). Completions route to a pluggable provider by a model-prefix "
        "string; embeddings and similarity run in-process with no key. List the schema to see the "
        "exact functions and signatures."
    ),
    "vgi.doc_md": (
        "# llm.main\n\n"
        "AI functions over Apache Arrow for DuckDB: completion, structured output, embeddings, "
        "aggregates, and utilities.\n\n"
        "## Overview\n\n"
        "Completions call a pluggable LLM provider (Anthropic / OpenRouter / OpenAI / Ollama) chosen "
        "by a model-prefix string. Embeddings and `ai_similarity` run locally (fastembed/ONNX) with no "
        "key. The aggregates reduce a whole group to one answer via chunked map-reduce.\n\n"
        "## Usage\n\n"
        "```sql\n"
        "SELECT ai_similarity(ai_embed('cat'), ai_embed('kitten')) AS score;\n"
        "```\n\n"
        "## Notes\n\n"
        "- Every function returns NULL rather than erroring on empty input or provider failure.\n"
        "- Embeddings are keyless; completions need a key (or keyless Ollama)."
    ),
    "domain": "artificial-intelligence",
    "category": "ai-sql",
    "topic": "llm-functions",
    "vgi.categories": _CATEGORIES,
    "vgi.example_queries": _SCHEMA_EXAMPLE_QUERIES,
}


def make_catalog() -> Catalog:
    """Build the ``llm`` catalog descriptor.

    Returns:
        The declarative :class:`~vgi.catalog.Catalog` registering every scalar
        and aggregate under the single ``llm.main`` schema.
    """
    return Catalog(
        name="llm",
        default_schema="main",
        comment=(
            f"AI SQL functions for DuckDB over a pluggable LLM provider "
            f"(Anthropic/OpenRouter/OpenAI/Ollama) + keyless local embeddings "
            f"({models.DEFAULT_MODEL}, {models.embedding_dim(None)}-dim)."
        ),
        source_url=REPO,
        tags=_CATALOG_TAGS,
        schemas=[
            Schema(
                name="main",
                comment="AISQL: completion, structured output, embeddings, aggregates, and utilities.",
                tags=_SCHEMA_TAGS,
                functions=[*SCALAR_FUNCTIONS, *AGGREGATE_FUNCTIONS],
            ),
        ],
    )
