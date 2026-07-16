# Copyright 2026 Query Farm LLC - https://query.farm

"""Per-object discovery/metadata tags shared by the AISQL scalars and aggregates.

The VGI strict lint profile gates a handful of per-object tags on *every*
function (and on the catalog/schema). This module builds them in one place so
the surface stays consistent:

- ``vgi.title`` (VGI124)    -- human-friendly display name (must NOT
  normalize-equal the machine name, or VGI125 fires).
- ``vgi.doc_llm`` (VGI112)  -- Markdown narrative aimed at an LLM/agent.
- ``vgi.doc_md`` (VGI113)   -- Markdown narrative aimed at human docs.
- ``vgi.keywords`` (VGI126/VGI138) -- a JSON array of search terms/synonyms.

``vgi.source_url`` is *not* emitted per object: provenance lives on the catalog
(``Catalog(source_url=...)``); repeating it on every object trips VGI139.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # pragma: no cover - typing only
    from vgi.metadata import FunctionExample


def example_queries_tag(examples: Sequence[FunctionExample]) -> str:
    """Serialize a ``FunctionExample`` list as a ``vgi.example_queries`` JSON list.

    The strict lint profile (VGI515) requires every function example to carry a
    description. The VGI extension fills the native ``duckdb_functions().examples``
    column from ``Meta.examples`` as bare SQL strings (no description); emitting
    the same SQL here *with* its description lets vgi-lint dedupe by normalized
    SQL and keep the described entry, so every example is described.

    Args:
        examples: The function's ``Meta.examples`` list.

    Returns:
        A JSON array string of ``{"description", "sql"}`` objects.
    """
    return json.dumps([{"description": e.description, "sql": e.sql} for e in examples])


def apply_combined_example_queries(functions: Sequence[Any]) -> None:
    """Give every overload of a function name the SAME described-example tag.

    The VGI extension surfaces a *single* overload's ``tags`` (hence one
    ``vgi.example_queries``) for **every** ``duckdb_functions()`` row of a name,
    while the native description-less ``examples`` column is per-overload. So each
    overload must advertise **all** overloads' example SQL for vgi-lint to dedupe
    (VGI515) every overload's native example against a described tag entry. This
    sets, on every class sharing a ``Meta.name``, a combined ``vgi.example_queries``
    built from the union of all those overloads' ``Meta.examples`` (deduped by
    normalized SQL).

    Args:
        functions: The function classes to post-process (a worker's scalar or
            aggregate registry).
    """
    by_name: dict[str, list[Any]] = {}
    for cls in functions:
        by_name.setdefault(cls.Meta.name, []).append(cls)
    for classes in by_name.values():
        combined: list[Any] = []
        seen: set[str] = set()
        for cls in classes:
            for example in getattr(cls.Meta, "examples", None) or []:
                key = " ".join(example.sql.split()).lower()
                if key in seen:
                    continue
                seen.add(key)
                combined.append(example)
        tag = example_queries_tag(combined)
        for cls in classes:
            cls.Meta.tags = {**cls.Meta.tags, "vgi.example_queries": tag}


def keywords_json(keywords: Sequence[str]) -> str:
    """Serialize keywords as a ``vgi.keywords`` JSON array string.

    Args:
        keywords: Search terms / synonyms.

    Returns:
        The JSON-array string form the strict lint profile requires.
    """
    return json.dumps(list(keywords))


def object_tags(
    *,
    title: str,
    doc_llm: str,
    doc_md: str,
    keywords: Sequence[str],
    category: str | None = None,
) -> dict[str, str]:
    """Assemble the per-object VGI124/112/113/126/138 tag set.

    Args:
        title: Human-friendly display name (must not normalize-equal the name).
        doc_llm: Markdown narrative aimed at an LLM/agent.
        doc_md: Markdown narrative aimed at human documentation.
        keywords: Search terms / synonyms, serialized to a JSON array.
        category: One of the schema's ``vgi.categories`` registry entries; every
            function carries exactly one so navigation/SEO listings stay filled.

    Returns:
        The assembled tag dictionary.
    """
    tags = {
        "vgi.title": title,
        "vgi.doc_llm": doc_llm,
        "vgi.doc_md": doc_md,
        "vgi.keywords": keywords_json(keywords),
    }
    if category is not None:
        tags["vgi.category"] = category
    return tags
