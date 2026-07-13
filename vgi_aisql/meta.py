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
