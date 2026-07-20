# Copyright 2026 Query Farm LLC - https://query.farm

"""AISQL scalar functions: per-row LLM completion, classification, and more.

Each function maps a column of input to a column of output, one provider call
per row (fanned across a bounded thread pool by :mod:`vgi_llm.engine`). The
contract everywhere is the same, and it is fail-loud, not fail-quiet:

- NULL or empty/whitespace-only input yields a NULL output row (no model call).
- A provider failure -- including a missing API key -- **raises** a DuckDB error;
  it is never swallowed to NULL. (The keyless ``ai_embed`` / ``ai_similarity`` /
  ``prompt`` / ``ai_count_tokens`` make no provider call, so they cannot fail this
  way.)
- STRUCT-returning functions parse the model's JSON; an unparseable reply
  **raises** (via ``_parse_or_raise``). ``ai_filter`` is the one exception: an
  unparseable yes/no answer coerces to NULL rather than raising.

Scalar functions are **positional-only** in VGI/DuckDB (the ``name := value``
named-argument syntax is a table-function feature), so the optional ``model``
argument is exposed as a second *arity overload* sharing the SQL name -- e.g.
``ai_complete(prompt)`` and ``ai_complete(prompt, model)`` -- exactly as
``vgi-embed`` does for ``embed``. ``ai_embed`` / ``ai_similarity`` are keyless
(local ONNX via :mod:`vgi_llm.models`); ``prompt`` and ``ai_count_tokens`` are
pure and never call a model.
"""

from __future__ import annotations

import functools
import json
from typing import TYPE_CHECKING, Annotated, Any

import pyarrow as pa
from vgi.arguments import ConstParam, Param, Returns, Secret, Setting
from vgi.metadata import FunctionExample
from vgi.scalar_function import ScalarFunction

from vgi_llm import engine, meta, models
from vgi_llm.providers import ImagePart, Message, ProviderError, ResponseFormat


def _parse_or_raise(func_name: str, completion: Any) -> dict[str, Any] | None:
    """Parse a JSON object from a completion, raising on non-JSON output.

    A ``None`` completion (a blank input row) returns None. A real completion
    whose text does not parse to a JSON object **raises** (surfaces as a DuckDB
    error naming the function and a snippet) -- no more silent all-NULL columns.

    Args:
        func_name: The SQL function name, for the error message.
        completion: The provider :class:`Completion`, or None for a blank row.

    Returns:
        The parsed object, or None for a blank input row.

    Raises:
        ProviderError: If a real completion did not parse to a JSON object.
    """
    if completion is None:
        return None
    obj = engine.parse_json_object(completion.text)
    if obj is None:
        snippet = (completion.text or "")[:200]
        raise ProviderError(f"{func_name}: expected a JSON object from the model, got: {snippet!r}")
    return obj


# pyarrow's array classes are generic to the type-checker (pyarrow-stubs) but NOT
# subscriptable at runtime, and the framework evaluates these annotations at import
# via ``get_type_hints``. Alias them so mypy sees the parametrised form while the
# runtime evaluation resolves to the bare, subscript-free class.
if TYPE_CHECKING:
    from typing import TypeAlias

    _ListArray: TypeAlias = pa.ListArray[Any]  # noqa: UP040 - runtime needs the non-subscript form below
    _Array: TypeAlias = pa.Array[Any]  # noqa: UP040 - runtime needs the non-subscript form below
else:
    _ListArray = pa.ListArray
    _Array = pa.Array

# ---------------------------------------------------------------------------
# Shared Arrow return types (fixed at bind).
# ---------------------------------------------------------------------------

_VECTOR = pa.list_(pa.float32())

_DETAILS_STRUCT = pa.struct(
    [
        ("text", pa.string()),
        ("model", pa.string()),
        ("input_tokens", pa.int64()),
        ("output_tokens", pa.int64()),
        ("finish_reason", pa.string()),
    ]
)

_CLASSIFY_STRUCT = pa.struct([("labels", pa.list_(pa.string()))])

_SENTIMENT_CATEGORY = pa.struct([("name", pa.string()), ("sentiment", pa.string())])
_SENTIMENT_STRUCT = pa.struct(
    [
        ("overall", pa.string()),
        ("categories", pa.list_(_SENTIMENT_CATEGORY)),
    ]
)

_MODEL_DOC = "Provider-prefixed model to route to; the leading path segment picks the backend, empty uses the default."


def _ex(sql: str, description: str) -> list[FunctionExample]:
    """Build a one-element example list.

    Args:
        sql: The example SQL.
        description: What the example demonstrates.

    Returns:
        A single-element ``FunctionExample`` list.
    """
    return [FunctionExample(sql=sql, description=description)]


def _guess_media_type(data: bytes) -> str:
    """Sniff an image media type from magic bytes (defaults to image/png).

    Args:
        data: The raw image bytes.

    Returns:
        The guessed IANA media type.
    """
    if data.startswith(b"\xff\xd8\xff"):
        return "image/jpeg"
    if data.startswith(b"GIF8"):
        return "image/gif"
    if data.startswith(b"RIFF") and data[8:12] == b"WEBP":
        return "image/webp"
    return "image/png"


# ===========================================================================
# ai_complete -- text completion
# ===========================================================================

_COMPLETE_TAGS = meta.object_tags(
    title="AI Text Completion",
    doc_llm=(
        "## ai_complete(prompt[, model])\n\n"
        "Send each row's `prompt` to a large language model and return its text "
        "reply as a `VARCHAR`. The optional second argument selects the "
        "provider/model (`anthropic/…`, `openrouter/…`, `openai/…`, `ollama/…`, "
        "or a bare model id for the default provider); omit it to use the "
        "configured default.\n\n"
        "**When to use.** Free-form generation, rewriting, or answering over a "
        "column of prompts. For a structured result use `ai_classify` / "
        "`ai_extract` / `ai_sentiment`; for a yes/no filter use `ai_filter`.\n\n"
        "**Input/output.** Input: one `VARCHAR` prompt per row. Output: one "
        "`VARCHAR` per row. NULL/empty prompt -> NULL (no model call); a provider "
        "error or a missing key **raises** a DuckDB error -- it is not swallowed "
        "to NULL. Calls are fanned across a bounded thread pool, so a wide column "
        "completes concurrently."
    ),
    doc_md=(
        "# ai_complete\n\n"
        "Per-row LLM text completion returning `VARCHAR`.\n\n"
        "## Notes\n\n"
        "- The model argument is positional; omit it for the default provider.\n"
        "- NULL/empty input -> NULL; a provider failure raises a DuckDB error.\n"
        "- Configure keys with an `llm` secret or provider env vars."
    ),
    keywords=[
        "ai",
        "llm",
        "complete",
        "completion",
        "generate",
        "text generation",
        "prompt",
        "chat",
        "claude",
        "openai",
        "openrouter",
        "ollama",
    ],
    category="completion",
)


def _run(
    prompts: list[str | None],
    *,
    build_messages: Any,
    model: str,
    llm_secret: dict[str, Any] | None,
    settings: engine.RuntimeSettings | None,
    response_format: ResponseFormat | None = None,
) -> list[Any]:
    """Dispatch a column of prompts through the engine, applying the settings.

    Centralises the ``RuntimeSettings`` -> provider-call wiring (effective model,
    sampling params, worker cap, timeout) shared by every LLM scalar.

    Args:
        prompts: The per-row prompt texts (blank rows yield None).
        build_messages: Turns one prompt into the provider message list.
        model: The per-call model argument ('' for the settings/provider default).
        llm_secret: Resolved unified ``llm`` secret fields, or None.
        settings: The resolved ``llm_*`` settings, or None for defaults.
        response_format: Optional structured-output JSON Schema request.

    Returns:
        One ``Completion`` (or None for a blank row) per input row.
    """
    s = settings or engine.RuntimeSettings()
    return engine.map_complete(
        prompts,
        build_messages=build_messages,
        model=s.effective_model(model),
        secrets=engine.build_secrets(llm_secret),
        params=s.completion_params(),
        response_format=response_format,
        max_workers=s.workers(),
        timeout=s.timeout_value(),
    )


def _complete(
    prompt: pa.StringArray,
    *,
    model: str,
    llm_secret: dict[str, Any] | None,
    settings: engine.RuntimeSettings | None = None,
) -> pa.StringArray:
    """Complete a column of prompts to a column of reply strings.

    Args:
        prompt: The per-row prompt texts.
        model: The model routing string ('' for default).
        llm_secret: Resolved unified ``llm`` secret fields, or None.
        settings: The resolved ``llm_*`` settings, or None for defaults.

    Returns:
        One VARCHAR reply (or NULL) per row.
    """
    completions = _run(
        prompt.to_pylist(),
        build_messages=engine.user_message,
        model=model,
        llm_secret=llm_secret,
        settings=settings,
    )
    return pa.array([c.text if c is not None else None for c in completions], type=pa.string())


class AiComplete(ScalarFunction):
    """``ai_complete(prompt)`` -- text completion with the default model."""

    class Meta:
        """Declarative metadata for ``ai_complete(prompt)``."""

        name = "ai_complete"
        description = "LLM text completion for each prompt row (default model); NULL on empty input, errors raise."
        categories = ["completion"]
        required_secrets = ["llm"]
        examples = _ex(
            "SELECT llm.main.ai_complete('Write a haiku about DuckDB')", "Complete a prompt with the default model"
        )
        tags = {**_COMPLETE_TAGS, "vgi.example_queries": meta.example_queries_tag(examples)}

    @classmethod
    def compute(
        cls,
        prompt: Annotated[pa.StringArray, Param(doc="Prompt text to complete.")],
        llm_secret: Annotated[dict[str, pa.Scalar[Any]] | None, Secret("llm")] = None,
        llm_max_tokens: Annotated[pa.Scalar[Any] | None, Setting()] = None,
        llm_temperature: Annotated[pa.Scalar[Any] | None, Setting()] = None,
        llm_top_p: Annotated[pa.Scalar[Any] | None, Setting()] = None,
        llm_model: Annotated[pa.Scalar[Any] | None, Setting()] = None,
        llm_max_workers: Annotated[pa.Scalar[Any] | None, Setting()] = None,
        llm_timeout: Annotated[pa.Scalar[Any] | None, Setting()] = None,
    ) -> Annotated[pa.StringArray, Returns()]:
        """Complete each prompt with the default model."""
        return _complete(
            prompt,
            model="",
            llm_secret=llm_secret,
            settings=engine.read_settings(
                llm_max_tokens=llm_max_tokens,
                llm_temperature=llm_temperature,
                llm_top_p=llm_top_p,
                llm_model=llm_model,
                llm_max_workers=llm_max_workers,
                llm_timeout=llm_timeout,
            ),
        )


class AiCompleteModel(ScalarFunction):
    """``ai_complete(prompt, model)`` -- text completion with an explicit model."""

    class Meta:
        """Declarative metadata for ``ai_complete(prompt, model)``."""

        name = "ai_complete"
        description = "LLM text completion for each prompt row with an explicit provider/model."
        categories = ["completion"]
        required_secrets = ["llm"]
        examples = _ex(
            "SELECT llm.main.ai_complete('Summarize DuckDB in one line', 'ollama/llama3.2')",
            "Complete a prompt with an explicit provider/model",
        )
        tags = {**_COMPLETE_TAGS, "vgi.example_queries": meta.example_queries_tag(examples)}

    @classmethod
    def compute(
        cls,
        prompt: Annotated[pa.StringArray, Param(doc="Prompt text to complete.")],
        model: Annotated[str, ConstParam(_MODEL_DOC)],
        llm_secret: Annotated[dict[str, pa.Scalar[Any]] | None, Secret("llm")] = None,
        llm_max_tokens: Annotated[pa.Scalar[Any] | None, Setting()] = None,
        llm_temperature: Annotated[pa.Scalar[Any] | None, Setting()] = None,
        llm_top_p: Annotated[pa.Scalar[Any] | None, Setting()] = None,
        llm_model: Annotated[pa.Scalar[Any] | None, Setting()] = None,
        llm_max_workers: Annotated[pa.Scalar[Any] | None, Setting()] = None,
        llm_timeout: Annotated[pa.Scalar[Any] | None, Setting()] = None,
    ) -> Annotated[pa.StringArray, Returns()]:
        """Complete each prompt with the explicit ``model``."""
        return _complete(
            prompt,
            model=model,
            llm_secret=llm_secret,
            settings=engine.read_settings(
                llm_max_tokens=llm_max_tokens,
                llm_temperature=llm_temperature,
                llm_top_p=llm_top_p,
                llm_model=llm_model,
                llm_max_workers=llm_max_workers,
                llm_timeout=llm_timeout,
            ),
        )


# ===========================================================================
# ai_complete_details -- completion + token/metadata envelope (STRUCT)
# ===========================================================================

_DETAILS_TAGS = meta.object_tags(
    title="AI Completion With Details",
    doc_llm=(
        "## ai_complete_details(prompt[, model])\n\n"
        "Like `ai_complete`, but returns a `STRUCT` envelope with the reply plus "
        "provider metadata: `{text VARCHAR, model VARCHAR, input_tokens BIGINT, "
        "output_tokens BIGINT, finish_reason VARCHAR}`. Use it when you need the "
        "resolved model name, token accounting (cost/telemetry), or the stop "
        "reason alongside the text.\n\n"
        "**Input/output.** Input: one `VARCHAR` prompt per row (+ optional model). "
        "Output: one `STRUCT` per row; NULL/empty prompt -> NULL row (no model "
        "call). A provider failure **raises** a DuckDB error. Read a field with "
        "dot access, e.g. `ai_complete_details(p).output_tokens`."
    ),
    doc_md=(
        "# ai_complete_details\n\n"
        "LLM completion returning a `STRUCT` with the text and token/metadata "
        "envelope.\n\n"
        "## Notes\n\n"
        "- Fields: `text`, `model`, `input_tokens`, `output_tokens`, `finish_reason`.\n"
        "- NULL/empty input -> NULL struct; a provider failure raises a DuckDB error."
    ),
    keywords=[
        "ai",
        "llm",
        "completion",
        "details",
        "tokens",
        "usage",
        "metadata",
        "finish reason",
        "struct",
        "telemetry",
    ],
    category="completion",
)


def _complete_details(
    prompt: pa.StringArray,
    *,
    model: str,
    llm_secret: dict[str, Any] | None,
    settings: engine.RuntimeSettings | None = None,
) -> pa.StructArray:
    """Complete prompts and wrap each reply in the details STRUCT.

    Args:
        prompt: The per-row prompt texts.
        model: The model routing string ('' for default).
        llm_secret: Resolved unified ``llm`` secret fields, or None.
        settings: The resolved ``llm_*`` settings, or None for defaults.

    Returns:
        One details STRUCT (or NULL) per row.
    """
    completions = _run(
        prompt.to_pylist(),
        build_messages=engine.user_message,
        model=model,
        llm_secret=llm_secret,
        settings=settings,
    )
    rows: list[dict[str, Any] | None] = []
    for c in completions:
        if c is None:
            rows.append(None)
        else:
            rows.append(
                {
                    "text": c.text,
                    "model": c.model,
                    "input_tokens": c.usage.input_tokens,
                    "output_tokens": c.usage.output_tokens,
                    "finish_reason": c.finish_reason,
                }
            )
    return pa.array(rows, type=_DETAILS_STRUCT)  # type: ignore[return-value]


class AiCompleteDetails(ScalarFunction):
    """``ai_complete_details(prompt)`` -- completion + metadata STRUCT (default model)."""

    class Meta:
        """Declarative metadata for ``ai_complete_details(prompt)``."""

        name = "ai_complete_details"
        description = "LLM completion returning a STRUCT of text + model + token usage + finish reason."
        categories = ["completion"]
        required_secrets = ["llm"]
        examples = _ex(
            "SELECT llm.main.ai_complete_details('Explain MVCC in one sentence').text",
            "Completion with token/metadata envelope",
        )
        tags = {**_DETAILS_TAGS, "vgi.example_queries": meta.example_queries_tag(examples)}

    @classmethod
    def compute(
        cls,
        prompt: Annotated[pa.StringArray, Param(doc="Prompt text to complete.")],
        llm_secret: Annotated[dict[str, pa.Scalar[Any]] | None, Secret("llm")] = None,
        llm_max_tokens: Annotated[pa.Scalar[Any] | None, Setting()] = None,
        llm_temperature: Annotated[pa.Scalar[Any] | None, Setting()] = None,
        llm_top_p: Annotated[pa.Scalar[Any] | None, Setting()] = None,
        llm_model: Annotated[pa.Scalar[Any] | None, Setting()] = None,
        llm_max_workers: Annotated[pa.Scalar[Any] | None, Setting()] = None,
        llm_timeout: Annotated[pa.Scalar[Any] | None, Setting()] = None,
    ) -> Annotated[pa.StructArray, Returns(arrow_type=_DETAILS_STRUCT)]:
        """Complete each prompt and return the details struct."""
        return _complete_details(
            prompt,
            model="",
            llm_secret=llm_secret,
            settings=engine.read_settings(
                llm_max_tokens=llm_max_tokens,
                llm_temperature=llm_temperature,
                llm_top_p=llm_top_p,
                llm_model=llm_model,
                llm_max_workers=llm_max_workers,
                llm_timeout=llm_timeout,
            ),
        )


class AiCompleteDetailsModel(ScalarFunction):
    """``ai_complete_details(prompt, model)`` -- completion + metadata STRUCT (explicit model)."""

    class Meta:
        """Declarative metadata for ``ai_complete_details(prompt, model)``."""

        name = "ai_complete_details"
        description = "LLM completion (explicit model) returning a STRUCT of text + model + usage + finish reason."
        categories = ["completion"]
        required_secrets = ["llm"]
        examples = _ex(
            "SELECT llm.main.ai_complete_details('Explain MVCC', 'openrouter/anthropic/claude-sonnet-5').output_tokens",
            "Completion details with an explicit provider/model",
        )
        tags = {**_DETAILS_TAGS, "vgi.example_queries": meta.example_queries_tag(examples)}

    @classmethod
    def compute(
        cls,
        prompt: Annotated[pa.StringArray, Param(doc="Prompt text to complete.")],
        model: Annotated[str, ConstParam(_MODEL_DOC)],
        llm_secret: Annotated[dict[str, pa.Scalar[Any]] | None, Secret("llm")] = None,
        llm_max_tokens: Annotated[pa.Scalar[Any] | None, Setting()] = None,
        llm_temperature: Annotated[pa.Scalar[Any] | None, Setting()] = None,
        llm_top_p: Annotated[pa.Scalar[Any] | None, Setting()] = None,
        llm_model: Annotated[pa.Scalar[Any] | None, Setting()] = None,
        llm_max_workers: Annotated[pa.Scalar[Any] | None, Setting()] = None,
        llm_timeout: Annotated[pa.Scalar[Any] | None, Setting()] = None,
    ) -> Annotated[pa.StructArray, Returns(arrow_type=_DETAILS_STRUCT)]:
        """Complete each prompt with ``model`` and return the details struct."""
        return _complete_details(
            prompt,
            model=model,
            llm_secret=llm_secret,
            settings=engine.read_settings(
                llm_max_tokens=llm_max_tokens,
                llm_temperature=llm_temperature,
                llm_top_p=llm_top_p,
                llm_model=llm_model,
                llm_max_workers=llm_max_workers,
                llm_timeout=llm_timeout,
            ),
        )


# ===========================================================================
# ai_complete_image -- multimodal completion (prompt + BLOB image)
# ===========================================================================

_IMAGE_TAGS = meta.object_tags(
    title="AI Image Completion",
    doc_llm=(
        "## ai_complete_image(prompt, image[, model])\n\n"
        "Multimodal completion: send a text `prompt` together with an `image` "
        "(a `BLOB` of PNG/JPEG/GIF/WebP bytes) to a vision-capable model and "
        "return the text reply as `VARCHAR`. Use it to caption, describe, OCR, or "
        "answer questions about images stored in a column.\n\n"
        "**Input/output.** Inputs: `VARCHAR` prompt, `BLOB` image, optional model. "
        "Output: one `VARCHAR` per row. NULL/empty prompt or NULL image -> NULL "
        "(no model call); a provider failure (including a non-vision model or a "
        "missing key) **raises** a DuckDB error. The media type is sniffed from "
        "the image's magic bytes."
    ),
    doc_md=(
        "# ai_complete_image\n\n"
        "Multimodal LLM completion over a text prompt + an image `BLOB`.\n\n"
        "## Notes\n\n"
        "- The image is a `BLOB` (PNG/JPEG/GIF/WebP); media type is auto-detected.\n"
        "- Requires a vision-capable model; NULL input -> NULL, a provider failure raises."
    ),
    keywords=[
        "ai",
        "llm",
        "vision",
        "multimodal",
        "image",
        "caption",
        "describe",
        "ocr",
        "blob",
        "gpt-4o",
        "claude",
    ],
    category="completion",
)


def _complete_image(
    prompt: pa.StringArray,
    image: pa.BinaryArray,
    *,
    model: str,
    llm_secret: dict[str, Any] | None,
    settings: engine.RuntimeSettings | None = None,
) -> pa.StringArray:
    """Complete each (prompt, image) pair to a reply string.

    Args:
        prompt: The per-row prompt texts.
        image: The per-row image BLOBs.
        model: The model routing string ('' for default).
        llm_secret: Resolved unified ``llm`` secret fields, or None.
        settings: The resolved ``llm_*`` settings, or None for defaults.

    Returns:
        One VARCHAR reply (or NULL) per row.
    """
    prompts = prompt.to_pylist()
    images = image.to_pylist()
    n = len(prompts)
    out: list[str | None] = [None] * n

    # A row is live only when it has both a non-empty prompt and an image blob.
    # We encode the row index as the "prompt" the engine dispatches, and look the
    # real prompt/blob up per row -- so two rows with identical prompt text but
    # different images still get distinct messages.
    live: dict[int, tuple[str, bytes]] = {}
    for i in range(n):
        text = prompts[i]
        blob = images[i]
        if text is not None and text.strip() and blob is not None:
            live[i] = (text, blob)
    if not live:
        return pa.array(out, type=pa.string())

    def _messages_for(index: int) -> list[Message]:
        text, blob = live[index]
        part = ImagePart(data=blob, media_type=_guess_media_type(blob))
        return [Message(role="user", content=[text, part])]

    order = sorted(live)
    completions = _run(
        [str(i) for i in order],
        build_messages=lambda enc: _messages_for(int(enc)),
        model=model,
        llm_secret=llm_secret,
        settings=settings,
    )
    for idx, completion in zip(order, completions, strict=True):
        out[idx] = completion.text if completion is not None else None
    return pa.array(out, type=pa.string())


class AiCompleteImage(ScalarFunction):
    """``ai_complete_image(prompt, image)`` -- multimodal completion (default model)."""

    class Meta:
        """Declarative metadata for ``ai_complete_image(prompt, image)``."""

        name = "ai_complete_image"
        description = "Multimodal LLM completion over a text prompt + an image BLOB (default model)."
        categories = ["completion"]
        required_secrets = ["llm"]
        examples = _ex(
            "SELECT llm.main.ai_complete_image('What is in this image?', '\x89PNG'::BLOB)",
            "Describe an image column with the default model",
        )
        tags = {**_IMAGE_TAGS, "vgi.example_queries": meta.example_queries_tag(examples)}

    @classmethod
    def compute(
        cls,
        prompt: Annotated[pa.StringArray, Param(doc="Instruction/question about the image.")],
        image: Annotated[
            pa.BinaryArray, Param(arrow_type=pa.binary(), doc="Image bytes to analyze (PNG/JPEG/GIF/WebP).")
        ],
        llm_secret: Annotated[dict[str, pa.Scalar[Any]] | None, Secret("llm")] = None,
        llm_max_tokens: Annotated[pa.Scalar[Any] | None, Setting()] = None,
        llm_temperature: Annotated[pa.Scalar[Any] | None, Setting()] = None,
        llm_top_p: Annotated[pa.Scalar[Any] | None, Setting()] = None,
        llm_model: Annotated[pa.Scalar[Any] | None, Setting()] = None,
        llm_max_workers: Annotated[pa.Scalar[Any] | None, Setting()] = None,
        llm_timeout: Annotated[pa.Scalar[Any] | None, Setting()] = None,
    ) -> Annotated[pa.StringArray, Returns()]:
        """Answer the prompt about each image with the default model."""
        return _complete_image(
            prompt,
            image,
            model="",
            llm_secret=llm_secret,
            settings=engine.read_settings(
                llm_max_tokens=llm_max_tokens,
                llm_temperature=llm_temperature,
                llm_top_p=llm_top_p,
                llm_model=llm_model,
                llm_max_workers=llm_max_workers,
                llm_timeout=llm_timeout,
            ),
        )


class AiCompleteImageModel(ScalarFunction):
    """``ai_complete_image(prompt, image, model)`` -- multimodal completion (explicit model)."""

    class Meta:
        """Declarative metadata for ``ai_complete_image(prompt, image, model)``."""

        name = "ai_complete_image"
        description = "Multimodal LLM completion over a text prompt + an image BLOB with an explicit model."
        categories = ["completion"]
        required_secrets = ["llm"]
        examples = _ex(
            "SELECT llm.main.ai_complete_image('Describe it', '\x89PNG'::BLOB, 'openai/gpt-4o')",
            "Describe an image with an explicit vision model",
        )
        tags = {**_IMAGE_TAGS, "vgi.example_queries": meta.example_queries_tag(examples)}

    @classmethod
    def compute(
        cls,
        prompt: Annotated[pa.StringArray, Param(doc="Instruction/question about the image.")],
        image: Annotated[
            pa.BinaryArray, Param(arrow_type=pa.binary(), doc="Image bytes to analyze (PNG/JPEG/GIF/WebP).")
        ],
        model: Annotated[str, ConstParam(_MODEL_DOC)],
        llm_secret: Annotated[dict[str, pa.Scalar[Any]] | None, Secret("llm")] = None,
        llm_max_tokens: Annotated[pa.Scalar[Any] | None, Setting()] = None,
        llm_temperature: Annotated[pa.Scalar[Any] | None, Setting()] = None,
        llm_top_p: Annotated[pa.Scalar[Any] | None, Setting()] = None,
        llm_model: Annotated[pa.Scalar[Any] | None, Setting()] = None,
        llm_max_workers: Annotated[pa.Scalar[Any] | None, Setting()] = None,
        llm_timeout: Annotated[pa.Scalar[Any] | None, Setting()] = None,
    ) -> Annotated[pa.StringArray, Returns()]:
        """Answer the prompt about each image with the explicit ``model``."""
        return _complete_image(
            prompt,
            image,
            model=model,
            llm_secret=llm_secret,
            settings=engine.read_settings(
                llm_max_tokens=llm_max_tokens,
                llm_temperature=llm_temperature,
                llm_top_p=llm_top_p,
                llm_model=llm_model,
                llm_max_workers=llm_max_workers,
                llm_timeout=llm_timeout,
            ),
        )


# ===========================================================================
# ai_classify -- multi-label classification (STRUCT{labels})
# ===========================================================================

_CLASSIFY_TAGS = meta.object_tags(
    title="AI Classification",
    doc_llm=(
        "## ai_classify(input, categories[, model])\n\n"
        "Classify each `input` text into one or more of the supplied "
        "`categories` (a `LIST<VARCHAR>`) and return `STRUCT{labels "
        "LIST<VARCHAR>}` -- the chosen labels drawn from your category set. The "
        "model is asked for strict JSON matching the struct; an unparseable "
        "reply -> NULL row.\n\n"
        "**When to use.** Route/triage a column of text against a fixed taxonomy "
        "(support tickets, intents, topics). For a boolean keep/drop decision use "
        "`ai_filter`; for free-form fields use `ai_extract`.\n\n"
        "**Input/output.** Inputs: `VARCHAR` text, `LIST<VARCHAR>` categories, "
        "optional model. Output: `STRUCT` with a `labels` list. NULL/empty input -> "
        "NULL (no model call); a provider failure, or a reply that is not the "
        "expected JSON, **raises** a DuckDB error."
    ),
    doc_md=(
        "# ai_classify\n\n"
        "Multi-label text classification against a category list, returning "
        "`STRUCT{labels LIST<VARCHAR>}`.\n\n"
        "## Notes\n\n"
        "- `categories` is a `LIST<VARCHAR>`; labels are drawn from it.\n"
        "- NULL/empty input -> NULL; a provider failure or unparseable reply raises."
    ),
    keywords=[
        "ai",
        "llm",
        "classify",
        "classification",
        "label",
        "labels",
        "category",
        "categories",
        "taxonomy",
        "routing",
        "triage",
    ],
    category="structured",
)

_CLASSIFY_SCHEMA = {
    "type": "object",
    "properties": {"labels": {"type": "array", "items": {"type": "string"}}},
    "required": ["labels"],
    "additionalProperties": False,
}


def _classify(
    text: pa.StringArray,
    categories: list[str],
    *,
    model: str,
    llm_secret: dict[str, Any] | None,
    settings: engine.RuntimeSettings | None = None,
) -> pa.StructArray:
    """Classify each text into a subset of ``categories``.

    Args:
        text: The per-row input texts.
        categories: The allowed category labels.
        model: The model routing string ('' for default).
        llm_secret: Resolved unified ``llm`` secret fields, or None.
        settings: The resolved ``llm_*`` settings, or None for defaults.

    Returns:
        One ``STRUCT{labels}`` (or NULL) per row.

    Raises:
        ProviderError: If the model's reply is not a JSON object with ``labels``.
    """
    cat_list = [c for c in (categories or []) if c]
    system = (
        "You are a text classifier. Choose every category from the allowed list that applies to the "
        'user\'s text. Reply ONLY with a JSON object of the form {"labels": ["..."]} (no prose, no code '
        f"fence) whose labels are drawn verbatim from this list: {cat_list}."
    )
    completions = _run(
        text.to_pylist(),
        build_messages=lambda p: engine.system_user_messages(system, p),
        model=model,
        llm_secret=llm_secret,
        settings=settings,
        response_format=ResponseFormat(json_schema=_CLASSIFY_SCHEMA, name="classification"),
    )
    allowed = set(cat_list)
    rows: list[dict[str, Any] | None] = []
    for c in completions:
        obj = _parse_or_raise("ai_classify", c)
        if obj is None:
            rows.append(None)
            continue
        if not isinstance(obj.get("labels"), list):
            raise ProviderError(f"ai_classify: JSON is missing a 'labels' array: {obj!r}")
        labels = [str(x) for x in obj["labels"] if not allowed or str(x) in allowed]
        rows.append({"labels": labels})
    return pa.array(rows, type=_CLASSIFY_STRUCT)  # type: ignore[return-value]


class AiClassify(ScalarFunction):
    """``ai_classify(input, categories)`` -- multi-label classification (default model)."""

    class Meta:
        """Declarative metadata for ``ai_classify(input, categories)``."""

        name = "ai_classify"
        description = "Classify text into a subset of the given categories; returns STRUCT{labels LIST<VARCHAR>}."
        categories = ["structured"]
        required_secrets = ["llm"]
        examples = _ex(
            "SELECT llm.main.ai_classify('my card was declined', ['billing','bug','feature']).labels",
            "Classify text against a category list",
        )
        tags = {**_CLASSIFY_TAGS, "vgi.example_queries": meta.example_queries_tag(examples)}

    @classmethod
    def compute(
        cls,
        input: Annotated[pa.StringArray, Param(doc="Text to classify.")],
        categories: Annotated[
            _ListArray, Param(arrow_type=pa.list_(pa.string()), doc="The set of category labels to choose from.")
        ],
        llm_secret: Annotated[dict[str, pa.Scalar[Any]] | None, Secret("llm")] = None,
        llm_max_tokens: Annotated[pa.Scalar[Any] | None, Setting()] = None,
        llm_temperature: Annotated[pa.Scalar[Any] | None, Setting()] = None,
        llm_top_p: Annotated[pa.Scalar[Any] | None, Setting()] = None,
        llm_model: Annotated[pa.Scalar[Any] | None, Setting()] = None,
        llm_max_workers: Annotated[pa.Scalar[Any] | None, Setting()] = None,
        llm_timeout: Annotated[pa.Scalar[Any] | None, Setting()] = None,
    ) -> Annotated[pa.StructArray, Returns(arrow_type=_CLASSIFY_STRUCT)]:
        """Classify each text into a subset of the (constant) first row's categories."""
        cats = categories[0].as_py() if len(categories) else []
        return _classify(
            input,
            cats or [],
            model="",
            llm_secret=llm_secret,
            settings=engine.read_settings(
                llm_max_tokens=llm_max_tokens,
                llm_temperature=llm_temperature,
                llm_top_p=llm_top_p,
                llm_model=llm_model,
                llm_max_workers=llm_max_workers,
                llm_timeout=llm_timeout,
            ),
        )


class AiClassifyModel(ScalarFunction):
    """``ai_classify(input, categories, model)`` -- multi-label classification (explicit model)."""

    class Meta:
        """Declarative metadata for ``ai_classify(input, categories, model)``."""

        name = "ai_classify"
        description = "Classify text into a subset of the given categories with an explicit provider/model."
        categories = ["structured"]
        required_secrets = ["llm"]
        examples = _ex(
            "SELECT llm.main.ai_classify(t, ['billing','bug'], 'ollama/llama3.2').labels "
            "FROM (VALUES ('my card was declined')) AS x(t)",
            "Classify with an explicit provider/model",
        )
        tags = {**_CLASSIFY_TAGS, "vgi.example_queries": meta.example_queries_tag(examples)}

    @classmethod
    def compute(
        cls,
        input: Annotated[pa.StringArray, Param(doc="Text to classify.")],
        categories: Annotated[
            _ListArray, Param(arrow_type=pa.list_(pa.string()), doc="The set of category labels to choose from.")
        ],
        model: Annotated[str, ConstParam(_MODEL_DOC)],
        llm_secret: Annotated[dict[str, pa.Scalar[Any]] | None, Secret("llm")] = None,
        llm_max_tokens: Annotated[pa.Scalar[Any] | None, Setting()] = None,
        llm_temperature: Annotated[pa.Scalar[Any] | None, Setting()] = None,
        llm_top_p: Annotated[pa.Scalar[Any] | None, Setting()] = None,
        llm_model: Annotated[pa.Scalar[Any] | None, Setting()] = None,
        llm_max_workers: Annotated[pa.Scalar[Any] | None, Setting()] = None,
        llm_timeout: Annotated[pa.Scalar[Any] | None, Setting()] = None,
    ) -> Annotated[pa.StructArray, Returns(arrow_type=_CLASSIFY_STRUCT)]:
        """Classify each text using the explicit ``model``."""
        cats = categories[0].as_py() if len(categories) else []
        return _classify(
            input,
            cats or [],
            model=model,
            llm_secret=llm_secret,
            settings=engine.read_settings(
                llm_max_tokens=llm_max_tokens,
                llm_temperature=llm_temperature,
                llm_top_p=llm_top_p,
                llm_model=llm_model,
                llm_max_workers=llm_max_workers,
                llm_timeout=llm_timeout,
            ),
        )


# ===========================================================================
# ai_filter -- boolean predicate over text (for WHERE)
# ===========================================================================

_FILTER_TAGS = meta.object_tags(
    title="AI Boolean Filter",
    doc_llm=(
        "## ai_filter(predicate, input[, model])\n\n"
        "Evaluate a natural-language `predicate` against each `input` text and "
        "return a `BOOLEAN` -- true when the predicate holds, false when it does "
        "not. Designed for `WHERE` and `CASE`: keep only the rows an LLM judges "
        "to match a condition you phrase in plain English.\n\n"
        "**When to use.** Semantic filtering that a `LIKE`/regex can't express "
        '("is this review angry?", "does this mention a refund?"). For the '
        "matched labels use `ai_classify`; for extracted fields use `ai_extract`.\n\n"
        "**Input/output.** Inputs: `VARCHAR` predicate, `VARCHAR` text, optional "
        "model. Output: `BOOLEAN`. NULL/empty input, or an answer with no clear "
        "yes/no, -> NULL (excluded by a `WHERE`); a provider failure or missing "
        "key **raises** a DuckDB error."
    ),
    doc_md=(
        "# ai_filter\n\n"
        "LLM boolean predicate over text, for semantic `WHERE` filtering.\n\n"
        "## Notes\n\n"
        "- Returns TRUE/FALSE; NULL/empty input or an unparseable reply -> NULL.\n"
        "- A NULL is treated as not-matching by a `WHERE` clause."
    ),
    keywords=[
        "ai",
        "llm",
        "filter",
        "boolean",
        "predicate",
        "where",
        "semantic filter",
        "match",
        "condition",
        "yes no",
    ],
    category="structured",
)


def _filter(
    predicate: str,
    text: pa.StringArray,
    *,
    model: str,
    llm_secret: dict[str, Any] | None,
    settings: engine.RuntimeSettings | None = None,
) -> pa.BooleanArray:
    """Evaluate ``predicate`` over each text, returning a boolean column.

    Args:
        predicate: The natural-language condition to test.
        text: The per-row input texts.
        model: The model routing string ('' for default).
        llm_secret: Resolved unified ``llm`` secret fields, or None.
        settings: The resolved ``llm_*`` settings, or None for defaults.

    Returns:
        One BOOLEAN (or NULL) per row.
    """
    system = (
        "You are a strict boolean classifier. Decide whether the following condition holds for the "
        f"user's text. Condition: {predicate!r}. Reply with EXACTLY one word and nothing else: "
        "either true or false."
    )
    completions = _run(
        text.to_pylist(),
        build_messages=lambda p: engine.system_user_messages(system, p),
        model=model,
        llm_secret=llm_secret,
        settings=settings,
    )
    out = [engine.coerce_bool(c.text) if c is not None else None for c in completions]
    return pa.array(out, type=pa.bool_())


class AiFilter(ScalarFunction):
    """``ai_filter(predicate, input)`` -- boolean predicate over text (default model)."""

    class Meta:
        """Declarative metadata for ``ai_filter(predicate, input)``."""

        name = "ai_filter"
        description = "Evaluate a natural-language predicate over text; BOOLEAN, NULL on empty/unclear, errors raise."
        categories = ["structured"]
        required_secrets = ["llm"]
        examples = _ex(
            "SELECT llm.main.ai_filter('the text is a question', 'How do I reset my password?')",
            "Boolean predicate over text",
        )
        tags = {**_FILTER_TAGS, "vgi.example_queries": meta.example_queries_tag(examples)}

    @classmethod
    def compute(
        cls,
        predicate: Annotated[str, ConstParam("Natural-language condition to test for each row.")],
        input: Annotated[pa.StringArray, Param(doc="Text the predicate is evaluated against.")],
        llm_secret: Annotated[dict[str, pa.Scalar[Any]] | None, Secret("llm")] = None,
        llm_max_tokens: Annotated[pa.Scalar[Any] | None, Setting()] = None,
        llm_temperature: Annotated[pa.Scalar[Any] | None, Setting()] = None,
        llm_top_p: Annotated[pa.Scalar[Any] | None, Setting()] = None,
        llm_model: Annotated[pa.Scalar[Any] | None, Setting()] = None,
        llm_max_workers: Annotated[pa.Scalar[Any] | None, Setting()] = None,
        llm_timeout: Annotated[pa.Scalar[Any] | None, Setting()] = None,
    ) -> Annotated[pa.BooleanArray, Returns()]:
        """Evaluate the predicate over each text with the default model."""
        return _filter(
            predicate or "",
            input,
            model="",
            llm_secret=llm_secret,
            settings=engine.read_settings(
                llm_max_tokens=llm_max_tokens,
                llm_temperature=llm_temperature,
                llm_top_p=llm_top_p,
                llm_model=llm_model,
                llm_max_workers=llm_max_workers,
                llm_timeout=llm_timeout,
            ),
        )


class AiFilterModel(ScalarFunction):
    """``ai_filter(predicate, input, model)`` -- boolean predicate over text (explicit model)."""

    class Meta:
        """Declarative metadata for ``ai_filter(predicate, input, model)``."""

        name = "ai_filter"
        description = "Evaluate a natural-language predicate over text with an explicit model; returns BOOLEAN."
        categories = ["structured"]
        required_secrets = ["llm"]
        examples = _ex(
            "SELECT llm.main.ai_filter('mentions a refund', body, 'ollama/llama3.2') "
            "FROM (VALUES ('Please refund my order')) AS x(body)",
            "Boolean predicate with an explicit provider/model",
        )
        tags = {**_FILTER_TAGS, "vgi.example_queries": meta.example_queries_tag(examples)}

    @classmethod
    def compute(
        cls,
        predicate: Annotated[str, ConstParam("Natural-language condition to test for each row.")],
        input: Annotated[pa.StringArray, Param(doc="Text the predicate is evaluated against.")],
        model: Annotated[str, ConstParam(_MODEL_DOC)],
        llm_secret: Annotated[dict[str, pa.Scalar[Any]] | None, Secret("llm")] = None,
        llm_max_tokens: Annotated[pa.Scalar[Any] | None, Setting()] = None,
        llm_temperature: Annotated[pa.Scalar[Any] | None, Setting()] = None,
        llm_top_p: Annotated[pa.Scalar[Any] | None, Setting()] = None,
        llm_model: Annotated[pa.Scalar[Any] | None, Setting()] = None,
        llm_max_workers: Annotated[pa.Scalar[Any] | None, Setting()] = None,
        llm_timeout: Annotated[pa.Scalar[Any] | None, Setting()] = None,
    ) -> Annotated[pa.BooleanArray, Returns()]:
        """Evaluate the predicate over each text with the explicit ``model``."""
        return _filter(
            predicate or "",
            input,
            model=model,
            llm_secret=llm_secret,
            settings=engine.read_settings(
                llm_max_tokens=llm_max_tokens,
                llm_temperature=llm_temperature,
                llm_top_p=llm_top_p,
                llm_model=llm_model,
                llm_max_workers=llm_max_workers,
                llm_timeout=llm_timeout,
            ),
        )


# ===========================================================================
# ai_extract -- schema-driven extraction to JSON (VARCHAR)
# ===========================================================================

_EXTRACT_TAGS = meta.object_tags(
    title="AI Structured Extraction",
    doc_llm=(
        "## ai_extract(input, response_format[, model])\n\n"
        "Extract structured data from each `input` text according to a "
        "`response_format` -- a JSON Schema string describing the fields you "
        "want -- and return the model's JSON result as a `VARCHAR`. Parse it "
        "with DuckDB's JSON functions (`->`, `json_extract`, "
        "`from_json(..., schema)`).\n\n"
        "**When to use.** Pull typed fields (names, dates, amounts, nested "
        "objects) out of unstructured text. For a fixed label set use "
        "`ai_classify`; for a boolean use `ai_filter`.\n\n"
        "**Input/output.** Inputs: `VARCHAR` text, a constant JSON-Schema string, "
        "optional model. Output: a JSON `VARCHAR`. NULL/empty input -> NULL (no "
        "model call); a provider failure, or a reply that is not a JSON object, "
        "**raises** a DuckDB error. The schema is sent to the provider as a "
        "structured-output constraint when supported."
    ),
    doc_md=(
        "# ai_extract\n\n"
        "Schema-driven extraction from text to a JSON `VARCHAR`.\n\n"
        "## Usage\n\n"
        "```sql\n"
        "SELECT ai_extract(\n"
        "  'Invoice 42 for $9.99 due 2026-01-01',\n"
        '  \'{"type":"object","properties":{"amount":{"type":"number"}}}\'\n'
        ")::JSON->>'amount';\n"
        "```\n\n"
        "## Notes\n\n"
        "- `response_format` is a JSON-Schema string (a positional constant).\n"
        "- Output is a JSON string; NULL/empty input -> NULL, an unparseable reply raises."
    ),
    keywords=[
        "ai",
        "llm",
        "extract",
        "extraction",
        "structured",
        "json",
        "schema",
        "fields",
        "parse",
        "entities",
    ],
    category="structured",
)


def _extract(
    text: pa.StringArray,
    response_format: str,
    *,
    model: str,
    llm_secret: dict[str, Any] | None,
    settings: engine.RuntimeSettings | None = None,
) -> pa.StringArray:
    """Extract structured JSON from each text per a JSON-Schema string.

    Args:
        text: The per-row input texts.
        response_format: The JSON-Schema string describing the target shape.
        model: The model routing string ('' for default).
        llm_secret: Resolved unified ``llm`` secret fields, or None.
        settings: The resolved ``llm_*`` settings, or None for defaults.

    Returns:
        One JSON VARCHAR (or NULL) per row.

    Raises:
        ProviderError: If the model's reply does not parse to a JSON object.
    """
    schema_obj: dict[str, Any] | None = None
    try:
        parsed = json.loads(response_format) if response_format else None
        if isinstance(parsed, dict):
            schema_obj = parsed
    except (ValueError, TypeError):
        schema_obj = None

    schema_text = response_format or '{"type": "object"}'
    system = (
        "You are a precise information-extraction engine. Extract data from the user's text and reply "
        "ONLY with a single JSON object (no prose, no code fence) that conforms to this JSON Schema: "
        + schema_text
        + "."
    )
    rf = ResponseFormat(json_schema=schema_obj, name="extraction") if schema_obj is not None else None
    completions = _run(
        text.to_pylist(),
        build_messages=lambda p: engine.system_user_messages(system, p),
        model=model,
        llm_secret=llm_secret,
        settings=settings,
        response_format=rf,
    )
    out: list[str | None] = []
    for c in completions:
        obj = _parse_or_raise("ai_extract", c)
        out.append(json.dumps(obj) if obj is not None else None)
    return pa.array(out, type=pa.string())


class AiExtract(ScalarFunction):
    """``ai_extract(input, response_format)`` -- JSON extraction (default model)."""

    class Meta:
        """Declarative metadata for ``ai_extract(input, response_format)``."""

        name = "ai_extract"
        description = "Extract structured JSON from text per a JSON-Schema string; returns a JSON VARCHAR."
        categories = ["structured"]
        required_secrets = ["llm"]
        examples = _ex(
            'SELECT llm.main.ai_extract(\'Bob is 42\', \'{"type":"object","properties":{"age":{"type":"integer"}}}\')',
            "Extract fields as JSON per a schema",
        )
        tags = {**_EXTRACT_TAGS, "vgi.example_queries": meta.example_queries_tag(examples)}

    @classmethod
    def compute(
        cls,
        input: Annotated[pa.StringArray, Param(doc="Text to extract structured data from.")],
        response_format: Annotated[str, ConstParam("JSON-Schema string describing the fields to extract.")],
        llm_secret: Annotated[dict[str, pa.Scalar[Any]] | None, Secret("llm")] = None,
        llm_max_tokens: Annotated[pa.Scalar[Any] | None, Setting()] = None,
        llm_temperature: Annotated[pa.Scalar[Any] | None, Setting()] = None,
        llm_top_p: Annotated[pa.Scalar[Any] | None, Setting()] = None,
        llm_model: Annotated[pa.Scalar[Any] | None, Setting()] = None,
        llm_max_workers: Annotated[pa.Scalar[Any] | None, Setting()] = None,
        llm_timeout: Annotated[pa.Scalar[Any] | None, Setting()] = None,
    ) -> Annotated[pa.StringArray, Returns()]:
        """Extract structured JSON from each text with the default model."""
        return _extract(
            input,
            response_format or "",
            model="",
            llm_secret=llm_secret,
            settings=engine.read_settings(
                llm_max_tokens=llm_max_tokens,
                llm_temperature=llm_temperature,
                llm_top_p=llm_top_p,
                llm_model=llm_model,
                llm_max_workers=llm_max_workers,
                llm_timeout=llm_timeout,
            ),
        )


class AiExtractModel(ScalarFunction):
    """``ai_extract(input, response_format, model)`` -- JSON extraction (explicit model)."""

    class Meta:
        """Declarative metadata for ``ai_extract(input, response_format, model)``."""

        name = "ai_extract"
        description = "Extract structured JSON from text per a JSON-Schema string with an explicit model."
        categories = ["structured"]
        required_secrets = ["llm"]
        examples = _ex(
            "SELECT llm.main.ai_extract(t, '{\"type\":\"object\"}', 'ollama/llama3.2') "
            "FROM (VALUES ('Bob is 42')) AS x(t)",
            "Extract JSON with an explicit provider/model",
        )
        tags = {**_EXTRACT_TAGS, "vgi.example_queries": meta.example_queries_tag(examples)}

    @classmethod
    def compute(
        cls,
        input: Annotated[pa.StringArray, Param(doc="Text to extract structured data from.")],
        response_format: Annotated[str, ConstParam("JSON-Schema string describing the fields to extract.")],
        model: Annotated[str, ConstParam(_MODEL_DOC)],
        llm_secret: Annotated[dict[str, pa.Scalar[Any]] | None, Secret("llm")] = None,
        llm_max_tokens: Annotated[pa.Scalar[Any] | None, Setting()] = None,
        llm_temperature: Annotated[pa.Scalar[Any] | None, Setting()] = None,
        llm_top_p: Annotated[pa.Scalar[Any] | None, Setting()] = None,
        llm_model: Annotated[pa.Scalar[Any] | None, Setting()] = None,
        llm_max_workers: Annotated[pa.Scalar[Any] | None, Setting()] = None,
        llm_timeout: Annotated[pa.Scalar[Any] | None, Setting()] = None,
    ) -> Annotated[pa.StringArray, Returns()]:
        """Extract structured JSON from each text with the explicit ``model``."""
        return _extract(
            input,
            response_format or "",
            model=model,
            llm_secret=llm_secret,
            settings=engine.read_settings(
                llm_max_tokens=llm_max_tokens,
                llm_temperature=llm_temperature,
                llm_top_p=llm_top_p,
                llm_model=llm_model,
                llm_max_workers=llm_max_workers,
                llm_timeout=llm_timeout,
            ),
        )


# ===========================================================================
# ai_sentiment -- overall + per-category sentiment (STRUCT)
# ===========================================================================

_SENTIMENT_TAGS = meta.object_tags(
    title="AI Sentiment Analysis",
    doc_llm=(
        "## ai_sentiment(input[, model])\n\n"
        "Analyse the sentiment of each `input` text and return "
        "`STRUCT{overall VARCHAR, categories LIST<STRUCT{name VARCHAR, sentiment "
        "VARCHAR}>}`. `overall` is one of positive/negative/neutral/mixed; "
        "`categories` breaks sentiment down by aspect the model detects (e.g. "
        "price, service). The model is asked for strict JSON; an unparseable "
        "reply -> NULL row.\n\n"
        "**Input/output.** Input: one `VARCHAR` per row (+ optional model). "
        "Output: one sentiment `STRUCT` per row; NULL/empty input -> NULL (no "
        "model call). A provider failure, or a reply that is not the expected "
        "JSON, **raises** a DuckDB error."
    ),
    doc_md=(
        "# ai_sentiment\n\n"
        "Aspect-based sentiment as a `STRUCT` of an overall label plus per-category "
        "breakdown.\n\n"
        "## Notes\n\n"
        "- `overall` in {positive, negative, neutral, mixed}.\n"
        "- `categories` is a per-aspect list; NULL/empty input -> NULL, an unparseable reply raises."
    ),
    keywords=[
        "ai",
        "llm",
        "sentiment",
        "opinion",
        "emotion",
        "polarity",
        "positive",
        "negative",
        "aspect",
        "reviews",
    ],
    category="structured",
)

_SENTIMENT_SCHEMA = {
    "type": "object",
    "properties": {
        "overall": {"type": "string", "enum": ["positive", "negative", "neutral", "mixed"]},
        "categories": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {"name": {"type": "string"}, "sentiment": {"type": "string"}},
                "required": ["name", "sentiment"],
                "additionalProperties": False,
            },
        },
    },
    "required": ["overall", "categories"],
    "additionalProperties": False,
}

_SENTIMENT_SYSTEM = (
    "You are a sentiment-analysis engine. Analyse the user's text and reply ONLY with a JSON object of "
    'the form {"overall": "positive|negative|neutral|mixed", "categories": [{"name": "aspect", '
    '"sentiment": "positive|negative|neutral"}]}. Include a category per notable aspect; use an empty '
    "list when none stand out."
)


def _sentiment(
    text: pa.StringArray,
    *,
    model: str,
    llm_secret: dict[str, Any] | None,
    settings: engine.RuntimeSettings | None = None,
) -> pa.StructArray:
    """Analyse sentiment for each text, returning the sentiment STRUCT.

    Args:
        text: The per-row input texts.
        model: The model routing string ('' for default).
        llm_secret: Resolved unified ``llm`` secret fields, or None.
        settings: The resolved ``llm_*`` settings, or None for defaults.

    Returns:
        One sentiment STRUCT (or NULL) per row.

    Raises:
        ProviderError: If the model's reply is not a JSON object with ``overall``.
    """
    completions = _run(
        text.to_pylist(),
        build_messages=lambda p: engine.system_user_messages(_SENTIMENT_SYSTEM, p),
        model=model,
        llm_secret=llm_secret,
        settings=settings,
        response_format=ResponseFormat(json_schema=_SENTIMENT_SCHEMA, name="sentiment"),
    )
    rows: list[dict[str, Any] | None] = []
    for c in completions:
        obj = _parse_or_raise("ai_sentiment", c)
        if obj is None:
            rows.append(None)
            continue
        if "overall" not in obj:
            raise ProviderError(f"ai_sentiment: JSON is missing the 'overall' field: {obj!r}")
        cats_raw = obj.get("categories")
        cats: list[dict[str, str]] = []
        if isinstance(cats_raw, list):
            for item in cats_raw:
                if isinstance(item, dict) and "name" in item and "sentiment" in item:
                    cats.append({"name": str(item["name"]), "sentiment": str(item["sentiment"])})
        rows.append({"overall": str(obj["overall"]), "categories": cats})
    return pa.array(rows, type=_SENTIMENT_STRUCT)  # type: ignore[return-value]


class AiSentiment(ScalarFunction):
    """``ai_sentiment(input)`` -- overall + per-aspect sentiment (default model)."""

    class Meta:
        """Declarative metadata for ``ai_sentiment(input)``."""

        name = "ai_sentiment"
        description = "Analyse sentiment; returns STRUCT{overall, categories LIST<STRUCT{name, sentiment}>}."
        categories = ["structured"]
        required_secrets = ["llm"]
        examples = _ex(
            "SELECT llm.main.ai_sentiment('The food was great but service was slow').overall",
            "Overall + per-aspect sentiment",
        )
        tags = {**_SENTIMENT_TAGS, "vgi.example_queries": meta.example_queries_tag(examples)}

    @classmethod
    def compute(
        cls,
        input: Annotated[pa.StringArray, Param(doc="Text to analyse for sentiment.")],
        llm_secret: Annotated[dict[str, pa.Scalar[Any]] | None, Secret("llm")] = None,
        llm_max_tokens: Annotated[pa.Scalar[Any] | None, Setting()] = None,
        llm_temperature: Annotated[pa.Scalar[Any] | None, Setting()] = None,
        llm_top_p: Annotated[pa.Scalar[Any] | None, Setting()] = None,
        llm_model: Annotated[pa.Scalar[Any] | None, Setting()] = None,
        llm_max_workers: Annotated[pa.Scalar[Any] | None, Setting()] = None,
        llm_timeout: Annotated[pa.Scalar[Any] | None, Setting()] = None,
    ) -> Annotated[pa.StructArray, Returns(arrow_type=_SENTIMENT_STRUCT)]:
        """Analyse each text's sentiment with the default model."""
        return _sentiment(
            input,
            model="",
            llm_secret=llm_secret,
            settings=engine.read_settings(
                llm_max_tokens=llm_max_tokens,
                llm_temperature=llm_temperature,
                llm_top_p=llm_top_p,
                llm_model=llm_model,
                llm_max_workers=llm_max_workers,
                llm_timeout=llm_timeout,
            ),
        )


class AiSentimentModel(ScalarFunction):
    """``ai_sentiment(input, model)`` -- overall + per-aspect sentiment (explicit model)."""

    class Meta:
        """Declarative metadata for ``ai_sentiment(input, model)``."""

        name = "ai_sentiment"
        description = "Analyse sentiment with an explicit model; returns the sentiment STRUCT."
        categories = ["structured"]
        required_secrets = ["llm"]
        examples = _ex(
            "SELECT llm.main.ai_sentiment(review, 'ollama/llama3.2').overall "
            "FROM (VALUES ('The food was great but service was slow')) AS x(review)",
            "Sentiment with an explicit provider/model",
        )
        tags = {**_SENTIMENT_TAGS, "vgi.example_queries": meta.example_queries_tag(examples)}

    @classmethod
    def compute(
        cls,
        input: Annotated[pa.StringArray, Param(doc="Text to analyse for sentiment.")],
        model: Annotated[str, ConstParam(_MODEL_DOC)],
        llm_secret: Annotated[dict[str, pa.Scalar[Any]] | None, Secret("llm")] = None,
        llm_max_tokens: Annotated[pa.Scalar[Any] | None, Setting()] = None,
        llm_temperature: Annotated[pa.Scalar[Any] | None, Setting()] = None,
        llm_top_p: Annotated[pa.Scalar[Any] | None, Setting()] = None,
        llm_model: Annotated[pa.Scalar[Any] | None, Setting()] = None,
        llm_max_workers: Annotated[pa.Scalar[Any] | None, Setting()] = None,
        llm_timeout: Annotated[pa.Scalar[Any] | None, Setting()] = None,
    ) -> Annotated[pa.StructArray, Returns(arrow_type=_SENTIMENT_STRUCT)]:
        """Analyse each text's sentiment with the explicit ``model``."""
        return _sentiment(
            input,
            model=model,
            llm_secret=llm_secret,
            settings=engine.read_settings(
                llm_max_tokens=llm_max_tokens,
                llm_temperature=llm_temperature,
                llm_top_p=llm_top_p,
                llm_model=llm_model,
                llm_max_workers=llm_max_workers,
                llm_timeout=llm_timeout,
            ),
        )


# ===========================================================================
# ai_summarize -- concise summary (VARCHAR)
# ===========================================================================

_SUMMARIZE_TAGS = meta.object_tags(
    title="AI Summarize",
    doc_llm=(
        "## ai_summarize(input[, model])\n\n"
        "Summarise each `input` text into a short, faithful summary returned as "
        "`VARCHAR`. One row in, one summary out. For summarising *across* rows of "
        "a group (map-reduce over a whole column) use the `ai_summarize_agg` "
        "aggregate instead.\n\n"
        "**Input/output.** Input: one `VARCHAR` per row (+ optional model). "
        "Output: one `VARCHAR` summary per row. NULL/empty input -> NULL (no "
        "model call); a provider failure **raises** a DuckDB error."
    ),
    doc_md=(
        "# ai_summarize\n\n"
        "Per-row LLM summarization returning `VARCHAR`.\n\n"
        "## Notes\n\n"
        "- Summarises one row at a time; use `ai_summarize_agg` across a group.\n"
        "- NULL/empty input -> NULL; a provider failure raises a DuckDB error."
    ),
    keywords=[
        "ai",
        "llm",
        "summarize",
        "summary",
        "summarise",
        "tldr",
        "condense",
        "abstract",
        "shorten",
    ],
    category="completion",
)

_SUMMARIZE_SYSTEM = (
    "You are a concise summarizer. Summarize the user's text faithfully in a few sentences. Reply with "
    "the summary only -- no preamble, no bullet labels."
)


def _summarize(
    text: pa.StringArray,
    *,
    model: str,
    llm_secret: dict[str, Any] | None,
    settings: engine.RuntimeSettings | None = None,
) -> pa.StringArray:
    """Summarize each text to a short summary string.

    Args:
        text: The per-row input texts.
        model: The model routing string ('' for default).
        llm_secret: Resolved unified ``llm`` secret fields, or None.
        settings: The resolved ``llm_*`` settings, or None for defaults.

    Returns:
        One VARCHAR summary (or NULL) per row.
    """
    completions = _run(
        text.to_pylist(),
        build_messages=lambda p: engine.system_user_messages(_SUMMARIZE_SYSTEM, p),
        model=model,
        llm_secret=llm_secret,
        settings=settings,
    )
    return pa.array([c.text if c is not None else None for c in completions], type=pa.string())


class AiSummarize(ScalarFunction):
    """``ai_summarize(input)`` -- concise summary (default model)."""

    class Meta:
        """Declarative metadata for ``ai_summarize(input)``."""

        name = "ai_summarize"
        description = "Summarize each text into a short summary (default model); NULL on empty input, errors raise."
        categories = ["completion"]
        required_secrets = ["llm"]
        examples = _ex(
            "SELECT llm.main.ai_summarize('DuckDB is an in-process SQL OLAP database ...')", "Summarize a text"
        )
        tags = {**_SUMMARIZE_TAGS, "vgi.example_queries": meta.example_queries_tag(examples)}

    @classmethod
    def compute(
        cls,
        input: Annotated[pa.StringArray, Param(doc="Text to summarize.")],
        llm_secret: Annotated[dict[str, pa.Scalar[Any]] | None, Secret("llm")] = None,
        llm_max_tokens: Annotated[pa.Scalar[Any] | None, Setting()] = None,
        llm_temperature: Annotated[pa.Scalar[Any] | None, Setting()] = None,
        llm_top_p: Annotated[pa.Scalar[Any] | None, Setting()] = None,
        llm_model: Annotated[pa.Scalar[Any] | None, Setting()] = None,
        llm_max_workers: Annotated[pa.Scalar[Any] | None, Setting()] = None,
        llm_timeout: Annotated[pa.Scalar[Any] | None, Setting()] = None,
    ) -> Annotated[pa.StringArray, Returns()]:
        """Summarize each text with the default model."""
        return _summarize(
            input,
            model="",
            llm_secret=llm_secret,
            settings=engine.read_settings(
                llm_max_tokens=llm_max_tokens,
                llm_temperature=llm_temperature,
                llm_top_p=llm_top_p,
                llm_model=llm_model,
                llm_max_workers=llm_max_workers,
                llm_timeout=llm_timeout,
            ),
        )


class AiSummarizeModel(ScalarFunction):
    """``ai_summarize(input, model)`` -- concise summary (explicit model)."""

    class Meta:
        """Declarative metadata for ``ai_summarize(input, model)``."""

        name = "ai_summarize"
        description = "Summarize each text into a short summary with an explicit provider/model."
        categories = ["completion"]
        required_secrets = ["llm"]
        examples = _ex(
            "SELECT llm.main.ai_summarize(body, 'ollama/llama3.2') "
            "FROM (VALUES ('DuckDB is an in-process OLAP database.')) AS x(body)",
            "Summarize with an explicit provider/model",
        )
        tags = {**_SUMMARIZE_TAGS, "vgi.example_queries": meta.example_queries_tag(examples)}

    @classmethod
    def compute(
        cls,
        input: Annotated[pa.StringArray, Param(doc="Text to summarize.")],
        model: Annotated[str, ConstParam(_MODEL_DOC)],
        llm_secret: Annotated[dict[str, pa.Scalar[Any]] | None, Secret("llm")] = None,
        llm_max_tokens: Annotated[pa.Scalar[Any] | None, Setting()] = None,
        llm_temperature: Annotated[pa.Scalar[Any] | None, Setting()] = None,
        llm_top_p: Annotated[pa.Scalar[Any] | None, Setting()] = None,
        llm_model: Annotated[pa.Scalar[Any] | None, Setting()] = None,
        llm_max_workers: Annotated[pa.Scalar[Any] | None, Setting()] = None,
        llm_timeout: Annotated[pa.Scalar[Any] | None, Setting()] = None,
    ) -> Annotated[pa.StringArray, Returns()]:
        """Summarize each text with the explicit ``model``."""
        return _summarize(
            input,
            model=model,
            llm_secret=llm_secret,
            settings=engine.read_settings(
                llm_max_tokens=llm_max_tokens,
                llm_temperature=llm_temperature,
                llm_top_p=llm_top_p,
                llm_model=llm_model,
                llm_max_workers=llm_max_workers,
                llm_timeout=llm_timeout,
            ),
        )


# ===========================================================================
# ai_count_tokens -- local heuristic token estimate (BIGINT, no model call)
# ===========================================================================

_COUNT_TOKENS_TAGS = meta.object_tags(
    title="AI Token Estimate",
    doc_llm=(
        "## ai_count_tokens(input[, model])\n\n"
        "Count the number of tokens in each `input` text with a **local "
        "tiktoken** tokenizer and return a `BIGINT` -- **no** network/provider "
        "call, so it is fast, deterministic, and free. The optional `model` "
        "argument selects the tokenizer: OpenAI/GPT-family models are counted "
        "**exactly**; other models (Anthropic, Ollama, ...) use `o200k_base`, a "
        "strong cross-model estimate. Use it to budget prompts, chunk long text, "
        "or estimate cost before calling a model.\n\n"
        "**Input/output.** Input: one `VARCHAR` per row. Output: `BIGINT` token count "
        "(>= 1 for non-empty text). NULL/empty input -> NULL. For exact Anthropic "
        "counts use the provider's own token-count API."
    ),
    doc_md=(
        "# ai_count_tokens\n\n"
        "Local, network-free token counting (tiktoken) returning `BIGINT`.\n\n"
        "## Notes\n\n"
        "- Exact for OpenAI models; `o200k_base` estimate for other models.\n"
        "- No provider call, deterministic; NULL/empty input returns NULL."
    ),
    keywords=[
        "ai",
        "llm",
        "tokens",
        "token count",
        "count tokens",
        "estimate",
        "budget",
        "cost",
        "chunk",
        "context window",
    ],
    category="utility",
)


@functools.cache
def _encoding(name: str) -> Any:
    """Load (and cache) a ``tiktoken`` encoding by name.

    Args:
        name: A tiktoken encoding name (e.g. ``o200k_base``).

    Returns:
        The tiktoken ``Encoding``.
    """
    import tiktoken

    return tiktoken.get_encoding(name)


def _encoding_for_model(model: str) -> Any:
    """Resolve a ``tiktoken`` encoding for a (possibly provider-prefixed) model.

    The model's provider prefix is stripped (``openai/gpt-4o`` -> ``gpt-4o``) and
    mapped to its OpenAI encoding; anything unknown (Anthropic/Ollama/others)
    falls back to ``o200k_base`` -- a strong cross-model default.

    Args:
        model: The model string, or '' for the default encoding.

    Returns:
        The tiktoken ``Encoding`` to count with.
    """
    bare = (model or "").split("/")[-1].strip()
    if bare:
        try:
            import tiktoken

            return tiktoken.encoding_for_model(bare)
        except Exception:  # noqa: BLE001 - unknown model -> strong default encoding
            pass
    return _encoding("o200k_base")


def _count_tokens(text: pa.StringArray, *, model: str = "") -> pa.Int64Array:
    """Count tokens per row with ``tiktoken`` (local, no network).

    Exact for OpenAI models; a close cross-model estimate elsewhere. NULL/empty
    input maps to NULL.

    Args:
        text: The per-row input texts.
        model: The model whose tokenizer to use ('' for the default encoding).

    Returns:
        One BIGINT token count (or NULL) per row.
    """
    enc = _encoding_for_model(model)
    out: list[int | None] = []
    for t in text.to_pylist():
        if t is None or not t.strip():
            out.append(None)
        else:
            out.append(len(enc.encode(t, disallowed_special=())))
    return pa.array(out, type=pa.int64())


class AiCountTokens(ScalarFunction):
    """``ai_count_tokens(input)`` -- local token estimate (no model call)."""

    class Meta:
        """Declarative metadata for ``ai_count_tokens(input)``."""

        name = "ai_count_tokens"
        description = "Count tokens per text with a local tiktoken tokenizer (no network); BIGINT."
        categories = ["utility"]
        examples = _ex("SELECT llm.main.ai_count_tokens('the quick brown fox')", "Count the tokens in a text")
        tags = {**_COUNT_TOKENS_TAGS, "vgi.example_queries": meta.example_queries_tag(examples)}

    @classmethod
    def compute(
        cls,
        input: Annotated[pa.StringArray, Param(doc="Text to count tokens for.")],
    ) -> Annotated[pa.Int64Array, Returns()]:
        """Count tokens per row with the default tokenizer."""
        return _count_tokens(input)


class AiCountTokensModel(ScalarFunction):
    """``ai_count_tokens(input, model)`` -- local token estimate (model advisory)."""

    class Meta:
        """Declarative metadata for ``ai_count_tokens(input, model)``."""

        name = "ai_count_tokens"
        description = "Count tokens per text with the tokenizer for an explicit model (local, no network); BIGINT."
        categories = ["utility"]
        examples = _ex(
            "SELECT llm.main.ai_count_tokens('the quick brown fox', 'openai/gpt-4o')",
            "Count tokens with a specific model's tokenizer",
        )
        tags = {**_COUNT_TOKENS_TAGS, "vgi.example_queries": meta.example_queries_tag(examples)}

    @classmethod
    def compute(
        cls,
        input: Annotated[pa.StringArray, Param(doc="Text to count tokens for.")],
        model: Annotated[str, ConstParam("Model whose tokenizer to use (OpenAI models are exact).")],
    ) -> Annotated[pa.Int64Array, Returns()]:
        """Count tokens per row with the tokenizer for ``model``."""
        return _count_tokens(input, model=model or "")


# ===========================================================================
# prompt -- pure, SAFE template substitution (no model, no str.format)
# ===========================================================================


def _safe_format(template: str, args: list[Any]) -> str | None:
    """Substitute positional ``args`` into ``template`` without ``str.format``.

    Supports ``{}`` (sequential) and ``{n}`` (explicit index) placeholders plus
    ``{{`` / ``}}`` escapes. Deliberately **rejects** format specs and
    attribute/index access (``{0.attr}``, ``{0[k]}``, ``{:...}``, ``{0:...}``) --
    so there is no attribute traversal and no ``{:>9999999999}`` allocation
    attack. A malformed template or an out-of-range index yields ``None`` for
    that row (this is a pure function -- it never calls a provider, so it returns
    NULL rather than raising).

    Args:
        template: The template string.
        args: The positional substitution values (any type; None -> "").

    Returns:
        The substituted string, or None for a malformed/out-of-range template.
    """
    parts: list[str] = []
    auto = 0
    i = 0
    n = len(template)
    while i < n:
        ch = template[i]
        if ch == "{":
            if template[i + 1 : i + 2] == "{":
                parts.append("{")
                i += 2
                continue
            end = template.find("}", i + 1)
            if end == -1:
                return None  # unterminated placeholder
            spec = template[i + 1 : end]
            if spec == "":
                key = auto
                auto += 1
            elif spec.isdigit():
                key = int(spec)
            else:
                return None  # format spec / attribute / index access -> rejected
            if key < 0 or key >= len(args):
                return None  # out-of-range index
            value = args[key]
            parts.append("" if value is None else str(value))
            i = end + 1
        elif ch == "}":
            if template[i + 1 : i + 2] == "}":
                parts.append("}")
                i += 2
                continue
            return None  # lone closing brace
        else:
            parts.append(ch)
            i += 1
    return "".join(parts)


_PROMPT_TAGS = meta.object_tags(
    title="Prompt Template",
    doc_llm=(
        "## prompt(template, args...)\n\n"
        "Build a prompt string from a `template` and one or more positional "
        "`args` by **safe** positional substitution: `{}` (sequential) and `{n}` "
        "(explicit index), with `{{`/`}}` escapes. This is **pure** -- it calls "
        "**no** model -- so it is the cheap building block for assembling prompts "
        "from columns before feeding them to `ai_complete` / `ai_classify` / "
        "etc. Unlike Python `str.format`, it rejects format specs and "
        "attribute/index access, so a template can never traverse object "
        "attributes or trigger a giant-width allocation.\n\n"
        "**Input/output.** Inputs: a `VARCHAR` `template` and >= 1 further column "
        "values (any type). Output: one `VARCHAR` per row. A NULL template, a "
        "malformed placeholder, or an out-of-range index -> NULL row."
    ),
    doc_md=(
        "# prompt\n\n"
        "Pure, safe prompt-template substitution (no model call).\n\n"
        "## Notes\n\n"
        "- `{}` (sequential) and `{n}` (index) placeholders; `{{`/`}}` escape braces.\n"
        "- Rejects format specs / attribute access (no `str.format`); malformed -> NULL."
    ),
    keywords=[
        "prompt",
        "template",
        "format",
        "substitute",
        "interpolate",
        "build prompt",
        "string format",
        "compose",
    ],
    category="utility",
)


class Prompt(ScalarFunction):
    """``prompt(template, args...)`` -- pure positional template substitution."""

    class Meta:
        """Declarative metadata for ``prompt(template, args...)``."""

        name = "prompt"
        description = "Safely substitute positional args into a template ({} / {n}); pure, no model call."
        categories = ["utility"]
        examples = _ex(
            "SELECT llm.main.prompt('Translate {} into {}', 'hello', 'French')",
            "Fill a template with positional arguments",
        )
        tags = {**_PROMPT_TAGS, "vgi.example_queries": meta.example_queries_tag(examples)}

    @classmethod
    def compute(
        cls,
        template: Annotated[pa.StringArray, Param(doc="Template with positional {} placeholders.")],
        args: Annotated[_Array, Param(doc="Values substituted into the template (one or more columns).", varargs=True)],
    ) -> Annotated[pa.StringArray, Returns()]:
        """Substitute the per-row ``args`` into each ``template`` (safe, no str.format)."""
        template_vals = template.to_pylist()
        arg_cols = [a.to_pylist() for a in args]
        out: list[str | None] = []
        for i, tmpl in enumerate(template_vals):
            if tmpl is None:
                out.append(None)
                continue
            row_args = [col[i] for col in arg_cols]
            out.append(_safe_format(tmpl, row_args))
        return pa.array(out, type=pa.string())


# ===========================================================================
# ai_embed -- local ONNX embedding (FLOAT[]), keyless
# ===========================================================================

_EMBED_TAGS = meta.object_tags(
    title="AI Text Embedding",
    doc_llm=(
        "## ai_embed(input[, model])\n\n"
        "Embed each `input` text into a fixed-length `FLOAT[]` vector using a "
        "local ONNX model (fastembed) -- **no API key, no network** after the "
        "one-time model download. The default model "
        f"(`{models.DEFAULT_MODEL}`) produces {models.embedding_dim(None)}-dim "
        "vectors; pass a second argument to pick another supported model.\n\n"
        "**When to use.** Semantic search, RAG, clustering, and dedup inside SQL: "
        "embed a corpus once, embed a query, and rank rows by `ai_similarity`. "
        "Pairs naturally with the DuckDB VSS extension.\n\n"
        "**Input/output.** Input: one `VARCHAR` per row (+ optional model). "
        "Output: one `FLOAT[]` per row; NULL/empty text -> NULL vector."
    ),
    doc_md=(
        "# ai_embed\n\n"
        "Keyless local text embedding to `FLOAT[]` (fastembed/ONNX).\n\n"
        "## Notes\n\n"
        "- Runs locally; no LLM key required.\n"
        "- NULL/empty input returns a NULL vector; score pairs with `ai_similarity`."
    ),
    keywords=[
        "ai",
        "embed",
        "embedding",
        "vector",
        "fastembed",
        "onnx",
        "semantic search",
        "retrieval",
        "rag",
        "keyless",
    ],
    category="embedding",
)


def _embed_array(text: pa.StringArray, *, model: str | None) -> _ListArray:
    """Embed a string array to a ``list<float32>`` array, NULL/empty -> NULL.

    Args:
        text: The per-row input texts.
        model: The embedding model name, or None/empty for the default.

    Returns:
        One ``FLOAT[]`` (or NULL) per row.
    """
    values = text.to_pylist()
    live_idx: list[int] = []
    live_text: list[str] = []
    for i, t in enumerate(values):
        if t is not None and t.strip():
            live_idx.append(i)
            live_text.append(t)

    vectors: list[list[float] | None] = [None] * len(values)
    if live_text:
        embedded = models.embed_texts(live_text, model=model)
        for i, vec in zip(live_idx, embedded, strict=True):
            vectors[i] = vec
    return pa.array(vectors, type=_VECTOR)  # type: ignore[return-value]


class AiEmbed(ScalarFunction):
    """``ai_embed(input)`` -- keyless local embedding with the default model."""

    class Meta:
        """Declarative metadata for ``ai_embed(input)``."""

        name = "ai_embed"
        description = (
            f"Embed text into a {models.embedding_dim(None)}-dim FLOAT[] with the local default model "
            f"({models.DEFAULT_MODEL}); keyless. NULL/empty -> NULL."
        )
        categories = ["embedding"]
        examples = _ex("SELECT len(llm.main.ai_embed('hello world'))", "Embed text into a FLOAT[] vector (keyless)")
        tags = {**_EMBED_TAGS, "vgi.example_queries": meta.example_queries_tag(examples)}

    @classmethod
    def compute(
        cls,
        input: Annotated[pa.StringArray, Param(doc="Text to embed.")],
    ) -> Annotated[_ListArray, Returns(arrow_type=_VECTOR)]:
        """Embed each text with the default local model."""
        return _embed_array(input, model=None)


class AiEmbedModel(ScalarFunction):
    """``ai_embed(input, model)`` -- keyless local embedding with an explicit model."""

    class Meta:
        """Declarative metadata for ``ai_embed(input, model)``."""

        name = "ai_embed"
        description = "Embed text into a FLOAT[] with an explicit local model (see the README for names); keyless."
        categories = ["embedding"]
        examples = _ex(
            "SELECT len(llm.main.ai_embed('hello world', 'BAAI/bge-base-en-v1.5'))",
            "Embed with an explicitly chosen local model",
        )
        tags = {**_EMBED_TAGS, "vgi.example_queries": meta.example_queries_tag(examples)}

    @classmethod
    def compute(
        cls,
        input: Annotated[pa.StringArray, Param(doc="Text to embed.")],
        model: Annotated[str, ConstParam("Local embedding model name (e.g. 'BAAI/bge-base-en-v1.5').")],
    ) -> Annotated[_ListArray, Returns(arrow_type=_VECTOR)]:
        """Embed each text with the explicit local ``model``."""
        return _embed_array(input, model=model or None)


# ===========================================================================
# ai_similarity -- cosine similarity of two FLOAT[] vectors (DOUBLE), keyless
# ===========================================================================

_SIMILARITY_TAGS = meta.object_tags(
    title="AI Cosine Similarity",
    doc_llm=(
        "## ai_similarity(a, b)\n\n"
        "Cosine similarity of two `FLOAT[]` vectors (typically from `ai_embed`), "
        "a `DOUBLE` in `[-1, 1]` where 1 means identical direction. Pure arithmetic "
        "-- **no model, no key, no I/O** -- so it is cheap over large joins.\n\n"
        "**When to use.** Rank or threshold embedding pairs: "
        "`ORDER BY ai_similarity(ai_embed(q), ai_embed(doc)) DESC` for retrieval, "
        "or `WHERE ai_similarity(...) > 0.8` for near-duplicate detection.\n\n"
        "**Input/output.** Inputs: two `FLOAT[]` vectors. Output: `DOUBLE`. NULL, "
        "empty, or length-mismatched pairs -> NULL (never an error)."
    ),
    doc_md=(
        "# ai_similarity\n\n"
        "Cosine similarity of two `FLOAT[]` vectors, in `[-1, 1]` (pure "
        "arithmetic, keyless).\n\n"
        "## Notes\n\n"
        "- 1.0 = identical direction, 0 = orthogonal, -1 = opposite.\n"
        "- NULL / empty / length-mismatch returns NULL."
    ),
    keywords=[
        "ai",
        "similarity",
        "cosine",
        "cosine similarity",
        "distance",
        "rank",
        "compare vectors",
        "nearest neighbor",
        "semantic search",
        "keyless",
    ],
    category="embedding",
)


class AiSimilarity(ScalarFunction):
    """``ai_similarity(a, b)`` -- cosine similarity of two FLOAT[] vectors."""

    class Meta:
        """Declarative metadata for ``ai_similarity(a, b)``."""

        name = "ai_similarity"
        description = "Cosine similarity of two FLOAT[] vectors, in [-1, 1] (pure arithmetic); NULL on mismatch."
        categories = ["embedding"]
        examples = _ex(
            "SELECT llm.main.ai_similarity(llm.main.ai_embed('cat'), llm.main.ai_embed('kitten'))",
            "Cosine similarity between two embeddings",
        )
        tags = {**_SIMILARITY_TAGS, "vgi.example_queries": meta.example_queries_tag(examples)}

    @classmethod
    def compute(
        cls,
        a: Annotated[_ListArray, Param(arrow_type=_VECTOR, doc="First embedding vector.")],
        b: Annotated[_ListArray, Param(arrow_type=_VECTOR, doc="Second embedding vector.")],
    ) -> Annotated[pa.DoubleArray, Returns(arrow_type=pa.float64())]:
        """Cosine similarity of each ``(a, b)`` vector pair."""
        out = [models.cosine_similarity(x, y) for x, y in zip(a.to_pylist(), b.to_pylist(), strict=False)]
        return pa.array(out, type=pa.float64())


def _similarity_text(a: pa.StringArray, b: pa.StringArray, *, model: str | None) -> pa.DoubleArray:
    """Embed two text columns locally and return their per-row cosine similarity.

    Args:
        a: The first text column.
        b: The second text column.
        model: The local embedding model name, or None/empty for the default.

    Returns:
        One DOUBLE cosine similarity (or NULL) per row.
    """
    va = _embed_array(a, model=model).to_pylist()
    vb = _embed_array(b, model=model).to_pylist()
    out = [models.cosine_similarity(x, y) for x, y in zip(va, vb, strict=False)]
    return pa.array(out, type=pa.float64())


class AiSimilarityText(ScalarFunction):
    """``ai_similarity(a, b)`` -- cosine similarity of two texts (embed + compare)."""

    class Meta:
        """Declarative metadata for the text form ``ai_similarity(a, b)``."""

        name = "ai_similarity"
        description = "Cosine similarity of two texts: embed both locally (keyless) and compare; DOUBLE in [-1, 1]."
        categories = ["embedding"]
        examples = _ex(
            "SELECT ROUND(llm.main.ai_similarity('cat', 'kitten'), 3) AS score",
            "Cosine similarity of two texts (embedded locally)",
        )
        tags = {**_SIMILARITY_TAGS, "vgi.example_queries": meta.example_queries_tag(examples)}

    @classmethod
    def compute(
        cls,
        a: Annotated[pa.StringArray, Param(doc="First text to embed and compare.")],
        b: Annotated[pa.StringArray, Param(doc="Second text to embed and compare.")],
    ) -> Annotated[pa.DoubleArray, Returns(arrow_type=pa.float64())]:
        """Embed each ``(a, b)`` text pair locally and return their cosine similarity."""
        return _similarity_text(a, b, model=None)


class AiSimilarityTextModel(ScalarFunction):
    """``ai_similarity(a, b, model)`` -- cosine similarity of two texts, explicit model."""

    class Meta:
        """Declarative metadata for the text form ``ai_similarity(a, b, model)``."""

        name = "ai_similarity"
        description = "Cosine similarity of two texts with an explicit local embedding model; DOUBLE in [-1, 1]."
        categories = ["embedding"]
        examples = _ex(
            "SELECT ROUND(llm.main.ai_similarity('cat', 'kitten', 'BAAI/bge-small-en-v1.5'), 3) AS score",
            "Cosine similarity of two texts with a chosen model",
        )
        tags = {**_SIMILARITY_TAGS, "vgi.example_queries": meta.example_queries_tag(examples)}

    @classmethod
    def compute(
        cls,
        a: Annotated[pa.StringArray, Param(doc="First text to embed and compare.")],
        b: Annotated[pa.StringArray, Param(doc="Second text to embed and compare.")],
        model: Annotated[str, ConstParam("Local embedding model name (e.g. 'BAAI/bge-small-en-v1.5').")],
    ) -> Annotated[pa.DoubleArray, Returns(arrow_type=pa.float64())]:
        """Embed each ``(a, b)`` text pair with ``model`` and return their cosine similarity."""
        return _similarity_text(a, b, model=model or None)


SCALAR_FUNCTIONS: list[type] = [
    AiComplete,
    AiCompleteModel,
    AiCompleteDetails,
    AiCompleteDetailsModel,
    AiCompleteImage,
    AiCompleteImageModel,
    AiClassify,
    AiClassifyModel,
    AiFilter,
    AiFilterModel,
    AiExtract,
    AiExtractModel,
    AiSentiment,
    AiSentimentModel,
    AiSummarize,
    AiSummarizeModel,
    AiCountTokens,
    AiCountTokensModel,
    Prompt,
    AiEmbed,
    AiEmbedModel,
    AiSimilarity,
    AiSimilarityText,
    AiSimilarityTextModel,
]


# Every overload of a name must advertise all overloads example SQL (see meta docs).
meta.apply_combined_example_queries(SCALAR_FUNCTIONS)
