# Copyright 2026 Query Farm LLC - https://query.farm

"""Shared execution core for the AISQL scalar and aggregate functions.

Everything that touches an LLM provider funnels through here so the behaviour is
identical everywhere and easy to test offline:

- :func:`resolve_provider` is the single routing seam. It wraps
  ``providers.resolve``; tests monkeypatch *this* symbol to inject a
  ``FakeProvider`` without going near the real SDKs.
- :func:`map_complete` fans a column of prompts across a **bounded**
  ``ThreadPoolExecutor``. It calls the provider **once per distinct prompt**
  (within-batch dedup), masks NULL/empty *input* to ``None``, and -- critically
  -- lets a provider/runtime failure **propagate as a DuckDB error** rather than
  swallowing it to NULL. Only blank input rows become NULL.
- :func:`parse_json_object` / :func:`coerce_bool` are the tolerant parsers the
  structured functions (``ai_classify`` / ``ai_sentiment`` / ``ai_extract`` /
  ``ai_filter``) share.
- :class:`RuntimeSettings` carries the resolved ``aisql_*`` DuckDB settings
  (sampling / routing / concurrency knobs) into the provider call.
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any

from vgi_aisql.providers import (
    Completion,
    CompletionParams,
    Message,
    ProviderError,
    ResponseFormat,
    resolve,
)

#: Default cap on concurrent per-distinct-prompt provider calls within one batch.
DEFAULT_MAX_WORKERS = 8


def resolve_provider(
    model: str | None,
    *,
    secrets: dict[str, Any] | None = None,
    client: Any | None = None,
    timeout: float | None = None,
) -> tuple[Any, str]:
    """Route a model string to a ``(provider, model_id)`` pair.

    This is the test seam: it simply delegates to ``providers.resolve`` in
    production, but offline tests replace this module-level symbol so a
    ``FakeProvider`` is returned instead of a real SDK-backed one.

    Args:
        model: The user-supplied model string, or None/empty for the default.
        secrets: Resolved VGI secrets mapping.
        client: Injectable SDK client (tests).
        timeout: Per-request timeout (seconds) forwarded to the provider.

    Returns:
        The provider instance and the model id to send to it.
    """
    return resolve(model, secrets=secrets, client=client, timeout=timeout)


def _is_blank(text: str | None) -> bool:
    """Whether ``text`` is NULL or empty/whitespace-only.

    Args:
        text: The value to test.

    Returns:
        True when the value should map to a NULL output row.
    """
    return text is None or not text.strip()


def map_complete(
    prompts: Sequence[str | None],
    *,
    build_messages: Callable[[str], list[Message]],
    model: str | None,
    secrets: dict[str, Any] | None = None,
    params: CompletionParams | None = None,
    response_format: ResponseFormat | None = None,
    max_workers: int = DEFAULT_MAX_WORKERS,
    timeout: float | None = None,
    client: Any | None = None,
) -> list[Completion | None]:
    """Complete a column of prompts, returning one :class:`Completion` per row.

    Blank/NULL *input* prompts map to ``None`` without a provider call. The
    remaining rows are de-duplicated (one provider call per **distinct** prompt),
    dispatched across a bounded thread pool, then fanned back to every row in
    input order. A provider-resolution failure, or any per-call failure, is
    **raised** (surfacing as a DuckDB error) -- it is never swallowed to NULL.

    Args:
        prompts: The per-row prompt texts (None/empty rows yield None).
        build_messages: Turns one prompt string into the provider message list.
        model: The model string to route on.
        secrets: Resolved VGI secrets mapping.
        params: Sampling parameters (defaults applied when None).
        response_format: Optional structured-output JSON Schema request.
        max_workers: Upper bound on concurrent provider calls.
        timeout: Per-request timeout (seconds) forwarded to the provider.
        client: Injectable SDK client (tests).

    Returns:
        One ``Completion`` (or ``None`` for a blank input row) per input row.

    Raises:
        ProviderError: On provider resolution failure or a per-call failure
            (wrapped with context) -- surfaced as a DuckDB error, not NULL.
    """
    out: list[Completion | None] = [None] * len(prompts)
    live = [(i, p) for i, p in enumerate(prompts) if not _is_blank(p)]
    if not live:
        return out

    # One call per distinct prompt (within-batch dedup): a repeated column value
    # is completed once and fanned to every row that carries it.
    distinct: list[str] = []
    seen: set[str] = set()
    for _i, p in live:
        assert p is not None
        if p not in seen:
            seen.add(p)
            distinct.append(p)

    provider, model_id = resolve_provider(model, secrets=secrets, client=client, timeout=timeout)
    call_params = params or CompletionParams()

    def _one(prompt: str) -> Completion:
        # No error swallowing: a provider failure propagates out of pool.map,
        # out of compute(), and surfaces as a DuckDB error.
        completion: Completion = provider.complete(
            build_messages(prompt),
            model=model_id,
            params=call_params,
            response_format=response_format,
        )
        return completion

    workers = max(1, min(max_workers, len(distinct)))
    try:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            by_prompt = dict(zip(distinct, pool.map(_one, distinct), strict=True))
    except ProviderError:
        raise
    except Exception as exc:  # noqa: BLE001 - add context, then re-raise (never swallow)
        raise ProviderError(f"aisql: LLM call failed: {exc}") from exc

    for i, p in live:
        assert p is not None
        out[i] = by_prompt[p]
    return out


def user_message(prompt: str) -> list[Message]:
    """Build a single-turn user message list from a plain prompt.

    Args:
        prompt: The user prompt text.

    Returns:
        A one-element message list.
    """
    return [Message(role="user", content=prompt)]


def system_user_messages(system: str, prompt: str) -> list[Message]:
    """Build a system + user message list.

    Args:
        system: The system instruction.
        prompt: The user prompt text.

    Returns:
        A two-element message list.
    """
    return [Message(role="system", content=system), Message(role="user", content=prompt)]


_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.DOTALL)


def _strip_fences(text: str) -> str:
    """Strip a Markdown code fence wrapper if the model added one.

    Args:
        text: Raw model output.

    Returns:
        The inner content when fenced, otherwise the original text.
    """
    match = _FENCE_RE.search(text)
    return match.group(1) if match else text


def parse_json_object(text: str | None) -> dict[str, Any] | None:
    """Parse a JSON object out of model output, tolerating fences/prose.

    Tries a direct parse first, then a fenced-block parse, then the substring
    from the first ``{`` to the last ``}``. Returns ``None`` (never raises) when
    nothing parses to a JSON object -- callers decide whether that is an error.

    Args:
        text: The raw model output, or None.

    Returns:
        The parsed object, or None on any failure.
    """
    if text is None:
        return None
    candidates = [text, _strip_fences(text)]
    start, end = text.find("{"), text.rfind("}")
    if 0 <= start < end:
        candidates.append(text[start : end + 1])
    for candidate in candidates:
        try:
            parsed = json.loads(candidate)
        except (ValueError, TypeError):
            continue
        if isinstance(parsed, dict):
            return parsed
    return None


_TRUE_WORDS = {"true", "yes"}
_FALSE_WORDS = {"false", "no"}


def coerce_bool(text: str | None) -> bool | None:
    """Coerce a yes/no model answer to a Python bool; None when unparseable.

    Scans for the first whole **word** in {``yes``, ``no``, ``true``, ``false``}
    (case-insensitive). Digit/letter tokens like ``0``/``1``/``t``/``f`` are
    deliberately ignored -- a preamble such as ``"Item 0: yes"`` must read as
    True, not be flipped by the leading ``0``. Returns ``None`` (a NULL row) when
    no yes/no word is present.

    Args:
        text: The raw model output, or None.

    Returns:
        The coerced boolean, or None when unparseable.
    """
    if text is None:
        return None
    for word in re.findall(r"[a-z]+", text.lower()):
        if word in _TRUE_WORDS:
            return True
        if word in _FALSE_WORDS:
            return False
    return None


def build_secrets(
    aisql_secret: dict[str, Any] | None,
    extra: dict[str, dict[str, Any] | None] | None = None,
) -> dict[str, Any]:
    """Assemble the ``secrets`` mapping ``providers.resolve`` expects.

    Args:
        aisql_secret: The resolved unified ``aisql`` secret fields, or None.
        extra: Optional additional provider-named secret entries.

    Returns:
        A ``{secret_type: fields}`` mapping (empty when nothing is configured).
    """
    secrets: dict[str, Any] = {}
    if aisql_secret:
        secrets["aisql"] = aisql_secret
    if extra:
        for name, entry in extra.items():
            if entry:
                secrets[name] = entry
    return secrets


# ---------------------------------------------------------------------------
# Runtime settings (the aisql_* DuckDB session settings)
# ---------------------------------------------------------------------------


def _as_py(scalar: Any) -> Any:
    """Unwrap a ``pa.Scalar`` (or plain value/None) to a Python value.

    Args:
        scalar: A ``pa.Scalar``, a Python value, or None.

    Returns:
        The Python value, or None.
    """
    if scalar is None:
        return None
    return scalar.as_py() if hasattr(scalar, "as_py") else scalar


@dataclass(frozen=True)
class RuntimeSettings:
    """Resolved ``aisql_*`` knobs, threaded from DuckDB settings into a call.

    Every field is optional: an unset setting keeps the provider/library
    default. ``model`` is the global default model used only when a call's own
    ``model`` argument is empty.
    """

    max_tokens: int | None = None
    temperature: float | None = None
    top_p: float | None = None
    model: str | None = None
    max_workers: int | None = None
    timeout: float | None = None

    def completion_params(self) -> CompletionParams:
        """Build :class:`CompletionParams`, applying only the settings that are set.

        Returns:
            Completion parameters (unset sampling knobs are left at the default,
            i.e. not sent to the provider).
        """
        kwargs: dict[str, Any] = {}
        if self.max_tokens is not None:
            kwargs["max_tokens"] = int(self.max_tokens)
        if self.temperature is not None:
            kwargs["temperature"] = float(self.temperature)
        if self.top_p is not None:
            kwargs["top_p"] = float(self.top_p)
        return CompletionParams(**kwargs)

    def effective_model(self, call_model: str) -> str:
        """The model to route on: the per-call arg, else the ``aisql_model`` default.

        Args:
            call_model: The model argument passed to the function ('' if omitted).

        Returns:
            The resolved model string (may be '' for the provider default).
        """
        return call_model or (self.model or "")

    def workers(self) -> int:
        """The concurrency cap (``aisql_max_workers`` or the default).

        Returns:
            The maximum number of concurrent provider calls.
        """
        return int(self.max_workers) if self.max_workers else DEFAULT_MAX_WORKERS

    def timeout_value(self) -> float | None:
        """The per-request timeout in seconds (``aisql_timeout`` or None).

        Returns:
            The timeout, or None to use the provider default.
        """
        return float(self.timeout) if self.timeout is not None else None


def read_settings(
    *,
    aisql_max_tokens: Any = None,
    aisql_temperature: Any = None,
    aisql_top_p: Any = None,
    aisql_model: Any = None,
    aisql_max_workers: Any = None,
    aisql_timeout: Any = None,
) -> RuntimeSettings:
    """Build :class:`RuntimeSettings` from the raw ``Setting()`` scalars.

    Each argument is the ``pa.Scalar`` (or None) delivered by a ``Setting()``
    compute parameter; unset settings arrive as None and keep their default.

    The sampling knobs use a negative **sentinel** for "unset" (their registered
    default) so they are always delivered but only sent to the provider when set
    to a real value; an empty ``aisql_model`` likewise means "no override".

    Args:
        aisql_max_tokens: Output-token cap setting scalar, or None.
        aisql_temperature: Sampling temperature setting scalar, or None.
        aisql_top_p: Nucleus-sampling setting scalar, or None.
        aisql_model: Global default model setting scalar, or None.
        aisql_max_workers: Concurrency-cap setting scalar, or None.
        aisql_timeout: Request-timeout setting scalar, or None.

    Returns:
        The resolved settings.
    """
    return RuntimeSettings(
        max_tokens=_as_py(aisql_max_tokens),
        temperature=_positive(_as_py(aisql_temperature)),
        top_p=_positive(_as_py(aisql_top_p)),
        model=_nonempty(_as_py(aisql_model)),
        max_workers=_as_py(aisql_max_workers),
        timeout=_as_py(aisql_timeout),
    )


def _positive(value: Any) -> float | None:
    """Treat a negative sentinel as "unset" for a sampling knob.

    Args:
        value: A numeric setting value, or None.

    Returns:
        The float value when >= 0, else None (unset -> not sent to the provider).
    """
    if value is None or value < 0:
        return None
    return float(value)


def _nonempty(value: Any) -> str | None:
    """Treat an empty string as "unset" for the default-model setting.

    Args:
        value: A string setting value, or None.

    Returns:
        The string when non-empty, else None.
    """
    return value if value else None
