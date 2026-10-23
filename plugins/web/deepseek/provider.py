"""DeepSeek Search + Fetch provider (``/v1/tools/search``, ``/v1/tools/fetch``; sync httpx).

Env: ``DEEPSEEK_API_KEY`` (the same key the chat API uses), ``DEEPSEEK_BASE_URL``.

Billing is **per successful request**, not per token: $0.75 per 1,000 calls on each
endpoint, charged from day one for direct API use. A request only bills when it
returns HTTP 200 with a result — failures, empty results and timeouts are free.

Rate limit: 5 requests/second per endpoint, per account, shared across API keys.

Opt-in only: available when the key is set AND ``deepseek`` is named in ``web.backend``,
``web.search_backend``, ``web.extract_backend`` or ``web.search_fallbacks``. The key is the
chat key, so key-only availability made the registry pick this paid backend on its own for any
install holding just that key, where it used to reach the free keyless tier.
The gate is capability-blind: naming deepseek for search (or only as a fallback) on an install
with no other keyed backend also makes it the extract (or primary) backend.
"""

from __future__ import annotations

import logging
import time
from typing import Any

import httpx

from plugins.web._common import (
    BaseWebSearchProvider,
    _interrupted,
    document,
    http_status_detail,
    page_error,
    provider_env,
    run_extract,
    run_search,
    search_ok,
    setup_schema,
    title_hit,
)

logger = logging.getLogger(__name__)

#: Server-side ceiling for ``count`` (1..50; this vendor allows more than the 20 others do).
COUNT_CAP = 50

#: Minimum spacing between calls — the account is limited to 5 req/s per endpoint.
_MIN_INTERVAL = 0.21

_last_call: dict[str, float] = {}

_RATE_LIMIT_RETRIES = 2


def _headers(api_key: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}


def _throttle(endpoint: str) -> None:
    """Keep successive calls to one endpoint at least 210 ms apart (5 req/s cap)."""
    now = time.monotonic()
    previous = _last_call.get(endpoint)
    if previous is not None:
        wait = _MIN_INTERVAL - (now - previous)
        if wait > 0:
            time.sleep(wait)
    _last_call[endpoint] = time.monotonic()


def _post(endpoint: str, payload: dict[str, Any]) -> dict[str, Any]:
    """POST to a DeepSeek tool endpoint; retry twice on 429 with linear backoff.

    Non-2xx raises ``ValueError`` carrying the response body, so the caller's
    failure classification (and Hermes's search failover) sees the vendor text.
    """
    api_key = provider_env("DEEPSEEK_API_KEY")
    if not api_key:
        raise ValueError(
            "DEEPSEEK_API_KEY is not set. Add it to ~/.hermes/.env or select a different "
            "web backend with `hermes tools`."
        )
    base_url = (provider_env("DEEPSEEK_BASE_URL") or "https://api.deepseek.com").rstrip("/")
    # DEEPSEEK_BASE_URL is the chat provider's variable, whose usual value ends in /v1.
    base_url = base_url.removesuffix("/v1")
    url = f"{base_url}/v1/tools/{endpoint}"

    attempt = 0
    while True:
        _throttle(endpoint)
        response = httpx.post(url, json=payload, timeout=60, headers=_headers(api_key))
        if response.status_code == 429 and attempt < _RATE_LIMIT_RETRIES:
            attempt += 1
            time.sleep(0.5 * attempt)
            continue
        if response.status_code >= 400:
            raise ValueError(http_status_detail(response))
        return response.json()


def _row(raw: dict[str, Any], position: int) -> dict[str, Any]:
    """Map one API result to the historical row shape.

    ``summary`` (600–800 chars, written to be read by a model) is preferred over
    ``snippet`` (100–200 chars, written for a UI). Missing pieces degrade to ``""``
    rather than raising — a malformed row must not sink the whole result set.
    """
    description = str(raw.get("summary") or raw.get("snippet") or "")
    return title_hit(str(raw.get("title") or ""), str(raw.get("url") or ""), description, position)


def _score(raw: Any) -> float:
    try:
        return float(raw.get("rerank_score") or 0.0)
    except (AttributeError, TypeError, ValueError):
        return 0.0


def _sort_by_relevance(results: list[Any]) -> list[dict[str, Any]]:
    """Best-first by ``rerank_score``. Non-dict rows are dropped and a bad score sorts last,
    so one malformed row cannot sink the result set."""
    rows = [r for r in results if isinstance(r, dict)]
    return sorted(rows, key=_score, reverse=True)


def _opted_in() -> bool:
    """True when config names ``deepseek`` as a web backend or a search fallback."""
    from tools.web_tools import _load_web_config

    web = _load_web_config()
    names = [web.get(k) for k in ("backend", "search_backend", "extract_backend")]
    fallbacks = web.get("search_fallbacks") or []
    names += [fallbacks] if isinstance(fallbacks, str) else list(fallbacks)
    return any(str(n).strip().lower() == "deepseek" for n in names if n)


class DeepSeekWebSearchProvider(BaseWebSearchProvider):
    """DeepSeek search + fetch provider (keyed; no keyless tier)."""

    NAME = "deepseek"
    DISPLAY_NAME = "DeepSeek"
    KEY_ENV = "DEEPSEEK_API_KEY"
    EXTRACT = True
    KEYLESS = False

    def is_available(self) -> bool:
        return super().is_available() and _opted_in()

    def search(self, query: str, limit: int = 5) -> dict[str, Any]:
        def _body() -> dict[str, Any]:
            count = max(1, min(int(limit or 10), COUNT_CAP))
            logger.info("DeepSeek search: '%s' (count=%d)", query, count)
            raw = _post(
                "search",
                {"query": query, "count": count, "need_content": False},
            )
            hits = _sort_by_relevance(list(raw.get("results") or []))
            return search_ok([_row(r, i + 1) for i, r in enumerate(hits)])

        return run_search("DeepSeek", logger, _body)

    def extract(self, urls: list[str], **kwargs: Any) -> list[dict[str, Any]]:
        """Fetch each URL; ``/v1/tools/fetch`` takes exactly one URL per request."""

        def _body() -> list[dict[str, Any]]:
            documents: list[dict[str, Any]] = []
            logger.info("DeepSeek fetch: %d URL(s)", len(urls))
            for i, url in enumerate(urls):
                if _interrupted():
                    documents.extend(page_error(u, "interrupted") for u in urls[i:])
                    break
                try:
                    raw = _post("fetch", {"url": url})
                    documents.append(
                        document(url, str(raw.get("title") or ""), str(raw.get("content") or ""))
                    )
                except Exception as exc:  # noqa: BLE001 — one bad URL must not sink the batch
                    documents.append(page_error(url, str(exc)[:300]))
            return documents

        return run_extract("DeepSeek", logger, urls, _body)

    def get_setup_schema(self) -> dict[str, Any]:
        return setup_schema(
            "DeepSeek",
            "paid · per request",
            "Search + fetch. $0.75 per 1,000 successful requests per endpoint; billed from the "
            "first call. Uses DEEPSEEK_API_KEY.",
            "DEEPSEEK_API_KEY",
            "DeepSeek API key (shared with the chat API)",
            "https://platform.deepseek.com/api_keys",
        )
