"""One-shot keyless-ring rescue for failed keyed/configured web calls.

Stateless by design: a rescue routes THIS call through the free-tier ring (plugins/web/keyless_mcp.py);
the next web_search/web_extract call attempts the chosen backend again. Callers must never cache a
rescue-served response, or the one-shot rescue becomes sticky for a whole TTL. Logs under the origin
(tools.web_tools) logger.
"""

import logging

logger = logging.getLogger("tools.web_tools")

# Ring vendor -> env var holding its paid key (keyed mode ⇒ eligible for rescue).
_RING_KEY_VARS = {
    "exa": "EXA_API_KEY", "parallel": "PARALLEL_API_KEY",
    "firecrawl": "FIRECRAWL_API_KEY", "keenable": "KEENABLE_API_KEY",
}


def _keyless_rescue_enabled() -> bool:
    """``web.keyless_rescue`` (default on), implicitly off when the keyless tier is disabled."""
    from tools.web_tools import _load_web_config
    if not _load_web_config().get("keyless_rescue", True):
        return False
    try:
        from agent.web_search_registry import _keyless_tier_enabled
        return _keyless_tier_enabled()
    except Exception as exc:  # noqa: BLE001 — registry optional
        logger.debug("keyless rescue tier check failed: %s", exc)
        return False


def _ring_vendor_keyless(name: str) -> bool:
    """Did *name*'s own provider route this call through the anonymous keyless ring?

    Mirrors the predicate each ring provider evaluates before calling, so eligibility reflects what
    actually happened. Firecrawl owns extra routes that bypass the ring without a key — the managed
    Nous Tool Gateway (persisted ``nous`` selection, or the legacy never-configured fallback when the
    gateway is ready) and a self-hosted ``FIRECRAWL_API_URL`` — so it is asked directly.
    """
    if name == "firecrawl":
        from plugins.web.firecrawl.provider import _use_keyless_ring
        return _use_keyless_ring()
    from agent.web_search_provider import get_provider_env
    from plugins.web.keyless_mcp import use_keyless
    key_var = _RING_KEY_VARS.get(name, "")
    return use_keyless(name, get_provider_env(key_var) if key_var else "")


def _rescue_eligible(provider) -> bool:
    """True when a failed call on *provider* should get a one-shot rescue.

    Eligible: any call that did NOT go through the keyless ring — a non-ring backend, a ring vendor
    in keyed mode, or a ring vendor routed through the managed gateway / a self-hosted instance. A
    ring vendor that walked the ring is NOT eligible: its failure means the ring already failed.
    """
    if not _keyless_rescue_enabled() or provider is None:
        return False
    try:
        from plugins.web.keyless_mcp import _KEYLESS_RING
        name = getattr(provider, "name", "")
        return name not in _KEYLESS_RING or not _ring_vendor_keyless(name)
    except Exception as exc:  # noqa: BLE001 — rescue is best-effort
        logger.debug("rescue eligibility check failed: %s", exc)
        return False


def _rescue_search(provider_name: str, original_error: str, query: str, limit: int) -> dict:
    """Rescue a failed search via the ring; annotate the result with the original failure."""
    from plugins.web.keyless_mcp import search_with_failover
    logger.warning(
        "web_search backend '%s' failed (%s); one-shot keyless rescue",
        provider_name, (original_error or "")[:200],
    )
    rescued = search_with_failover(provider_name, query, limit)
    if rescued.get("success"):
        rescued.setdefault("data", {}).update(
            rescued_from=provider_name,
            backend_error=(
                f"Configured backend '{provider_name}' failed this call "
                f"({(original_error or 'unknown error')[:300]}); result served by the keyless free tier. "
                f"The next call will use '{provider_name}' again."
            ),
        )
        return rescued
    # Ring also failed: the ORIGINAL error names the user's setup, so lead with it.
    return {
        "success": False,
        "error": (
            f"{original_error or 'search failed'} "
            f"(keyless rescue also failed: {rescued.get('error', 'unknown')})"
        ),
    }


def _policy_blocked_result(result: dict) -> bool:
    """True for a website-policy refusal — intentional, never rescued (it would fetch blocked content)."""
    error = str(result.get("error") or "").lower()
    return bool(result.get("blocked_by_policy")) or "blocked by website policy" in error


def _rescue_extract(provider_name: str, urls: list, results: list) -> list:
    """Rescue a whole-batch extract failure via the ring.

    Only genuine failures are re-fetched; policy-blocked entries are preserved verbatim. If the provider
    broke url/result order parity, every entry is treated as rescueable and the ring's list replaces the
    batch wholesale.
    """
    from plugins.web.keyless_mcp import extract_with_failover

    parity = len(results) == len(urls)
    rescue_idx = [i for i, r in enumerate(results) if not parity or not _policy_blocked_result(r)]
    if not rescue_idx:
        return results  # every failure is an intentional policy block

    rescue_urls = [urls[i] for i in rescue_idx] if parity else list(urls)
    errors = (results[i].get("error") for i in rescue_idx if results[i].get("error"))
    original_error = next(errors, "extract failed")
    logger.warning(
        "web_extract backend '%s' failed all %d URL(s) (%s); one-shot keyless rescue",
        provider_name, len(rescue_urls), (original_error or "")[:200],
    )
    rescued = extract_with_failover(provider_name, list(rescue_urls))
    if rescued and all(r.get("error", "") for r in rescued):
        return results  # rescue also failed everywhere: keep original errors
    for r in rescued:
        meta = None if r.get("error") else r.setdefault("metadata", {})
        if isinstance(meta, dict):
            meta["rescued_from"] = provider_name
            meta["backend_error"] = (original_error or "")[:300]
    if parity and len(rescued) == len(rescue_idx):
        replacements = dict(zip(rescue_idx, rescued))
        return [replacements.get(i, r) for i, r in enumerate(results)]
    return rescued


# --- Opt-in configured failover: ``web.search_fallbacks: [name, ...]`` -------------------------------
# Unset/empty => no behaviour change. Tried BEFORE the keyless-ring rescue. "No results" is a success
# (success=True, empty list) and never reaches here; only provider errors are classified.
_FAILOVER_MARKERS = (
    # payment / quota
    "payment required", "insufficient credit", "out of credits", "quota", "402", "429", "rate limit",
    "too many requests", "usage limit", "billing",
    # auth
    "unauthorized", "forbidden", "401", "403", "invalid api key", "api key", "not set", "authentication",
    # transient
    "timed out", "timeout", "connection", "temporarily", "unavailable", "502", "503", "504", "500",
    "bad gateway", "reset by peer",
)


def _failover_class(error: str) -> str:
    """'quota' | 'auth' | 'transient' | '' (unclassified: a malformed query fails everywhere, don't fail over)."""
    e = (error or "").lower()
    if any(m in e for m in ("payment required", "insufficient credit", "out of credits", "quota", "402", "429",
                            "rate limit", "too many requests", "usage limit", "billing")):
        return "quota"
    if any(m in e for m in ("unauthorized", "forbidden", "401", "403", "invalid api key", "api key",
                            "not set", "authentication")):
        return "auth"
    if any(m in e for m in _FAILOVER_MARKERS):
        return "transient"
    return ""


def _search_fallback_names() -> list:
    from tools.web_tools import _load_web_config
    raw = _load_web_config().get("search_fallbacks") or []
    if isinstance(raw, str):
        raw = [raw]
    return [str(n).lower().strip() for n in raw if str(n).strip()] if isinstance(raw, (list, tuple)) else []


def _fallback_search(provider_name: str, original_error: str, query: str, limit: int):
    """Try ``web.search_fallbacks`` in order after *provider_name* failed. Returns the annotated success
    response, or ``None`` (nothing configured / error not failover-class / nobody served)."""
    names = [n for n in _search_fallback_names() if n != provider_name]
    if not names or not _failover_class(original_error):
        return None
    from tools.web_tools import _provider_is_ready, _registered_web_provider
    for name in names:
        provider = _registered_web_provider(name)
        if provider is None or not provider.supports_search() or not _provider_is_ready(provider):
            continue
        try:
            resp = provider.search(query, limit)
        except Exception as exc:  # noqa: BLE001 — next fallback
            logger.info("search fallback '%s' raised: %s", name, str(exc)[:200])
            continue
        if resp.get("success"):
            logger.warning(
                "web_search backend '%s' failed (%s: %s); served by fallback '%s'",
                provider_name, _failover_class(original_error), (original_error or "")[:120], name,
            )
            resp.setdefault("data", {}).update(
                served_by=name, failed_over_from=provider_name,
                backend_error=f"Backend '{provider_name}' failed ({(original_error or '')[:200]}); served by '{name}'.",
            )
            return resp
        logger.info("search fallback '%s' failed: %s", name, str(resp.get("error", ""))[:200])
    return None
