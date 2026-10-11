"""Opt-in ``web.search_fallbacks`` failover: provider error of quota/auth/transient class -> next listed
backend serves, result marked; "no results" (a success) never fails over; unset => unchanged."""

import json

import pytest

import tools.web_tools as web_tools


class _P:
    def __init__(self, name, resp, ready=True):
        self.name, self._resp, self._ready, self.calls = name, resp, ready, 0

    def supports_search(self):
        return True

    def is_available(self):
        return self._ready

    def is_keyless_available(self):
        return False

    def search(self, query, limit=5):
        self.calls += 1
        if isinstance(self._resp, Exception):
            raise self._resp
        return self._resp


OK = {"success": True, "data": {"web": [{"url": "https://x.example", "title": "t", "description": "d", "position": 1}]}}
EMPTY = {"success": True, "data": {"web": []}}
PAYMENT = {"success": False, "error": "Payment Required: Insufficient credits"}


@pytest.fixture
def setup(monkeypatch):
    def _go(primary, others, cfg_fallbacks):
        provs = {p.name: p for p in [primary, *others]}
        monkeypatch.setattr(web_tools, "_ensure_web_plugins_loaded", lambda: None)
        monkeypatch.setattr(web_tools, "_load_web_config",
                            lambda: {"backend": primary.name, "search_fallbacks": cfg_fallbacks, "keyless_rescue": False})
        monkeypatch.setattr("agent.web_search_registry.get_provider", lambda n: provs.get(n))
        monkeypatch.setattr(web_tools, "_registered_web_provider", lambda n: provs.get(n))
        return json.loads(web_tools.web_search_tool("q", limit=2))
    return _go


def test_payment_error_fails_over_to_next_available(setup):
    first, skipped, second = _P("firecrawl", PAYMENT), _P("tavily", OK, ready=False), _P("ddgs", OK)
    out = setup(first, [skipped, second], ["tavily", "ddgs"])
    assert out["success"] is True
    assert out["data"]["served_by"] == "ddgs" and out["data"]["failed_over_from"] == "firecrawl"
    assert skipped.calls == 0  # is_available() False => skipped


def test_raised_exception_also_fails_over(setup):
    out = setup(_P("firecrawl", RuntimeError("Connection timed out")), [_P("ddgs", OK)], ["ddgs"])
    assert out["data"]["served_by"] == "ddgs"


def test_no_results_is_success_and_does_not_fail_over(setup):
    second = _P("ddgs", OK)
    out = setup(_P("firecrawl", EMPTY), [second], ["ddgs"])
    assert out["success"] is True and out["data"]["web"] == []
    assert second.calls == 0


def test_unset_fallbacks_keeps_upstream_behaviour(setup):
    second = _P("ddgs", OK)
    out = setup(_P("firecrawl", PAYMENT), [second], [])
    assert out["success"] is False and second.calls == 0
