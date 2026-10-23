"""Unit tests for the DeepSeek web search + fetch provider (no network)."""

from __future__ import annotations

import pytest

from plugins.web.deepseek import provider as ds


@pytest.fixture(autouse=True)
def _clear_throttle():
    ds._last_call.clear()
    yield
    ds._last_call.clear()


def test_search_maps_rows_and_prefers_summary(monkeypatch):
    monkeypatch.setattr(ds, "_post", lambda endpoint, payload: {
        "results": [
            {"title": "A", "url": "https://a.example", "snippet": "short", "summary": "long " * 20,
             "rerank_score": 1.2},
            {"title": "B", "url": "https://b.example", "snippet": "b snippet", "rerank_score": 2.4},
        ]
    })
    out = ds.DeepSeekWebSearchProvider().search("q", 5)
    assert out["success"] is True
    rows = out["data"]["web"]
    # sorted best-first by rerank_score
    assert [r["title"] for r in rows] == ["B", "A"]
    assert rows[0]["position"] == 1 and rows[1]["position"] == 2
    # summary preferred, snippet used when summary is absent
    assert rows[0]["description"] == "b snippet"
    assert rows[1]["description"].startswith("long long")


def test_search_falls_back_to_snippet_and_tolerates_missing_fields(monkeypatch):
    monkeypatch.setattr(ds, "_post", lambda endpoint, payload: {"results": [{}]})
    rows = ds.DeepSeekWebSearchProvider().search("q", 3)["data"]["web"]
    assert rows == [{"title": "", "url": "", "description": "", "position": 1}]


def test_search_clamps_count(monkeypatch):
    seen = {}

    def _fake(endpoint, payload):
        seen.update(payload)
        return {"results": []}

    monkeypatch.setattr(ds, "_post", _fake)
    ds.DeepSeekWebSearchProvider().search("q", 200)
    assert seen["count"] == ds.COUNT_CAP == 50
    assert seen["need_content"] is False


def test_search_surfaces_api_error(monkeypatch):
    def _boom(endpoint, payload):
        raise ValueError('{"error":{"message":"rate limited"}}')

    monkeypatch.setattr(ds, "_post", _boom)
    out = ds.DeepSeekWebSearchProvider().search("q", 5)
    assert out["success"] is False
    assert "rate limited" in out["error"]


def test_extract_builds_documents(monkeypatch):
    monkeypatch.setattr(ds, "_post", lambda endpoint, payload: {
        "title": "Example", "content": "# Example\n\nBody.",
    })
    docs = ds.DeepSeekWebSearchProvider().extract(["https://example.com"])
    assert len(docs) == 1
    assert docs[0]["url"] == "https://example.com"
    assert docs[0]["title"] == "Example"
    assert docs[0]["content"].startswith("# Example")
    assert docs[0]["raw_content"] == docs[0]["content"]
    assert "error" not in docs[0]


def test_extract_one_bad_url_does_not_sink_the_batch(monkeypatch):
    def _fake(endpoint, payload):
        if "bad" in payload["url"]:
            raise ValueError("failed to fetch URL")
        return {"title": "ok", "content": "fine"}

    monkeypatch.setattr(ds, "_post", _fake)
    docs = ds.DeepSeekWebSearchProvider().extract(["https://good.example", "https://bad.example"])
    assert docs[0]["content"] == "fine"
    assert "error" not in docs[0]
    assert "failed to fetch URL" in docs[1]["error"]
    assert docs[1]["url"] == "https://bad.example"


def test_missing_key_is_a_clear_error(monkeypatch):
    monkeypatch.setattr(ds, "provider_env", lambda name: "")
    out = ds.DeepSeekWebSearchProvider().search("q", 5)
    assert out["success"] is False
    assert "DEEPSEEK_API_KEY is not set" in out["error"]


def test_availability_follows_the_key(monkeypatch):
    # is_available() is inherited from BaseWebSearchProvider, which resolves
    # provider_env from plugins.web._common — patch there, not on the provider module.
    import plugins.web._common as common

    monkeypatch.setattr(ds, "_opted_in", lambda: True)
    monkeypatch.setattr(common, "provider_env", lambda name: "sk-test")
    assert ds.DeepSeekWebSearchProvider().is_available() is True
    monkeypatch.setattr(common, "provider_env", lambda name: "")
    assert ds.DeepSeekWebSearchProvider().is_available() is False


@pytest.mark.parametrize("web, expected", [
    ({}, False),
    ({"backend": "exa", "search_fallbacks": ["keenable"]}, False),
    ({"search_fallbacks": ["deepseek", "keenable"]}, True),
    ({"search_fallbacks": "DeepSeek"}, True),
    ({"backend": "deepseek"}, True),
    ({"search_backend": "deepseek"}, True),
    ({"extract_backend": "deepseek"}, True),
])
def test_the_chat_key_alone_does_not_make_it_available(monkeypatch, web, expected):
    """The key is the chat key. Key-only availability let the registry pick this paid
    backend for any install holding just that key, instead of the free keyless tier."""
    import plugins.web._common as common
    import tools.web_tools as web_tools

    monkeypatch.setattr(common, "provider_env", lambda name: "sk-test")
    monkeypatch.setattr(web_tools, "_load_web_config", lambda: web)
    assert ds.DeepSeekWebSearchProvider().is_available() is expected


class _Resp:
    def __init__(self, status, body=None):
        self.status_code = status
        self._body = body if body is not None else {}
        self.text = str(self._body)

    def json(self):
        return self._body


@pytest.fixture
def http(monkeypatch):
    """Drive the real _post: key set, sleeps skipped, calls recorded."""
    calls = []
    replies = []

    def _post(url, json, timeout, headers):
        calls.append({"url": url, "json": json, "headers": headers})
        return replies.pop(0)

    monkeypatch.setattr(ds, "provider_env",
                        lambda name: "sk-test" if name == "DEEPSEEK_API_KEY" else "")
    monkeypatch.setattr(ds.httpx, "post", _post)
    monkeypatch.setattr(ds.time, "sleep", lambda s: None)
    return calls, replies


def test_post_retries_429_then_succeeds(http):
    calls, replies = http
    replies += [_Resp(429), _Resp(429), _Resp(200, {"results": []})]
    assert ds._post("search", {"query": "q"}) == {"results": []}
    assert len(calls) == 3
    assert calls[0]["url"] == "https://api.deepseek.com/v1/tools/search"
    assert calls[0]["headers"]["Authorization"] == "Bearer sk-test"


def test_post_gives_up_after_the_retries(http):
    calls, replies = http
    replies += [_Resp(429, {"error": "rate limited"})] * 3
    with pytest.raises(ValueError):
        ds._post("search", {"query": "q"})
    assert len(calls) == 3


def test_post_raises_on_server_error(http):
    _, replies = http
    replies.append(_Resp(500, {"error": "boom"}))
    with pytest.raises(ValueError):
        ds._post("fetch", {"url": "https://x.example"})


@pytest.mark.parametrize("base", ["https://api.deepseek.com/v1", "https://api.deepseek.com/v1/",
                                  "https://api.deepseek.com"])
def test_chat_style_base_url_is_not_doubled(http, monkeypatch, base):
    calls, replies = http
    monkeypatch.setattr(ds, "provider_env",
                        lambda name: {"DEEPSEEK_API_KEY": "sk-test", "DEEPSEEK_BASE_URL": base}[name])
    replies.append(_Resp(200, {"results": []}))
    ds._post("search", {"query": "q"})
    assert calls[0]["url"] == "https://api.deepseek.com/v1/tools/search"


def test_malformed_rows_do_not_sink_the_search(monkeypatch):
    monkeypatch.setattr(ds, "_post", lambda endpoint, payload: {
        "results": ["junk", {"title": "A", "url": "https://a.example", "rerank_score": "high"},
                    {"title": "B", "url": "https://b.example", "rerank_score": 1.0}]
    })
    out = ds.DeepSeekWebSearchProvider().search("q", 5)
    assert out["success"] is True
    assert [r["title"] for r in out["data"]["web"]] == ["B", "A"]


def test_extract_stops_at_an_interrupt(monkeypatch):
    seen = []
    flags = iter([False, True])
    monkeypatch.setattr(ds, "_interrupted", lambda: next(flags))
    monkeypatch.setattr(ds, "_post", lambda e, p: seen.append(p["url"]) or {"content": "x"})
    docs = ds.DeepSeekWebSearchProvider().extract(["https://1.example", "https://2.example"])
    assert seen == ["https://1.example"]
    assert docs[1]["error"] == "interrupted"
