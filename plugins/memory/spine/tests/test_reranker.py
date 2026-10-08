"""Re-ranker (reranker.py): blends hybrid order with a cross-encoder, fails open."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from spine import reranker  # noqa: E402


class _FakeModel:
    def __init__(self, scores):
        self.scores = scores

    def predict(self, pairs, **_):
        return [self.scores[text] for _, text in pairs]


def _rows(*names):
    return [{"id": n, "content": n} for n in names]


def test_strong_model_score_lifts_a_low_hybrid_candidate(monkeypatch):
    # "answer" sits last in hybrid order; the model is sure it is the best match.
    rows = _rows("a", "b", "c", "answer")
    monkeypatch.setattr(reranker, "_load", lambda name: _FakeModel(
        {"a": 0.1, "b": 0.0, "c": -1.0, "answer": 9.0}))
    out = [r["id"] for r in reranker.rerank("q", rows)]
    assert out.index("answer") < 3, out
    assert sorted(out) == sorted(r["id"] for r in rows), "rerank dropped or duplicated rows"


def test_agreeing_model_keeps_hybrid_order(monkeypatch):
    rows = _rows("a", "b", "c")
    monkeypatch.setattr(reranker, "_load", lambda name: _FakeModel({"a": 3, "b": 2, "c": 1}))
    assert [r["id"] for r in reranker.rerank("q", rows)] == ["a", "b", "c"]


def test_unloadable_model_fails_open(monkeypatch):
    rows = _rows("a", "b", "c")
    monkeypatch.setattr(reranker, "_load", lambda name: None)
    assert reranker.rerank("q", rows) == rows


def test_predict_error_fails_open(monkeypatch):
    class Boom:
        def predict(self, *a, **k):
            raise RuntimeError("model exploded")
    rows = _rows("a", "b")
    monkeypatch.setattr(reranker, "_load", lambda name: Boom())
    assert reranker.rerank("q", rows) == rows


def test_search_hybrid_without_pool_never_loads_the_model(monkeypatch, tmp_path):
    """rerank_pool=0 (the default) must be byte-identical to the old path."""
    from spine.index import MemoryIndex

    called = []
    monkeypatch.setattr(reranker, "rerank", lambda *a, **k: called.append(1) or a[1])
    idx = MemoryIndex(str(tmp_path / "m.db"))
    idx.open()
    try:
        idx.search_hybrid("anything", None, profile="*", k=5)
        idx.search_hybrid("anything", [0.0] * 768, profile="*", k=5)
    finally:
        idx.close()
    assert not called


def test_demoted_row_ranks_below_an_identical_active_row(tmp_path):
    """Demoted stays searchable but yields to active (index.DEMOTED_WEIGHT)."""
    from spine.index import MemoryIndex, DEMOTED_WEIGHT

    assert 0 < DEMOTED_WEIGHT < 1
    idx = MemoryIndex(str(tmp_path / "m.db"))
    idx.open()
    try:
        now = "2026-10-07T00:00:00+00:00"
        for oid, status in (("A_DEMOTED", "demoted"), ("B_ACTIVE", "active")):
            idx.upsert_observation({"id": oid, "profile": "agent:main", "type": "fact",
                                    "content": "quokka ledger reconciliation rule", "status": status,
                                    "created_at": now, "last_confirmed": now}, [0.1] * 768)
        idx.conn.commit()
        hits = [h["id"] for h in idx.search_hybrid("quokka ledger", [0.1] * 768, profile="*", k=5)]
    finally:
        idx.close()
    assert hits.index("B_ACTIVE") < hits.index("A_DEMOTED"), hits


def test_failed_load_is_retried_after_the_backoff(monkeypatch):
    calls = []

    class Fake:
        def __init__(self, *a, **k):
            calls.append(1)
            if len(calls) == 1:
                raise OSError("transient download failure")

    import types
    monkeypatch.setitem(sys.modules, "sentence_transformers",
                        types.SimpleNamespace(CrossEncoder=Fake))
    monkeypatch.setattr(reranker, "_model", None)
    monkeypatch.setattr(reranker, "_failed_at", 0.0)
    clock = [1000.0]
    monkeypatch.setattr(reranker.time, "monotonic", lambda: clock[0])
    assert reranker._load("m") is None            # first load fails
    assert reranker._load("m") is None and len(calls) == 1   # inside backoff: no retry
    clock[0] += reranker._RETRY_AFTER_S + 1
    assert reranker._load("m") is not None and len(calls) == 2  # retried and recovered


def test_warm_up_runs_once_in_the_background(monkeypatch):
    import threading as _th
    import spine as spine_mod
    from spine.config import SpineConfig

    started = []

    class FakeThread:
        def __init__(self, target, name, daemon):
            assert daemon, "warm-up must never keep the process alive"
            self.target = target
        def start(self):
            started.append(self.target)

    monkeypatch.setattr(spine_mod, "_warm_started", False)
    monkeypatch.setattr(spine_mod.threading, "Thread", FakeThread)
    loaded = []
    monkeypatch.setattr(reranker, "_load", lambda name: loaded.append(name))
    import spine.embedder as emb
    monkeypatch.setattr(emb, "embedder_available", lambda: True)
    cfg = SpineConfig(rerank_pool=50)
    spine_mod._warm_models_once(cfg)
    spine_mod._warm_models_once(cfg)           # second session: no second thread
    assert len(started) == 1
    started[0]()                               # run the body synchronously
    assert loaded == [cfg.rerank_model]
