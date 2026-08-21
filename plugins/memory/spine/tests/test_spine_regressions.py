"""Regression tests for defects found by two independent reviews of spine.

Every test here corresponds to a bug that actually shipped. The plugin had no
tests at all, so each fix rested on a claim in a commit message. These pin the
behaviour instead.

Run: ~/.hermes/hermes-agent/venv/bin/python -m pytest plugins/memory/spine/tests -q
"""
import json
import os
import sqlite3
import sys
import tempfile
import threading
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))))
sys.path.insert(0, os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))
# spine/ itself: eval_run.py and sync_claude_memories.py are scripts, not part
# of the package, so they import as top-level modules.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from spine import loops  # noqa: E402


# ── hot-core file format ──────────────────────────────────────────────

def _round_trips(raw: str) -> bool:
    """The exact check memory_tool._detect_external_drift performs."""
    parts = [e.strip() for e in raw.split("\n§\n") if e.strip()]
    return raw.strip() == "\n§\n".join(parts)


def _hotcore(tmp_path_str, body):
    open(tmp_path_str, "w", encoding="utf-8").write(body)
    loops.HOTCORE_PATH = tmp_path_str


def test_promote_keeps_the_file_round_trippable():
    """A raw append produced '\\n\\n§\\n', which makes memory() refuse ALL writes."""
    f = tempfile.mktemp(suffix=".md")
    _hotcore(f, "[F] alpha\n§\n[F] beta\n")
    assert _round_trips(open(f).read())
    for i in range(3):
        assert loops._promote_to_hotcore(f"id{i}", f"entry {i}", "fact", None) == "written"
        assert _round_trips(open(f).read()), f"drifted after promote {i}"
    os.remove(f)


def test_promote_is_tristate():
    """duplicate must be distinguishable from failed, or rows strand as 'active'."""
    f = tempfile.mktemp(suffix=".md")
    _hotcore(f, "[F] alpha\n")
    assert loops._promote_to_hotcore("a", "brand new", "fact", None) == "written"
    assert loops._promote_to_hotcore("a", "brand new", "fact", None) == "duplicate"
    loops.HOTCORE_PATH = "/nonexistent-dir-xyz/MEMORY.md"
    assert loops._promote_to_hotcore("a", "x", "fact", None) == "failed"
    os.remove(f)


def test_promote_dedupe_is_tag_insensitive():
    """A re-tagged copy of an existing block must not append a second one.

    sync_hotcore re-remembers hot-core blocks as type "fact", so "[W] x" comes
    back as "[F] x". A full-string comparison never matched and the file grew a
    duplicate every run -- a growth engine inside the prompt injected on every
    call.
    """
    f = tempfile.mktemp(suffix=".md")
    _hotcore(f, "[W] Chin prefers short replies\n§\n[R] never use em dashes\n")
    before = len(loops._read_hotcore_blocks(f))
    # same bodies, different type -> different tag
    assert loops._promote_to_hotcore("a", "Chin prefers short replies", "fact", None) == "duplicate"
    assert loops._promote_to_hotcore("b", "never use em dashes", "fact", None) == "duplicate"
    assert len(loops._read_hotcore_blocks(f)) == before, "dedupe appended a re-tagged copy"
    os.remove(f)


def test_excise_removes_exactly_one_block():
    blocks = ["[R] shared text", "[F] shared text", "[W] other"]
    out = loops._excise_block(blocks, "shared text")
    assert len(out) == 2, "removed more than one block"
    assert loops._excise_block(["[F] a"], "absent") is None


def test_hotcore_lock_serialises_a_concurrent_writer():
    """The demote rewrite must not discard a memory() append landing mid-pass."""
    f = tempfile.mktemp(suffix=".md")
    _hotcore(f, "[F] alpha\n§\n[F] beta\n")
    order = []

    def rewriter():
        with loops._hotcore_lock():
            order.append("rewrite:start")
            time.sleep(0.6)
            blocks = loops._read_hotcore_blocks(f)
            loops._write_hotcore_blocks(f, [b for b in blocks if "beta" not in b])
            order.append("rewrite:done")

    def appender():
        time.sleep(0.2)
        with loops._hotcore_lock():
            order.append("append:acquired")
            blocks = loops._read_hotcore_blocks(f)
            loops._write_hotcore_blocks(f, blocks + ["[W] concurrent"])

    t1, t2 = threading.Thread(target=rewriter), threading.Thread(target=appender)
    t1.start(); t2.start(); t1.join(); t2.join()
    body = open(f).read()
    assert order.index("rewrite:done") < order.index("append:acquired")
    assert "concurrent" in body, "concurrent write was destroyed"
    assert "beta" not in body, "excision did not happen"
    assert _round_trips(body)
    os.remove(f)


# ── eval scorer ───────────────────────────────────────────────────────

def _judge(case, docs):
    """Score through eval_run's REAL judge(), not a reimplementation."""
    import eval_run
    return eval_run.judge(case, [{"content": d} for d in docs])


def test_min_sources_rejects_one_doc_holding_every_needle():
    case = {"id": "t", "q": "q", "hop": "multi",
            "expect_all": ["alpha", "beta"], "min_sources": 2}
    r = _judge(case, ["alpha and beta together", "alpha only"])
    assert r["n_sources"] == 1, f"expected 1 contributor, got {r['n_sources']}"
    assert not r["passed"]


def test_min_sources_accepts_two_genuine_contributors():
    case = {"id": "t", "q": "q", "hop": "multi",
            "expect_all": ["alpha", "beta"], "min_sources": 2}
    r = _judge(case, ["alpha only", "beta only"])
    assert r["n_sources"] == 2 and r["passed"]


def test_conj_ignores_source_count_but_still_needs_every_needle():
    case = {"id": "t", "q": "q", "hop": "conj", "expect_all": ["alpha", "beta"]}
    assert _judge(case, ["alpha and beta"])["passed"]
    assert not _judge(case, ["alpha only"])["passed"]


# ── canonical store ───────────────────────────────────────────────────

def test_sync_carries_status_and_age_forward():
    """Records are rebuilt as 'active'; a demote must not be reverted."""
    import sync_claude_memories as sync
    fresh = {"content": "same", "status": "active",
             "created_at": "2026-08-01", "last_confirmed": "2026-08-01"}
    prior = {"content": "same", "status": "demoted",
             "created_at": "2026-01-01", "last_confirmed": "2026-02-02"}
    assert sync.merge_prior(dict(fresh), None) == "added"
    r = dict(fresh)
    assert sync.merge_prior(r, prior) == "unchanged"
    assert r["status"] == "demoted", "status reverted to active"
    assert r["created_at"] == "2026-01-01", "age was reset"
    r2 = dict(fresh)
    assert sync.merge_prior(r2, dict(prior, content="different")) == "changed"
    assert r2["status"] == "demoted" and r2["created_at"] == "2026-01-01"


def test_wildcard_profile_never_produces_a_star_filename():
    """A literal '*.jsonl' would be invisible to every rebuild."""
    from spine.tools import _get_writer
    from spine.config import SpineConfig
    w = _get_writer(SpineConfig(canonical_root=tempfile.mkdtemp()), "*")
    assert "*" not in str(w._path), f"wildcard leaked into {w._path}"
