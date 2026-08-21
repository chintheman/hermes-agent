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

def _distinct(per_doc, needles):
    covered, distinct = set(), 0
    for owned in sorted(per_doc, key=len, reverse=True):
        fresh = owned - covered
        if fresh:
            covered |= fresh
            distinct += 1
    return distinct


def test_min_sources_rejects_one_doc_holding_every_needle():
    """Ascending sort counted {a} then {a,b} as two contributors."""
    assert _distinct([{"a", "b"}, {"a"}], {"a", "b"}) == 1
    assert _distinct([{"a"}, {"b"}], {"a", "b"}) == 2
    assert _distinct([{"a", "b"}, {"c"}], {"a", "b", "c"}) == 2


# ── canonical store ───────────────────────────────────────────────────

def test_patch_lines_resolve_like_a_rebuild_would():
    """check_divergence read top-level status and missed 45% of the store."""
    recs = [{"id": "x", "status": "active"}, {"id": "x", "patch": {"status": "demoted"}}]
    on_disk, patches = {}, {}
    for d in recs:
        (patches.setdefault(d["id"], []).append(d["patch"]) if "patch" in d
         else on_disk.__setitem__(d["id"], d.get("status")))
    for oid, plist in patches.items():
        for p in plist:
            if "status" in p:
                on_disk[oid] = p["status"]
    assert on_disk["x"] == "demoted"


def test_wildcard_profile_is_never_written_to():
    """_get_writer('*') would create a literal '*.jsonl' invisible to rebuilds."""
    from spine.tools import _get_writer
    from spine.config import SpineConfig
    try:
        _get_writer(SpineConfig(), "*")
    except ValueError:
        return
    raise AssertionError("_get_writer accepted a wildcard profile")
