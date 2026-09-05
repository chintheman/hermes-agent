"""Tests for hot-core rule scoping (hotcore-retrieval-split, phase 2).

The failure that matters here is silent: a behavioural rule that stops reaching
the model makes the agent misbehave with nobody able to say why. So the tests
lean on the safe-default and never-defer guarantees, not just the happy path.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))))
sys.path.insert(0, os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))

from spine.rule_scope import (  # noqa: E402
    IMAGE, SUBJECT, TOOL, UNIVERSAL, TurnContext, fires, parse_trigger,
    reachable_kinds, select, split_blocks, summarise,
)


# ── parse_trigger: unclassified and malformed must both be universal ──

def test_no_marker_is_universal():
    assert parse_trigger("[C] Execute, do not hand homework.").kind == UNIVERSAL


def test_unknown_kind_falls_back_to_universal():
    # A typo must not silently delete a rule from every prompt.
    assert parse_trigger("[W] thing @when:planetary=mars").kind == UNIVERSAL


def test_valueless_subject_is_universal():
    # "@when:subject" with nothing to match could never fire.
    assert parse_trigger("[W] thing @when:subject").kind == UNIVERSAL


def test_parses_kinds_and_values():
    t = parse_trigger("[W] ledger note @when:subject=ledger,prediction")
    assert t.kind == SUBJECT and t.values == ("ledger", "prediction")
    assert parse_trigger("[W] garmin @when:tool=garmin_*").kind == TOOL
    assert parse_trigger("[C] screenshot rule @when:image").kind == IMAGE


# ── fires: triggers are facts, not guesses ──

def test_image_trigger_matches_payload_only():
    t = parse_trigger("[C] screenshot @when:image")
    assert fires(t, TurnContext(query="anything", has_image=True))
    assert not fires(t, TurnContext(query="anything", has_image=False))


def test_tool_trigger_matches_glob_and_nothing_else():
    t = parse_trigger("[W] garmin @when:tool=garmin_*")
    assert fires(t, TurnContext(tool_name="garmin_get_activities"))
    assert not fires(t, TurnContext(tool_name="web_search"))
    assert not fires(t, TurnContext())  # no tool this turn


def test_subject_trigger_respects_word_boundaries():
    t = parse_trigger("[W] ledger @when:subject=ledger")
    assert fires(t, TurnContext(query="check the ledger please"))
    assert fires(t, TurnContext(query="LEDGER grading"))
    # must not match inside another word
    assert not fires(t, TurnContext(query="the ledgerdemain of it"))
    assert not fires(t, TurnContext(query="unrelated question"))


def test_universal_always_fires_even_on_an_empty_turn():
    assert fires(parse_trigger("[C] always"), TurnContext())


# ── select: the two safety guarantees ──

def test_untagged_block_is_never_deferred():
    hot, deferred = select(["[C] no marker here"], TurnContext(query="x"))
    assert hot and not deferred


def test_rule_block_is_never_deferred_even_if_tagged():
    # [R] is the constitution. A mis-tagged rule must still load.
    blocks = ["[R] durable observations go to remember() @when:image"]
    hot, deferred = select(blocks, TurnContext(has_image=False))
    assert hot == blocks and deferred == []


def test_irrelevant_block_defers_and_relevant_one_does_not():
    blocks = [
        "[W] garmin restart @when:tool=garmin_*",
        "[C] screenshot rule @when:image",
        "[C] universal correction",
    ]
    hot, deferred = select(blocks, TurnContext(query="hi", tool_name="garmin_get"))
    assert blocks[0] in hot        # its tool is being called
    assert blocks[2] in hot        # universal
    assert blocks[1] in deferred   # no image this turn


def test_a_trigger_that_always_fires_is_not_a_trigger():
    # Negative control for the whole design: on a turn with no query, no tool
    # and no image, only universal/[R] content should survive.
    blocks = [
        "[W] a @when:subject=ledger",
        "[W] b @when:tool=garmin_*",
        "[C] c @when:image",
        "[R] d",
        "[C] e",
    ]
    hot, deferred = select(blocks, TurnContext())
    assert sorted(hot) == sorted(["[R] d", "[C] e"])
    assert len(deferred) == 3


# ── plumbing ──

def test_split_blocks_handles_the_section_separator():
    assert split_blocks("[C] one\n§\n[W] two\n§\n") == ["[C] one", "[W] two"]


def test_summarise_reports_byte_savings():
    blocks = ["[C] universal", "[W] x @when:image"]
    s = summarise(blocks, TurnContext(query="q", has_image=False))
    assert s["hot_blocks"] == 1 and s["deferred_blocks"] == 1
    assert s["deferred_bytes"] > 0
    assert s["deferred_triggers"] == ["image:"]


def test_reachable_kinds_covers_every_kind_fires_understands():
    for kind in reachable_kinds():
        assert kind in {UNIVERSAL, SUBJECT, TOOL, IMAGE}


# ── deliverable(): the snapshot + delivery must lose nothing ──

from spine.rule_scope import deliverable  # noqa: E402

SAMPLE = [
    "[R] constitution rule",
    "[C] universal correction",
    "[W] garmin note @when:tool=garmin_*",
    "[C] screenshot rule @when:image",
    "[W] ledger note @when:subject=ledger",
]


def _snapshot():
    """What the frozen system-prompt snapshot holds: empty-context selection."""
    hot, _ = select(SAMPLE, TurnContext())
    return hot


def test_snapshot_is_exactly_universal_and_rules():
    assert _snapshot() == ["[R] constitution rule", "[C] universal correction"]


def test_delivery_never_duplicates_the_snapshot():
    extra = deliverable(SAMPLE, TurnContext(query="ledger check"))
    assert not set(extra) & set(_snapshot())


def test_nothing_is_lost_for_any_turn():
    """snapshot + delivered must cover every block that fires. The whole design
    rests on this: a rule that fires but reaches neither is silently gone."""
    contexts = [
        TurnContext(),
        TurnContext(query="ledger check"),
        TurnContext(query="anything", has_image=True),
        TurnContext(tool_name="garmin_get_activities"),
        TurnContext(query="ledger", tool_name="garmin_x", has_image=True),
    ]
    for ctx in contexts:
        should_fire, _ = select(SAMPLE, ctx)
        delivered = set(_snapshot()) | set(deliverable(SAMPLE, ctx))
        missing = set(should_fire) - delivered
        assert not missing, f"lost {missing} for {ctx}"


def test_delivery_is_empty_when_nothing_extra_fires():
    assert deliverable(SAMPLE, TurnContext(query="totally unrelated")) == []


def test_rule_blocks_are_never_delivered_twice():
    # [R] is always in the snapshot, so it must never appear in the delta.
    extra = deliverable(SAMPLE, TurnContext(query="ledger", has_image=True))
    assert not any(b.startswith("[R]") for b in extra)


# ── the dangerous one: filtering the snapshot must never reach disk ──

def test_scope_filter_never_shrinks_what_gets_persisted(tmp_path, monkeypatch):
    """If a write path ever read the filtered snapshot, enabling scoping would
    delete every tagged rule from MEMORY.md. save_to_disk must always persist
    live entries, which the filter does not touch."""
    import tools.memory_tool as mt

    mem_dir = tmp_path / "memories"
    mem_dir.mkdir()
    blocks = [
        "[R] constitution rule",
        "[C] universal correction",
        "[W] garmin note @when:tool=garmin_*",
        "[C] screenshot rule @when:image",
    ]
    (mem_dir / "MEMORY.md").write_text("\n§\n".join(blocks), encoding="utf-8")
    (mem_dir / "USER.md").write_text("", encoding="utf-8")
    monkeypatch.setattr(mt, "get_memory_dir", lambda: mem_dir)

    store = mt.MemoryStore(memory_char_limit=90000, user_char_limit=1375,
                           memory_enabled=True, user_profile_enabled=True,
                           scope_filter=True)
    store.load_from_disk()

    # The snapshot IS filtered ...
    snap = store._system_prompt_snapshot["memory"]
    assert "@when:" not in snap
    assert "garmin note" not in snap

    # ... but live state is whole, and so is anything persisted from it.
    assert len(store.memory_entries) == len(blocks)
    store.save_to_disk("memory")
    on_disk = (mem_dir / "MEMORY.md").read_text(encoding="utf-8")
    for b in blocks:
        assert b in on_disk, f"scope filter destroyed {b!r} on disk"


def test_scope_filter_off_is_byte_identical_to_unfiltered(tmp_path, monkeypatch):
    """The flag must be a true no-op when off — the revert path depends on it."""
    import tools.memory_tool as mt

    mem_dir = tmp_path / "memories"
    mem_dir.mkdir()
    (mem_dir / "MEMORY.md").write_text(
        "[C] plain\n§\n[W] tagged @when:image", encoding="utf-8")
    (mem_dir / "USER.md").write_text("", encoding="utf-8")
    monkeypatch.setattr(mt, "get_memory_dir", lambda: mem_dir)

    def build(flag):
        s = mt.MemoryStore(memory_char_limit=90000, user_char_limit=1375,
                           memory_enabled=True, user_profile_enabled=True,
                           scope_filter=flag)
        s.load_from_disk()
        return s._system_prompt_snapshot["memory"]

    assert "tagged" in build(False)
    assert "tagged" not in build(True)
