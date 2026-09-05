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
