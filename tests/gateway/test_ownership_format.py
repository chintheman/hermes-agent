"""Tests for gateway/ownership_format.py — the deterministic pre-send collapse of every labelled
MINE/YOURS line into ONE block at the end (Chin, 2026-09-27: "I need 1 summary. 1 yours and
mine"). The rule lives in ~/wiki/_system/ownership-format.json, shared with Claude Code's own
Stop hook (~/.hermes/hooks/one-ownership-block.py); this module is the deterministic backstop on
the Hermes send path, independent of what SKIM_MASTER_RULES_GUIDANCE asked the model to do.
"""

import json

import pytest

from gateway import ownership_format as of


def test_single_block_untouched():
    """A message that already has exactly one canonical MINE-then-YOURS block at the end is
    byte-identical after formatting — the pipeline is idempotent on already-correct input."""
    text = "Fixed the bug.\n\nMINE: shipped the patch\nYOURS: review the PR"
    assert of.format_ownership_block(text) == text


def test_no_labels_untouched():
    text = "Just a normal reply with no ownership block at all."
    assert of.format_ownership_block(text) == text


def test_two_blocks_merged_into_one_at_end():
    """Multiple MINE/YOURS lines scattered through the message collapse into one block, MINE
    first then YOURS, at the very end."""
    text = "MINE: fixed A\n\nsome text\n\nYOURS: check B\n\nmore text\n\nMINE: fixed C"
    result = of.format_ownership_block(text)
    assert result == "some text\n\nmore text\n\nMINE: fixed A; fixed C\nYOURS: check B"
    # Exactly one MINE line and one YOURS line, both at the very end.
    lines = result.splitlines()
    assert lines[-2].startswith("MINE:")
    assert lines[-1].startswith("YOURS:")
    assert sum(l.startswith("MINE:") for l in lines) == 1
    assert sum(l.startswith("YOURS:") for l in lines) == 1


def test_duplicate_lines_deduped():
    text = "MINE: shipped it\n\nbody\n\nMINE: shipped it"
    assert of.format_ownership_block(text) == "body\n\nMINE: shipped it"


def test_duplicate_lines_deduped_case_and_whitespace_exact_only():
    """Only EXACT duplicate values are dropped; near-duplicates are joined, not deduped, since
    dropping them silently could hide a genuinely different follow-up."""
    text = "MINE: shipped it\n\nbody\n\nMINE: Shipped it"
    result = of.format_ownership_block(text)
    assert result == "body\n\nMINE: shipped it; Shipped it"


def test_mine_nothing_running_dropped():
    text = "All good.\n\nMINE: nothing running\nYOURS: approve the PR"
    assert of.format_ownership_block(text) == "All good.\n\nYOURS: approve the PR"


@pytest.mark.parametrize("value", ["nothing", "none", "n/a", "nothing pending.", "Nothing Yet"])
def test_empty_values_are_case_insensitively_dropped(value):
    text = f"Status.\n\nMINE: {value}\nYOURS: take a look"
    result = of.format_ownership_block(text)
    assert "MINE:" not in result
    assert "YOURS: take a look" in result


def test_both_empty_gives_no_block():
    text = "Status update.\n\nMINE: nothing\nYOURS: none"
    assert of.format_ownership_block(text) == "Status update."


def test_single_label_allowed_mine_only():
    text = "Done.\n\nMINE: shipped it"
    assert of.format_ownership_block(text) == "Done.\n\nMINE: shipped it"


def test_single_label_allowed_yours_only():
    text = "Done.\n\nYOURS: approve the PR"
    assert of.format_ownership_block(text) == "Done.\n\nYOURS: approve the PR"


def test_labels_inside_fenced_code_block_untouched():
    """A pasted code example containing the literal text "MINE:" must never be treated as a
    real ownership line — even though a REAL block exists outside the fence."""
    text = "Here:\n\n```\nMINE: not a real label\n```\n\nMINE: real one"
    result = of.format_ownership_block(text)
    assert result == "Here:\n\n```\nMINE: not a real label\n```\n\nMINE: real one"


def test_labels_inside_tilde_fence_untouched():
    text = "~~~\nMINE: also protected\n~~~\n\nMINE: the real one"
    result = of.format_ownership_block(text)
    assert "MINE: also protected" in result  # still inside the fence, untouched
    assert result.endswith("MINE: the real one")


def test_lowercase_mine_yours_in_prose_untouched():
    """Case-sensitive matching: ordinary English "mine"/"yours" must never be treated as labels."""
    text = "this decision is mine to make, not yours. The call is mine."
    assert of.format_ownership_block(text) == text


def test_bold_label_value_not_polluted_by_closing_markup():
    """The canonical bold form from Hermes's own Skim Master Rules guidance prompt:
    **MINE:** value — the closing ``**`` sits right after the colon and must not leak into the
    extracted value."""
    text = "**MINE:** fixed it\n\nbody text\n\n**YOURS:** approve"
    result = of.format_ownership_block(text)
    assert result == "body text\n\nMINE: fixed it\nYOURS: approve"


def test_spec_missing_falls_back_to_defaults(monkeypatch, tmp_path):
    """A spec path that does not exist on disk falls back to the built-in default spec, and
    formatting behavior is unchanged."""
    monkeypatch.setattr(of, "_SPEC_PATH", tmp_path / "does-not-exist.json")
    text = "Body.\n\nMINE: nothing running\nYOURS: approve"
    assert of.format_ownership_block(text) == "Body.\n\nYOURS: approve"


def test_spec_unparseable_falls_back_to_defaults(monkeypatch, tmp_path):
    spec_path = tmp_path / "ownership-format.json"
    spec_path.write_text("{ not valid json")
    monkeypatch.setattr(of, "_SPEC_PATH", spec_path)
    text = "Body.\n\nMINE: nothing\nYOURS: approve"
    assert of.format_ownership_block(text) == "Body.\n\nYOURS: approve"


def test_spec_missing_labels_key_falls_back_to_defaults(monkeypatch, tmp_path):
    spec_path = tmp_path / "ownership-format.json"
    spec_path.write_text(json.dumps({"version": 1}))
    monkeypatch.setattr(of, "_SPEC_PATH", spec_path)
    text = "Body.\n\nMINE: shipped it"
    assert of.format_ownership_block(text) == "Body.\n\nMINE: shipped it"


def test_spec_reloads_on_mtime_change(monkeypatch, tmp_path):
    """The spec is cached, but a real mtime change (an operator editing the file) is picked up
    at the next call without a restart."""
    spec_path = tmp_path / "ownership-format.json"
    spec_path.write_text(json.dumps(dict(of._DEFAULT_SPEC, empty_values=["nothing"])))
    monkeypatch.setattr(of, "_SPEC_PATH", spec_path)
    of._spec_cache["mtime"] = None
    of._spec_cache["spec"] = None

    assert "MINE:" not in of.format_ownership_block("Body.\n\nMINE: nothing")

    import os
    import time

    time.sleep(0.01)
    new_spec = dict(of._DEFAULT_SPEC, empty_values=["some-other-token"])  # "nothing" no longer drops
    spec_path.write_text(json.dumps(new_spec))
    os.utime(spec_path, None)

    assert "MINE: nothing" in of.format_ownership_block("Body.\n\nMINE: nothing")


def test_formatter_exception_sends_original_text(monkeypatch):
    """Any internal failure (spec load, regex, anything) must never break a send: the ORIGINAL
    text goes out unchanged, logged rather than raised."""

    def _boom(*_a, **_kw):
        raise RuntimeError("boom")

    monkeypatch.setattr(of, "_load_spec", _boom)
    text = "Body.\n\nMINE: shipped it\nYOURS: approve"
    assert of.format_ownership_block(text) == text


def test_formatter_exception_with_explicit_spec_still_safe(monkeypatch):
    """Even with an explicit (already-loaded) spec passed in, a failure inside the formatting
    logic itself still degrades to the original text rather than raising."""
    monkeypatch.setattr(of, "_protected_line_flags", lambda *_a, **_kw: (_ for _ in ()).throw(RuntimeError("boom")))
    text = "Body.\n\nMINE: shipped it"
    assert of.format_ownership_block(text, spec=of._DEFAULT_SPEC) == text


def test_non_string_input_returned_as_is():
    assert of.format_ownership_block(None) is None
    assert of.format_ownership_block("") == ""
