"""Tests for the bundled verbatim-reply plugin (plugins/verbatim-reply/).

Covers each hook, the dash scrub, path-escape attempts, the skill lock, and end-to-end tests that
feed a tool result through the real transform_tool_result path and then through
agent.turn_finalizer.apply_llm_output_transform.
"""

import importlib.util
import json
import logging
import os
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

REPO = Path(__file__).resolve().parents[2]
ID1 = "20261019T104416Z970a163e"
ID2 = "20261019T104417Zdeadbeef"


def _load():
    spec = importlib.util.spec_from_file_location(
        "verbatim_reply_under_test", REPO / "plugins" / "verbatim-reply" / "__init__.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture
def vr(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("HERMES_VERBATIM_DIR", str(tmp_path / "verbatim"))
    (tmp_path / "verbatim").mkdir()
    mod = _load()
    mod._PENDING.clear()
    mod._LATEST_USER.clear()
    return mod


def stage(vr, vid, text):
    p = vr.verbatim_dir() / (vid + ".md")
    p.write_text(text)
    return p


# ---------------------------------------------------------------------------
# transform_tool_result
# ---------------------------------------------------------------------------

class TestToolResult:
    def run(self, vr, result, session="s1", turn="t1"):
        return vr._on_transform_tool_result(tool_name="terminal", args={}, result=result, session_id=session, turn_id=turn)

    def test_json_id_is_collected_and_the_result_is_never_changed(self, vr):
        stage(vr, ID1, "Card on its way, under a minute.\n")
        res = json.dumps({"queued": "/x", "message": "Card on its way, under a minute.", "verbatim_id": ID1})
        assert self.run(vr, res) is None
        assert vr._PENDING[("s1", "t1")]["texts"] == ["Card on its way, under a minute."]

    def test_id_inside_an_escaped_json_string(self, vr):
        stage(vr, ID1, "hello")
        inner = json.dumps({"verbatim_id": ID1})
        outer = json.dumps({"output": inner, "exit_code": 0})
        assert '\\"verbatim_id\\"' in outer
        self.run(vr, outer)
        assert vr._PENDING[("s1", "t1")]["ids"] == [ID1]

    def test_marker_lines_first_and_last_dedupe_to_one(self, vr):
        stage(vr, ID1, "**Done**\nline")
        out = "[verbatim %s]\n**Done**\nline\n[verbatim %s]" % (ID1, ID1)
        self.run(vr, json.dumps({"output": out}))
        assert vr._PENDING[("s1", "t1")]["texts"] == ["**Done**\nline"]

    def test_marker_survives_head_or_tail(self, vr):
        stage(vr, ID1, "body")
        self.run(vr, "[verbatim %s]\nbody" % ID1)
        self.run(vr, "body\n[verbatim %s]" % ID1, turn="t2")
        assert vr._PENDING[("s1", "t1")]["texts"] == vr._PENDING[("s1", "t2")]["texts"] == ["body"]

    def test_order_is_kept_and_ids_are_deduped_across_calls(self, vr):
        stage(vr, ID1, "first")
        stage(vr, ID2, "second")
        self.run(vr, "[verbatim %s]" % ID1)
        self.run(vr, "[verbatim %s] [verbatim %s]" % (ID2, ID1))
        assert vr._PENDING[("s1", "t1")]["texts"] == ["first", "second"]

    def test_keyed_by_session_and_turn(self, vr):
        stage(vr, ID1, "a")
        self.run(vr, "[verbatim %s]" % ID1, session="s1", turn="t1")
        assert ("s1", "t1") in vr._PENDING
        assert ("s2", "t1") not in vr._PENDING and ("s1", "t2") not in vr._PENDING

    @pytest.mark.parametrize("vid", ["../../etc/passwd", "short", "a" * 65, "has space in it", "ok_id_01/../x"])
    def test_bad_ids_are_ignored(self, vr, vid):
        self.run(vr, json.dumps({"verbatim_id": vid}))
        self.run(vr, "[verbatim %s]" % vid)
        assert not vr._PENDING

    def test_symlink_escape_is_refused(self, vr, tmp_path):
        secret = tmp_path / "outside.md"
        secret.write_text("SECRET")
        os.symlink(secret, vr.verbatim_dir() / (ID1 + ".md"))
        self.run(vr, "[verbatim %s]" % ID1)
        assert not vr._PENDING
        assert vr.read_staged(ID1) is None

    def test_missing_empty_and_stale_files_are_ignored(self, vr):
        stage(vr, ID2, "  \n")
        old = stage(vr, "20250101T000000Zaaaaaaaa", "old")
        past = time.time() - 7200
        os.utime(old, (past, past))
        for vid in (ID1, ID2, "20250101T000000Zaaaaaaaa"):
            self.run(vr, "[verbatim %s]" % vid)
        assert not vr._PENDING

    def test_fails_open_on_anything_odd(self, vr, monkeypatch):
        assert self.run(vr, None) is None
        assert self.run(vr, 12345) is None

        def boom(vid):
            raise RuntimeError("boom")

        monkeypatch.setattr(vr, "read_staged", boom)
        assert self.run(vr, "[verbatim %s]" % ID1) is None

    def test_default_dir_follows_hermes_home(self, tmp_path, monkeypatch):
        monkeypatch.delenv("HERMES_VERBATIM_DIR", raising=False)
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        assert _load().verbatim_dir() == tmp_path / "data" / "op" / "verbatim"


# ---------------------------------------------------------------------------
# transform_llm_output
# ---------------------------------------------------------------------------

class TestLlmOutput:
    def out(self, vr, text="model words", platform="telegram", session="s1", turn="t1"):
        return vr._on_transform_llm_output(response_text=text, session_id=session, model="m", platform=platform, turn_id=turn)

    def test_scripted_text_replaces_the_model_text_joined_by_a_blank_line(self, vr, caplog):
        vr._PENDING[("s1", "t1")] = {"ts": time.time(), "ids": [ID1, ID2], "texts": ["one", "two"]}
        with caplog.at_level(logging.INFO):
            assert self.out(vr) == "one\n\ntwo"
        assert "verbatim-reply: replaced model text with 2 scripted message(s)" in caplog.text
        assert ("s1", "t1") not in vr._PENDING
        assert self.out(vr) is None

    def test_scripted_text_is_not_dash_scrubbed(self, vr):
        vr._PENDING[("s1", "t1")] = {"ts": time.time(), "ids": [ID1], "texts": ["a — b"]}
        assert self.out(vr) == "a — b"

    def test_other_turns_are_untouched(self, vr):
        vr._PENDING[("s1", "t1")] = {"ts": time.time(), "ids": [ID1], "texts": ["one"]}
        assert self.out(vr, turn="t2") is None
        assert ("s1", "t1") in vr._PENDING

    def test_old_entries_are_dropped(self, vr):
        vr._PENDING[("s1", "t1")] = {"ts": time.time() - 3600, "ids": [ID1], "texts": ["stale"]}
        assert self.out(vr) is None
        assert not vr._PENDING

    def test_the_map_is_bounded(self, vr):
        for i in range(vr.PENDING_MAX + 40):
            vr._PENDING[("s", str(i))] = {"ts": time.time(), "ids": [], "texts": ["x"]}
        self.out(vr, turn="none")
        assert len(vr._PENDING) <= vr.PENDING_MAX

    def test_dashes_are_scrubbed_on_telegram_only(self, vr):
        assert self.out(vr, "fine — really") == "fine, really"
        assert self.out(vr, "fine — really", platform="cli") is None
        assert self.out(vr, "no dashes here") is None


class TestDashScrub:
    @pytest.mark.parametrize("src,want", [
        ("a — b", "a, b"),
        ("a – b", "a, b"),
        ("5–7 reps", "5-7 reps"),
        ("5 – 7 reps", "5-7 reps"),
        ("2026—10", "2026-10"),
        ("word—word", "word, word"),
        ("word–word", "word, word"),
        ("so, — then", "so, then"),
        ("trailing —\nnext", "trailing\nnext"),
        ("— item one\n— item two", "- item one\n- item two"),
        ("plain hyphen-ated - fine", "plain hyphen-ated - fine"),
    ])
    def test_plain_text(self, vr, src, want):
        assert vr.scrub_dashes(src) == want

    def test_inline_code_is_untouched(self, vr):
        src = "use `a — b` here — now"
        assert vr.scrub_dashes(src) == "use `a — b` here, now"

    def test_double_backtick_span_is_untouched(self, vr):
        assert vr.scrub_dashes("x ``a–b`` y—z") == "x ``a–b`` y, z"

    def test_fenced_blocks_are_untouched_even_unterminated(self, vr):
        src = "before — x\n```\ncode — here\n```\nafter — y"
        assert vr.scrub_dashes(src) == "before, x\n```\ncode — here\n```\nafter, y"
        assert vr.scrub_dashes("a — b\n```\nc — d") == "a, b\n```\nc — d"

    def test_tilde_fence(self, vr):
        assert vr.scrub_dashes("~~~\nx — y\n~~~") == "~~~\nx — y\n~~~"


# ---------------------------------------------------------------------------
# pre_llm_call + pre_tool_call: add-note
# ---------------------------------------------------------------------------

NOTE = "python3 ~/.hermes/skills/productivity/health-wellness-coach/scripts/coach_log.py add-note --date 2026-10-19 --text %s"
NOTE_BLOCK = {"action": "block", "message": "add-note saves Chin's own words only. Quote his message exactly, or do not save a note."}


class TestAddNote:
    def say(self, vr, msg, session="s1"):
        vr._on_pre_llm_call(session_id=session, user_message=msg)

    def pre(self, vr, cmd, session="s1", tool="terminal"):
        return vr._on_pre_tool_call(tool_name=tool, args={"command": cmd}, session_id=session)

    def test_a_quote_of_his_words_is_allowed(self, vr):
        self.say(vr, "Knee felt fine today, but my lower back was tight on the deadlift!")
        assert self.pre(vr, NOTE % "'lower back was tight on the deadlift'") is None
        assert self.pre(vr, NOTE % '"Knee felt fine today"') is None

    def test_matching_ignores_case_whitespace_and_punctuation(self, vr):
        self.say(vr, "Back   was TIGHT, honestly.")
        assert self.pre(vr, NOTE % "'back was tight honestly'") is None

    def test_invented_text_is_blocked_with_the_exact_message(self, vr):
        self.say(vr, "good session")
        assert self.pre(vr, NOTE % "'9/10 session'") == NOTE_BLOCK

    def test_a_note_from_an_earlier_message_is_blocked(self, vr):
        self.say(vr, "knee hurt")
        self.say(vr, "[photo of food]")
        assert self.pre(vr, NOTE % "'knee hurt'") == NOTE_BLOCK

    def test_no_user_message_known_blocks(self, vr):
        assert self.pre(vr, NOTE % "'anything'") == NOTE_BLOCK

    def test_sessions_are_separate(self, vr):
        self.say(vr, "hello there", session="s2")
        assert self.pre(vr, NOTE % "'hello there'", session="s1") == NOTE_BLOCK

    @pytest.mark.parametrize("cmd", [
        "python3 coach_log.py add-note --date D",
        "python3 coach_log.py add-note --text",
        "python3 coach_log.py add-note --text 'unterminated",
        "python3 coach_log.py add-note --text ''",
        "python3 coach_log.py add-note --text \"$NOTE\"",
        "python3 coach_log.py add-note --text '...'",
    ])
    def test_unparseable_or_empty_blocks(self, vr, cmd):
        self.say(vr, "knee felt fine")
        assert self.pre(vr, cmd) == NOTE_BLOCK

    def test_equals_form_and_compound_commands(self, vr):
        self.say(vr, "knee felt fine")
        assert self.pre(vr, "cd x && python3 coach_log.py add-note --text='knee felt fine' && echo ok") is None
        both = "python3 coach_log.py add-note --text 'knee felt fine'; python3 coach_log.py add-note --text 'invented'"
        assert self.pre(vr, both) == NOTE_BLOCK

    def test_other_coach_log_verbs_and_other_tools_are_untouched(self, vr):
        assert self.pre(vr, "python3 coach_log.py show --date 2026-10-19") is None
        assert self.pre(vr, "echo add-note") is None
        assert vr._on_pre_tool_call(tool_name="read_file", args={"path": "x"}, session_id="s1") is None

    def test_multimodal_user_message(self, vr):
        self.say(vr, [{"type": "text", "text": "knee felt fine"}, {"type": "image_url", "image_url": {"url": "x"}}])
        assert self.pre(vr, NOTE % "'knee felt fine'") is None


# ---------------------------------------------------------------------------
# pre_tool_call: skill lock
# ---------------------------------------------------------------------------

LOCKED = {"action": "block", "message": "This skill is locked. Changes to it go through Claude Code."}


class TestSkillLock:
    def pre(self, vr, tool, args):
        return vr._on_pre_tool_call(tool_name=tool, args=args, session_id="s1")

    @pytest.mark.parametrize("args", [
        {"operations": [{"name": "health-wellness-coach", "action": "patch", "old_string": "a", "new_string": "b"}]},
        {"operations": [{"name": "Body-Composition-Blocks", "action": "delete"}]},
        {"operations": [{"name": "other", "action": "create", "content": "x"},
                        {"name": "health-wellness-coach", "action": "remove_file", "file_path": "x"}]},
        {"name": "health-wellness-coach", "action": "write_file", "file_path": "a", "file_content": "x"},
        {"operations": [{"name": "productivity/health-wellness-coach", "action": "delete"}]},
    ])
    def test_skill_manage_on_a_locked_skill(self, vr, args):
        assert self.pre(vr, "skill_manage", args) == LOCKED

    def test_skill_manage_on_another_skill_is_allowed(self, vr):
        ok = {"operations": [{"name": "garmin", "action": "patch", "old_string": "a", "new_string": "b"}]}
        assert self.pre(vr, "skill_manage", ok) is None
        assert self.pre(vr, "skill_manage", {"operations": []}) is None

    @pytest.mark.parametrize("path", [
        "~/.hermes/skills/productivity/health-wellness-coach/SKILL.md",
        "~/.hermes/skills/productivity/health-wellness-coach/scripts/coach_log.py",
        "~/projects/health-wellness-coach/skill/SKILL.md",
        "~/.hermes/skills/health/body-composition-blocks/SKILL.md",
        "~/.hermes/skills/productivity/health-wellness-coach/../health-wellness-coach/SKILL.md",
        "~/.hermes/skills/health/../productivity/health-wellness-coach/SKILL.md",
        "skills/productivity/health-wellness-coach/SKILL.md",
        "../skills/health/body-composition-blocks/x",
    ])
    @pytest.mark.parametrize("tool", ["write_file", "patch"])
    def test_file_tools_on_locked_paths(self, vr, tool, path):
        assert self.pre(vr, tool, {"path": path, "content": "x", "old_string": "a", "new_string": "b"}) == LOCKED

    def test_symlink_into_a_locked_dir_is_resolved(self, vr, tmp_path):
        target = tmp_path / "repo" / "skill"
        target.mkdir(parents=True)
        link_parent = Path(os.environ["HERMES_HOME"]) / "skills" / "productivity"
        link_parent.mkdir(parents=True)
        os.symlink(target, link_parent / "health-wellness-coach")
        sneaky = tmp_path / "sneaky"
        os.symlink(target, sneaky)
        assert self.pre(vr, "write_file", {"path": str(sneaky / "SKILL.md"), "content": "x"}) == LOCKED
        assert self.pre(vr, "write_file", {"path": str(target / "SKILL.md"), "content": "x"}) == LOCKED

    def test_v4a_patch_paths(self, vr):
        bad = "*** Begin Patch\n*** Update File: ~/.hermes/skills/health/body-composition-blocks/SKILL.md\n@@\n-a\n+b\n*** End Patch"
        assert self.pre(vr, "patch", {"mode": "patch", "patch": bad}) == LOCKED
        ok = "*** Begin Patch\n*** Update File: ~/notes/a.md\n@@\n-a\n+b\n*** End Patch"
        assert self.pre(vr, "patch", {"mode": "patch", "patch": ok}) is None

    def test_other_paths_and_tools_are_allowed(self, vr, tmp_path):
        assert self.pre(vr, "write_file", {"path": str(tmp_path / "a.txt"), "content": "x"}) is None
        other = "~/.hermes/skills/productivity/other-skill/SKILL.md"
        assert self.pre(vr, "write_file", {"path": other, "content": "x"}) is None
        mine = "~/.hermes/skills/productivity/health-wellness-coach/SKILL.md"
        assert self.pre(vr, "read_file", {"path": mine}) is None
        run = {"command": "python3 ~/.hermes/skills/productivity/health-wellness-coach/scripts/coach_engine.py card"}
        assert self.pre(vr, "terminal", run) is None
        assert self.pre(vr, "write_file", "not a dict") is None
        assert self.pre(vr, "write_file", {"path": None}) is None


# ---------------------------------------------------------------------------
# Registration, discovery and the end-to-end path
# ---------------------------------------------------------------------------

class FakeCtx:
    def __init__(self):
        self.hooks = {}

    def register_hook(self, name, cb):
        self.hooks.setdefault(name, []).append(cb)


def _wire(vr, monkeypatch):
    ctx = FakeCtx()
    vr.register(ctx)
    monkeypatch.setattr(
        "hermes_cli.lifecycle.invoke_hook",
        lambda name, **kw: [r for r in (cb(**kw) for cb in ctx.hooks.get(name, [])) if r is not None])
    monkeypatch.setattr("hermes_cli.lifecycle.has_hook", lambda name: bool(ctx.hooks.get(name)))
    return ctx


def test_register_adds_the_four_hooks_all_valid(vr):
    from hermes_cli.plugins import VALID_HOOKS
    ctx = FakeCtx()
    vr.register(ctx)
    assert set(ctx.hooks) == {"transform_tool_result", "transform_llm_output", "pre_llm_call", "pre_tool_call"}
    assert set(ctx.hooks) <= VALID_HOOKS


def test_bundled_plugin_is_discovered_but_needs_plugins_enabled(tmp_path, monkeypatch):
    import hermes_yaml as yaml
    from hermes_cli import plugins as pmod
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    mgr = pmod.PluginManager()
    mgr.discover_and_load()
    loaded = mgr._plugins["verbatim-reply"]
    assert loaded.manifest.source == "bundled" and not loaded.enabled and "not enabled" in loaded.error
    (home / "config.yaml").write_text(yaml.safe_dump({"plugins": {"enabled": ["verbatim-reply"]}}))
    mgr = pmod.PluginManager()
    mgr.discover_and_load()
    loaded = mgr._plugins["verbatim-reply"]
    assert loaded.enabled and not loaded.error
    for hook in ("transform_tool_result", "transform_llm_output", "pre_llm_call", "pre_tool_call"):
        assert mgr._hooks.get(hook), hook


def test_end_to_end_tool_result_to_final_reply(vr, monkeypatch):
    """A staged id in a terminal result reaches Chin as the staged text, not the model's own words."""
    import model_tools
    from agent.turn_finalizer import apply_llm_output_transform

    _wire(vr, monkeypatch)
    card = ("**Done · Sun 18 Oct**\n\n| Movement | Done |\n|---|---|\n| row | 3 sets |\n\n"
            "⚠️ Unresolved: Curl.\nall jobs ran · vabc1234")
    stage(vr, ID1, card)
    raw = json.dumps({"output": "[verbatim %s]\n%s\n[verbatim %s]" % (ID1, card, ID1), "exit_code": 0})

    ids = model_tools._CallIds(task_id="task", session_id="sess-1", tool_call_id="call-1", turn_id="turn-1")
    assert model_tools._apply_transform_tool_result_hook("terminal", {"command": "coach_engine.py card"}, raw, 5, ids) == raw

    agent = SimpleNamespace(session_id="sess-1", model="m", platform="telegram")
    logger = logging.getLogger("test")
    model_words = "Here is your summary — nice work!"
    got = apply_llm_output_transform(agent, model_words, turn_id="turn-1", platform="telegram", logger=logger)
    assert got == (card, True, model_words)

    # A turn with no scripted text on Telegram only gets the dash scrub.
    text, transformed, _ = apply_llm_output_transform(agent, "ok — done", turn_id="turn-2", platform="telegram", logger=logger)
    assert (text, transformed) == ("ok, done", True)


def test_end_to_end_two_tool_calls_in_one_turn(vr, monkeypatch):
    import model_tools
    from agent.turn_finalizer import apply_llm_output_transform

    _wire(vr, monkeypatch)
    stage(vr, ID1, "Renamed Row to cable seated row.")
    stage(vr, ID2, "Renamed Curl to barbell curl.")
    ids = model_tools._CallIds(session_id="sess-1", turn_id="turn-9")
    for vid in (ID1, ID2):
        reply = json.dumps({"output": json.dumps({"verbatim_id": vid})})
        model_tools._apply_transform_tool_result_hook("terminal", {}, reply, 1, ids)
    agent = SimpleNamespace(session_id="sess-1", model="m", platform="telegram")
    text, transformed, _ = apply_llm_output_transform(agent, "done", turn_id="turn-9", platform="telegram", logger=logging.getLogger("t"))
    assert text == "Renamed Row to cable seated row.\n\nRenamed Curl to barbell curl." and transformed
