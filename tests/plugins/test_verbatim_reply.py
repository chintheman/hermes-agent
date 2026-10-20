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
ID3 = "20261019T104418Zfeedface"


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
    monkeypatch.setenv("HWC_VERBATIM_DIR", str(tmp_path / "verbatim"))
    (tmp_path / "verbatim").mkdir()
    mod = _load()
    mod._PENDING.clear()
    mod._USER_BY_TURN.clear()
    mod._USER_BY_SESSION.clear()
    mod._DELIVERED.clear()
    return mod


def _llm(vr, **kw):
    """The two registered callbacks in call order, first string wins (what the hook loop does)."""
    return vr._on_transform_llm_output_first(**kw) or vr._on_transform_llm_output(**kw)


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
        out = f"[verbatim {ID1}]\n**Done**\nline\n[verbatim {ID1}]"
        self.run(vr, json.dumps({"output": out}))
        assert vr._PENDING[("s1", "t1")]["texts"] == ["**Done**\nline"]

    def test_marker_survives_head_or_tail(self, vr):
        stage(vr, ID1, "body")
        self.run(vr, f"[verbatim {ID1}]\nbody")
        self.run(vr, f"body\n[verbatim {ID1}]", turn="t2")
        assert vr._PENDING[("s1", "t1")]["texts"] == vr._PENDING[("s1", "t2")]["texts"] == ["body"]

    def test_order_is_kept_and_ids_are_deduped_across_calls(self, vr):
        stage(vr, ID1, "first")
        stage(vr, ID2, "second")
        self.run(vr, f"[verbatim {ID1}]")
        self.run(vr, f"[verbatim {ID2}] [verbatim {ID1}]")
        assert vr._PENDING[("s1", "t1")]["texts"] == ["first", "second"]

    def test_keyed_by_session_and_turn(self, vr):
        stage(vr, ID1, "a")
        self.run(vr, f"[verbatim {ID1}]", session="s1", turn="t1")
        assert ("s1", "t1") in vr._PENDING
        assert ("s2", "t1") not in vr._PENDING and ("s1", "t2") not in vr._PENDING

    @pytest.mark.parametrize("vid", ["../../etc/passwd", "short", "a" * 65, "has space in it", "ok_id_01/../x"])
    def test_bad_ids_are_ignored(self, vr, vid):
        self.run(vr, json.dumps({"verbatim_id": vid}))
        self.run(vr, f"[verbatim {vid}]")
        assert not vr._PENDING

    def test_symlink_escape_is_refused(self, vr, tmp_path):
        secret = tmp_path / "outside.md"
        secret.write_text("SECRET")
        os.symlink(secret, vr.verbatim_dir() / (ID1 + ".md"))
        self.run(vr, f"[verbatim {ID1}]")
        assert not vr._PENDING
        assert vr.read_staged(ID1) is None

    def test_missing_empty_and_stale_files_are_ignored(self, vr):
        stage(vr, ID2, "  \n")
        old = stage(vr, "20250101T000000Zaaaaaaaa", "old")
        past = time.time() - 7200
        os.utime(old, (past, past))
        for vid in (ID1, ID2, "20250101T000000Zaaaaaaaa"):
            self.run(vr, f"[verbatim {vid}]")
        assert not vr._PENDING

    def test_fails_open_on_anything_odd(self, vr, monkeypatch):
        assert self.run(vr, None) is None
        assert self.run(vr, 12345) is None

        def boom(vid):
            raise RuntimeError("boom")

        monkeypatch.setattr(vr, "read_staged", boom)
        assert self.run(vr, f"[verbatim {ID1}]") is None

    def test_default_dir_follows_hermes_home(self, tmp_path, monkeypatch):
        monkeypatch.delenv("HWC_VERBATIM_DIR", raising=False)
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        assert _load().verbatim_dir() == tmp_path / "data" / "op" / "verbatim"


# ---------------------------------------------------------------------------
# transform_llm_output
# ---------------------------------------------------------------------------

class TestLlmOutput:
    def out(self, vr, text="", platform="telegram", session="s1", turn="t1"):
        return _llm(vr, response_text=text, session_id=session, model="m", platform=platform, turn_id=turn)

    def test_scripted_text_replaces_the_model_text_joined_by_a_blank_line(self, vr, caplog):
        vr._PENDING[("s1", "t1")] = {"ts": time.time(), "ids": [ID1, ID2], "texts": ["one", "two"]}
        with caplog.at_level(logging.INFO):
            assert self.out(vr) == "one\n\ntwo"
        assert "verbatim-reply: reply is no model text and 2 scripted message(s)" in caplog.text
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

    def register_hook(self, name, cb, first=False):
        lst = self.hooks.setdefault(name, [])
        lst.insert(0, cb) if first else lst.append(cb)


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
    raw = json.dumps({"output": f"[verbatim {ID1}]\n{card}\n[verbatim {ID1}]", "exit_code": 0})

    ids = model_tools._CallIds(task_id="task", session_id="sess-1", tool_call_id="call-1", turn_id="turn-1")
    assert model_tools._apply_transform_tool_result_hook("terminal", {"command": "coach_engine.py card"}, raw, 5, ids) == raw

    agent = SimpleNamespace(session_id="sess-1", model="m", platform="telegram")
    logger = logging.getLogger("test")
    model_words = "Here is your summary — nice work!"
    got = apply_llm_output_transform(agent, model_words, turn_id="turn-1", platform="telegram", logger=logger)
    assert got == ("Here is your summary, nice work!\n\n" + card, True, model_words)

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
    assert text == "done\n\nRenamed Row to cable seated row.\n\nRenamed Curl to barbell curl." and transformed


# ---------------------------------------------------------------------------
# Round 3 review fixes
# ---------------------------------------------------------------------------

class TestReviewFixes:
    def tool(self, vr, result, name="terminal", session="s1", turn="t1"):
        return vr._on_transform_tool_result(tool_name=name, args={}, result=result, session_id=session, turn_id=turn)

    def out(self, vr, text="", platform="telegram", session="s1", turn="t1"):
        return _llm(vr, response_text=text, session_id=session, model="m", platform=platform, turn_id=turn)

    # 1. dedupe by text
    def test_same_text_under_two_ids_is_sent_once(self, vr):
        stage(vr, ID1, "the card\n")
        stage(vr, ID2, "the card")
        self.tool(vr, f"[verbatim {ID1}]")
        self.tool(vr, f"[verbatim {ID2}]")
        assert vr._PENDING[("s1", "t1")]["texts"] == ["the card"]
        assert self.out(vr) == "the card"

    def test_a_question_the_model_answered_later_in_the_turn_is_not_sent(self, vr):
        q = "\u2754 **I need numbers for this dish**\nSend kcal and protein ranges."
        stage(vr, ID1, q)
        stage(vr, ID2, q.replace("dish", "drink"))
        stage(vr, ID3, "the food card")
        qres = lambda vid: json.dumps({"output": json.dumps({"status": "needs_question", "verbatim_id": vid})})
        self.tool(vr, qres(ID1))
        self.tool(vr, qres(ID2))
        self.tool(vr, json.dumps({"output": json.dumps({"status": "ok", "verbatim_id": ID3})}))
        assert self.out(vr) == "the food card"

    def test_a_question_that_is_the_last_scripted_text_is_sent(self, vr):
        stage(vr, ID1, "the food card")
        stage(vr, ID2, "Which dish is this?")
        self.tool(vr, json.dumps({"status": "ok", "verbatim_id": ID1}))
        self.tool(vr, json.dumps({"status": "needs_question", "verbatim_id": ID2}))
        assert self.out(vr) == "the food card\n\nWhich dish is this?"

    # 20 Oct: the model's own words are kept (Chin asked a question and never saw the answer)
    def test_the_model_answer_comes_first_and_the_card_is_attached(self, vr):
        stage(vr, ID1, "\u26aa **Paris Baguette iced latte**\nkcal 165 to 205\nall jobs ran \u00b7 v1")
        self.tool(vr, json.dumps({"status": "ok", "verbatim_id": ID1}))
        got = self.out(vr, text="Found it on the Paris Baguette site \u2014 their latte, so I updated it.")
        assert got == ("Found it on the Paris Baguette site, their latte, so I updated it.\n\n"
                       "\u26aa **Paris Baguette iced latte**\nkcal 165 to 205\nall jobs ran \u00b7 v1")

    def test_a_card_the_model_already_pasted_is_not_sent_twice(self, vr):
        stage(vr, ID1, "**Card**\nline two")
        self.tool(vr, json.dumps({"status": "ok", "verbatim_id": ID1}))
        assert self.out(vr, text="Logged.\n\n**Card**\nline two") == "Logged.\n\n**Card**\nline two"

    def test_a_tool_question_is_not_attached_when_the_model_wrote_its_own(self, vr):
        stage(vr, ID1, "\u2754 **I need numbers for this dish**\nSend kcal and protein ranges.")
        self.tool(vr, json.dumps({"status": "needs_question", "verbatim_id": ID1}))
        assert self.out(vr, text="Was that the large or the regular latte?") == "Was that the large or the regular latte?"

    def test_different_texts_are_both_kept(self, vr):
        stage(vr, ID1, "one")
        stage(vr, ID2, "two")
        self.tool(vr, f"[verbatim {ID1}]")
        self.tool(vr, f"[verbatim {ID2}]")
        assert self.out(vr) == "one\n\ntwo"

    # 2. only terminal, process and execute_code tools; never twice
    @pytest.mark.parametrize("name", ["read_file", "session_search", "web_extract", "web_search", "search_files", ""])
    def test_ids_from_other_tools_are_ignored(self, vr, name):
        stage(vr, ID1, "old card")
        self.tool(vr, f"[verbatim {ID1}]", name=name)
        assert not vr._PENDING

    @pytest.mark.parametrize("name", ["terminal", "process", "process_manage", "execute_code"])
    def test_ids_from_the_script_running_tools_are_taken(self, vr, name):
        stage(vr, ID1, "fresh card")
        self.tool(vr, f"[verbatim {ID1}]", name=name)
        assert vr._PENDING[("s1", "t1")]["texts"] == ["fresh card"]

    def test_a_delivered_id_is_never_delivered_twice(self, vr):
        stage(vr, ID1, "card")
        self.tool(vr, f"[verbatim {ID1}]")
        assert self.out(vr) == "card"
        self.tool(vr, f"[verbatim {ID1}]", turn="t2")      # an old id echoed in a later turn
        assert not vr._PENDING
        assert self.out(vr, turn="t2") is None

    def test_the_delivered_set_is_bounded(self, vr):
        for i in range(vr.DELIVERED_MAX + 30):
            vr._remember_delivered([f"id{i:08d}"], time.time())
        assert len(vr._DELIVERED) == vr.DELIVERED_MAX
        assert "id00000000" not in vr._DELIVERED

    # 3. lookup fallback and warnings
    def test_lookup_falls_back_to_the_single_entry_with_that_turn_id(self, vr, caplog):
        stage(vr, ID1, "card")
        self.tool(vr, f"[verbatim {ID1}]", session="old-session", turn="t9")
        with caplog.at_level(logging.INFO):
            assert self.out(vr, session="compressed-session", turn="t9") == "card"
        assert "session id changed mid-turn" in caplog.text

    def test_no_fallback_when_two_entries_share_the_turn_id(self, vr):
        vr._PENDING[("a", "t9")] = {"ts": time.time(), "ids": [ID1], "texts": ["a"]}
        vr._PENDING[("b", "t9")] = {"ts": time.time(), "ids": [ID2], "texts": ["b"]}
        assert self.out(vr, session="c", turn="t9") is None

    def test_a_miss_while_entries_exist_warns(self, vr, caplog):
        vr._PENDING[("a", "t1")] = {"ts": time.time(), "ids": [ID1], "texts": ["a"]}
        with caplog.at_level(logging.WARNING):
            assert self.out(vr, session="b", turn="t2") is None
        assert "no scripted entry" in caplog.text

    def test_no_warning_when_nothing_waits(self, vr, caplog):
        with caplog.at_level(logging.WARNING):
            self.out(vr)
        assert "verbatim-reply" not in caplog.text

    def test_an_entry_that_expires_undelivered_warns(self, vr, caplog):
        vr._PENDING[("a", "t1")] = {"ts": time.time() - 3600, "ids": [ID1], "texts": ["a"]}
        with caplog.at_level(logging.WARNING):
            self.out(vr, session="b", turn="t2")
        assert "expired undelivered" in caplog.text

    # 4. marker lines are stripped from every final text
    def test_marker_lines_are_stripped_even_without_scripted_text(self, vr):
        text = f"[verbatim {ID1}]\nhello\n[verbatim {ID1}]"
        assert self.out(vr, text, platform="cli") == "hello\n"

    def test_inline_marker_and_telegram_scrub_together(self, vr):
        assert self.out(vr, f"see [verbatim {ID1}] this — now") == "see this, now"

    def test_a_non_marker_bracket_is_kept(self, vr):
        assert self.out(vr, "[verbatim short] and [verbatim] ok", platform="cli") is None

    def test_scripted_text_has_markers_stripped_too(self, vr):
        vr._PENDING[("s1", "t1")] = {"ts": time.time(), "ids": [ID1], "texts": [f"a [verbatim {ID1}]"]}
        assert self.out(vr) == "a"

    # 5. add-note guard
    def pre(self, vr, cmd, session="s1", turn="t1"):
        return vr._on_pre_tool_call(tool_name="terminal", args={"command": cmd}, session_id=session, turn_id=turn)

    def say(self, vr, msg, session="s1", turn="t1"):
        vr._on_pre_llm_call(session_id=session, turn_id=turn, user_message=msg)

    def test_grep_for_the_verb_is_not_a_call(self, vr):
        assert self.pre(vr, "grep add-note coach_log.py") is None
        assert self.pre(vr, "grep -n add-note ~/x/scripts/coach_log.py") is None
        assert self.pre(vr, "cat coach_log.py | grep add-note") is None
        assert self.pre(vr, "sed -n 1,20p coach_log.py # add-note") is None

    def test_a_python_call_is_still_checked(self, vr):
        self.say(vr, "knee felt fine")
        assert self.pre(vr, "python3 coach_log.py add-note --text 'invented words'") == NOTE_BLOCK
        assert self.pre(vr, "python coach_log.py add-note --text 'invented words'") == NOTE_BLOCK
        assert self.pre(vr, "python3 -u ~/s/coach_log.py add-note --date D --text 'invented words'") == NOTE_BLOCK
        assert self.pre(vr, "python3 coach_log.py add-note --text 'knee felt fine'") is None

    def test_word_boundaries(self, vr):
        self.say(vr, "knee felt fine")
        assert self.pre(vr, "python3 coach_log.py add-notes --text 'invented words'") is None   # a different token
        assert self.pre(vr, "python3 my_coach_log.py_old add-note --text 'x y'") is None

    def test_one_word_text_is_blocked(self, vr):
        self.say(vr, "knee")
        assert self.pre(vr, "python3 coach_log.py add-note --text knee") == NOTE_BLOCK
        self.say(vr, "knee felt fine")
        assert self.pre(vr, "python3 coach_log.py add-note --text 'knee felt'") is None

    def test_latest_user_message_is_keyed_by_turn_with_session_fallback(self, vr):
        self.say(vr, "turn one words here", turn="t1")
        self.say(vr, "turn two words here", turn="t2")
        assert self.pre(vr, "python3 coach_log.py add-note --text 'turn two words'", turn="t2") is None
        assert self.pre(vr, "python3 coach_log.py add-note --text 'turn one words'", turn="t2") == NOTE_BLOCK
        assert self.pre(vr, "python3 coach_log.py add-note --text 'turn one words'", turn="t1") is None
        # unknown turn: the session's latest message
        assert self.pre(vr, "python3 coach_log.py add-note --text 'turn two words'", turn="t3") is None
        # session id changed but the turn is known
        assert self.pre(vr, "python3 coach_log.py add-note --text 'turn one words'", session="other", turn="t1") is None

    def test_unparseable_python_call_blocks_but_a_grep_with_bad_quotes_does_not(self, vr):
        self.say(vr, "knee felt fine")
        assert self.pre(vr, "python3 coach_log.py add-note --text 'unterminated") == NOTE_BLOCK
        assert self.pre(vr, "grep 'add-note coach_log.py") is None

    # 6. dash scrub
    @pytest.mark.parametrize("src", [
        "see https://example.com/a–b now",
        "link http://x.y/z—w here",
        "open t.me/c/123–456 please",
        "ftp://host/a—b and more",
    ])
    def test_urls_are_untouched(self, vr, src):
        assert vr.scrub_dashes(src) == src

    def test_text_around_a_url_is_still_scrubbed(self, vr):
        assert vr.scrub_dashes("a — https://x.io/a–b — b") == "a, https://x.io/a–b, b"

    def test_table_cell_that_is_only_a_dash_becomes_a_hyphen(self, vr):
        src = "| Move | Done |\n|---|---|\n| row | — |\n| curl | – |"
        assert vr.scrub_dashes(src) == "| Move | Done |\n|---|---|\n| row | - |\n| curl | - |"

    @pytest.mark.parametrize("src,want", [
        ("2026-10-05 – 2026-10-07", "2026-10-05 to 2026-10-07"),
        ("2026-10-05—2026-10-07", "2026-10-05 to 2026-10-07"),
        ("Oct 5 – Oct 7", "Oct 5 to Oct 7"),
        ("Oct 5–Oct 7", "Oct 5 to Oct 7"),
        ("5 Oct – 7 Oct", "5 Oct to 7 Oct"),
        ("Oct 5 – 7", "Oct 5 to 7"),
        ("away Oct 24—28 now", "away Oct 24 to 28 now"),
        ("weeks 3–4", "weeks 3-4"),
        ("a pause — here", "a pause, here"),
    ])
    def test_dates(self, vr, src, want):
        assert vr.scrub_dashes(src) == want

    # 7. one env var name and the repo helper
    def test_only_hwc_verbatim_dir_is_read(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HERMES_VERBATIM_DIR", str(tmp_path / "old-name"))
        monkeypatch.delenv("HWC_VERBATIM_DIR", raising=False)
        monkeypatch.setenv("HERMES_HOME", str(tmp_path / "h"))
        assert _load().verbatim_dir() == tmp_path / "h" / "data" / "op" / "verbatim"
        monkeypatch.setenv("HWC_VERBATIM_DIR", str(tmp_path / "new-name"))
        assert _load().verbatim_dir() == tmp_path / "new-name"

    def test_the_source_has_no_hard_coded_home_and_no_old_env_name(self):
        src = (REPO / "plugins" / "verbatim-reply" / "__init__.py").read_text()
        assert "HERMES_VERBATIM_DIR" not in src
        assert "get_hermes_home" in src

    # 8. the skill lock limit is documented
    def test_docstring_documents_the_terminal_limit(self):
        doc = _load().__doc__
        assert "does NOT inspect `terminal` or `execute_code`" in doc


# ---------------------------------------------------------------------------
# Contract with the coach repo (verbatim.py)
# ---------------------------------------------------------------------------

COACH_SCRIPTS = Path(os.environ.get("HWC_COACH_SCRIPTS") or "/Users/0xsteamboat/projects/health-wellness-coach/skill/scripts")


@pytest.mark.skipif(not (COACH_SCRIPTS / "verbatim.py").exists(), reason="coach repo not on this machine")
def test_id_patterns_are_identical_in_both_repos():
    src = (COACH_SCRIPTS / "verbatim.py").read_text()
    coach_id = src.split('ID_RE = re.compile(r"^')[1].split('$")')[0]
    assert coach_id == _load().ID_PATTERN


@pytest.mark.skipif(not (COACH_SCRIPTS / "verbatim.py").exists(), reason="coach repo not on this machine")
def test_real_coach_staging_round_trips_through_the_plugin(vr, tmp_path, monkeypatch):
    """Real verbatim.stage output (JSON id form and marker form) is found and read back by the plugin."""
    spec = importlib.util.spec_from_file_location("coach_verbatim_under_test", COACH_SCRIPTS / "verbatim.py")
    cv = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cv)
    for _ in range(20):
        vid = cv.stage("**Done**\nline")
        assert vid and vr.ID_RE.match(vid)
        as_json = json.dumps({"output": json.dumps({"ok": True, "message": "m", "verbatim_id": vid})})
        as_marker = cv.wrap_text("**Done**\nline")
        for res in (as_json, json.dumps({"output": as_marker})):
            ids = vr.find_ids(res)
            assert len(ids) == 1 and vr.read_staged(ids[0]) in ("**Done**\nline",)
    # the marker form: first and last lines carry the same id
    ids = vr.find_ids(cv.wrap_text("x y"))
    assert len(ids) == 1


# ---------------------------------------------------------------------------
# Round 4: the scripted text wins whatever the load order
# ---------------------------------------------------------------------------

def _manager_with(vr, tmp_path, monkeypatch, rivals_before, rivals_after):
    """A real PluginManager whose transform_llm_output callbacks were registered in the given order."""
    from hermes_cli.plugins import PluginContext, PluginManager
    from hermes_cli.plugins_manifest import PluginManifest

    mgr = PluginManager()

    def ctx(name):
        return PluginContext(PluginManifest(name=name, source="user"), mgr)

    for name, fn in rivals_before:
        ctx(name).register_hook("transform_llm_output", fn)
    vr.register(ctx("verbatim-reply"))
    for name, fn in rivals_after:
        ctx(name).register_hook("transform_llm_output", fn)
    return mgr


def rewrite(**kw):
    return "REWRITTEN BY RIVAL: " + kw["response_text"]


def footer(**kw):
    return kw["response_text"] + "\n(footer)"


@pytest.mark.parametrize("where", ["before", "after", "both"])
def test_scripted_text_beats_rewriting_plugins_in_any_load_order(vr, tmp_path, monkeypatch, where):
    rivals = [("plain-words-rewrite", rewrite), ("output-shape-guard", rewrite), ("groq-meter", footer)]
    before = rivals if where in ("before", "both") else []
    after = rivals if where in ("after", "both") else []
    mgr = _manager_with(vr, tmp_path, monkeypatch, before, after)
    vr._PENDING[("s1", "t1")] = {"ts": time.time(), "ids": [ID1], "texts": ["SCRIPTED CARD"]}
    results = mgr.invoke_hook("transform_llm_output", response_text="model words", session_id="s1",
                              model="m", platform="telegram", turn_id="t1")
    assert next(r for r in results if isinstance(r, str)) == "model words\n\nSCRIPTED CARD"


def test_negative_control_without_first_a_rival_loaded_earlier_wins(vr, tmp_path, monkeypatch):
    """Shows the problem is real: registered in plain order after a rival, the scripted text loses."""
    from hermes_cli.plugins import PluginContext, PluginManager
    from hermes_cli.plugins_manifest import PluginManifest

    mgr = PluginManager()
    PluginContext(PluginManifest(name="plain-words-rewrite", source="user"), mgr).register_hook(
        "transform_llm_output", rewrite)
    PluginContext(PluginManifest(name="verbatim-reply", source="user"), mgr).register_hook(
        "transform_llm_output", vr._on_transform_llm_output_first)       # no first=True
    vr._PENDING[("s1", "t1")] = {"ts": time.time(), "ids": [ID1], "texts": ["SCRIPTED CARD"]}
    results = mgr.invoke_hook("transform_llm_output", response_text="model words", session_id="s1",
                              model="m", platform="telegram", turn_id="t1")
    assert next(r for r in results if isinstance(r, str)).startswith("REWRITTEN BY RIVAL")


def test_without_scripted_text_the_rival_still_wins(vr, tmp_path, monkeypatch):
    mgr = _manager_with(vr, tmp_path, monkeypatch, [("plain-words-rewrite", rewrite)], [])
    results = mgr.invoke_hook("transform_llm_output", response_text="model words", session_id="s1",
                              model="m", platform="telegram", turn_id="t1")
    assert next(r for r in results if isinstance(r, str)) == "REWRITTEN BY RIVAL: model words"


def test_markers_are_stripped_even_when_a_rival_would_win(vr, tmp_path, monkeypatch):
    mgr = _manager_with(vr, tmp_path, monkeypatch, [("plain-words-rewrite", rewrite)], [("groq-meter", footer)])
    text = f"hello [verbatim {ID1}]"
    results = mgr.invoke_hook("transform_llm_output", response_text=text, session_id="s1",
                              model="m", platform="cli", turn_id="t1")
    winner = next(r for r in results if isinstance(r, str))
    assert "[verbatim" not in winner and winner == "hello"


def test_a_second_scripted_turn_is_also_delivered_and_the_first_is_not_replayed(vr, tmp_path, monkeypatch):
    mgr = _manager_with(vr, tmp_path, monkeypatch, [("plain-words-rewrite", rewrite)], [])
    call = lambda turn: next((r for r in mgr.invoke_hook(
        "transform_llm_output", response_text="x", session_id="s1", model="m", platform="cli", turn_id=turn)
        if isinstance(r, str)), None)
    vr._PENDING[("s1", "t1")] = {"ts": time.time(), "ids": [ID1], "texts": ["ONE"]}
    assert call("t1") == "x\n\nONE"
    assert call("t1").startswith("REWRITTEN")
    vr._PENDING[("s1", "t2")] = {"ts": time.time(), "ids": [ID2], "texts": ["TWO"]}
    assert call("t2") == "x\n\nTWO"


def test_first_registration_is_unloaded_cleanly(vr, tmp_path, monkeypatch):
    mgr = _manager_with(vr, tmp_path, monkeypatch, [("plain-words-rewrite", rewrite)], [])
    cbs = mgr._hooks["transform_llm_output"]
    assert cbs[0] == vr._on_transform_llm_output_first
    assert cbs.index(vr._on_transform_llm_output) > 0


def test_register_falls_back_when_the_host_has_no_first_argument(vr):
    class OldCtx:
        def __init__(self):
            self.calls = []

        def register_hook(self, name, cb):
            self.calls.append((name, cb))

    ctx = OldCtx()
    vr.register(ctx)
    names = [n for n, _ in ctx.calls]
    assert names.count("transform_llm_output") == 2
