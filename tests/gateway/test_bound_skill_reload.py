"""A bound skill reaches an ongoing chat when its file changes (Chin, 2026-10-22): topic 5's chat
started 20 Oct and never saw two days of coach rule changes, because bound skills loaded only on a
new session."""

from types import SimpleNamespace

import pytest

from gateway.run import GatewayRunner

SKILL = {"content": "rules v1"}


@pytest.fixture()
def runner(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setattr("agent.skill_commands._load_skill_payload",
                        lambda name, **kw: ({"name": name, "content": SKILL["content"]}, tmp_path / name, name))
    monkeypatch.setattr("agent.skill_commands._build_skill_message",
                        lambda loaded, d, note, **kw: note + "\n" + loaded["content"])
    SKILL["content"] = "rules v1"
    return GatewayRunner.__new__(GatewayRunner)


def turn(runner, new, sid="s1"):
    event = SimpleNamespace(text="summary for today")
    runner._hmwa_auto_load_skills(event, "coach", "q", "key", session_id=sid, only_changed=not new)
    return event.text


def test_a_new_session_loads_the_skill_and_an_unchanged_one_is_not_resent(runner):
    assert "rules v1" in turn(runner, new=True)
    assert turn(runner, new=False) == "summary for today"


def test_an_edited_skill_reaches_the_ongoing_chat_once(runner):
    turn(runner, new=True)
    SKILL["content"] = "rules v2"
    text = turn(runner, new=False)
    assert "was updated since this chat started" in text and "rules v2" in text
    assert text.endswith("summary for today")
    assert turn(runner, new=False) == "summary for today"


def test_a_chat_started_before_this_fix_reloads_once(runner):
    assert "rules v1" in turn(runner, new=False, sid="old")
    assert turn(runner, new=False, sid="old") == "summary for today"


def test_other_sessions_keep_their_own_versions(runner):
    turn(runner, new=True, sid="a")
    turn(runner, new=True, sid="b")
    SKILL["content"] = "rules v2"
    assert "rules v2" in turn(runner, new=False, sid="a")
    assert "rules v2" in turn(runner, new=False, sid="b")
