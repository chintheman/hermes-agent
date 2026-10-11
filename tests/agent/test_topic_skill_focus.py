"""Invariants for per-topic ``skill_focus``: demote to names-only, never hide."""

import pytest

from agent.coding_context import (
    TOPIC_CORE_SKILL_CATEGORIES,
    topic_compact_skill_categories,
    topic_skill_focus,
)
from agent.prompt_builder import build_skills_system_prompt

TOPICS = [{"chat_id": "-1001", "topics": [
    {"name": "Ops", "thread_id": 7, "skill_focus": ["devops"]},
    {"name": "Plain", "thread_id": 8},
]}]


def test_focus_lookup_matches_chat_and_thread_only():
    assert topic_skill_focus(TOPICS, "-1001", "7") == ("devops",)
    assert topic_skill_focus(TOPICS, "-1001", "8") is None   # topic without skill_focus
    assert topic_skill_focus(TOPICS, "-100999", "7") is None  # other chat
    assert topic_skill_focus(TOPICS, "-1001", "") is None
    assert topic_skill_focus(TOPICS, "", "7") is None         # no chat id: never a thread-id-only match
    assert topic_skill_focus({"-1001": TOPICS[0]["topics"]}, "", "7") is None
    legacy = {"-1001": [{"thread_id": 7, "skill_focus": "health"}]}
    assert topic_skill_focus(legacy, "-1001", "7") == ("health",)


def test_compact_set_spares_focus_and_core_and_none_means_unchanged():
    present = {"devops", "health", "finance", *TOPIC_CORE_SKILL_CATEGORIES}
    assert topic_compact_skill_categories(present, ("devops",)) == {"health", "finance"}
    assert topic_compact_skill_categories(present, None) == frozenset()


def test_focus_never_hides_a_skill_and_shrinks_the_block(monkeypatch, tmp_path):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    for cat, name in (("devops", "deployer"), ("health", "coach"), ("meta", "helper")):
        d = tmp_path / "skills" / cat / name
        d.mkdir(parents=True)
        (d / "SKILL.md").write_text(f"---\nname: {name}\ndescription: Long {name} description text\n---\n")
    present = {"devops", "health", "meta"}
    full = build_skills_system_prompt()
    focused = build_skills_system_prompt(
        compact_categories=topic_compact_skill_categories(present, ("devops",)))
    for name in ("deployer", "coach", "helper"):
        assert name in focused                                   # never hidden
    assert "Long coach description text" not in focused          # demoted
    assert "Long deployer description text" in focused           # focus stays full
    assert "Long helper description text" in focused             # core stays full
    assert "Long coach description text" in full                 # control: full render has it


@pytest.mark.parametrize("shape", ["bare", "platforms"])
def test_topic_focus_read_from_every_telegram_config_shape(monkeypatch, tmp_path, shape):
    """Regression: a top-level ``telegram:`` section (the common user shape) was ignored."""
    import yaml
    from agent.system_prompt import _topic_compact_categories
    from gateway.session_context import clear_session_vars, set_session_vars
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    for cat in ("devops", "health", "meta"):
        (tmp_path / "skills" / cat).mkdir(parents=True)
    tg = {"extra": {"group_topics": TOPICS}}
    cfg = {"telegram": tg} if shape == "bare" else {"platforms": {"telegram": tg}}
    (tmp_path / "config.yaml").write_text(yaml.safe_dump(cfg))
    tokens = set_session_vars(platform="telegram", chat_id="-1001", thread_id="7")
    try:
        assert _topic_compact_categories(tmp_path / "skills") == {"health"}
    finally:
        clear_session_vars(tokens)
