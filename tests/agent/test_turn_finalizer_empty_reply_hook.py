"""An empty model reply still lets a transform_llm_output plugin supply the text.

A scripted message (the verbatim-reply plugin) must reach the user even when the model says nothing
after its tool call. Only a string from the hook counts; the empty outcome is never recorded, so a
later call in the same turn with real text still fires the hook.
"""

import logging
from types import SimpleNamespace

from agent.turn_finalizer import apply_llm_output_transform, finalize_turn
from tests.agent.test_turn_finalizer_final_response_persistence import FakeAgent

LOG = logging.getLogger("test")


def _hooks(monkeypatch, answer, calls):
    def invoke_hook(name, **kwargs):
        calls.append((name, kwargs.get("response_text")))
        return [answer] if name == "transform_llm_output" and answer is not None else []

    monkeypatch.setattr("hermes_cli.lifecycle.invoke_hook", invoke_hook)


def _agent():
    return SimpleNamespace(session_id="s", model="m", platform="telegram")


def test_empty_reply_without_allow_empty_is_untouched_and_the_hook_is_not_fired(monkeypatch):
    calls = []
    _hooks(monkeypatch, "scripted", calls)
    assert apply_llm_output_transform(_agent(), "", turn_id="t", logger=LOG) == ("", False, None)
    assert calls == []


def test_empty_reply_with_allow_empty_takes_the_hook_string(monkeypatch):
    calls = []
    _hooks(monkeypatch, "scripted", calls)
    agent = _agent()
    assert apply_llm_output_transform(agent, "", turn_id="t", logger=LOG, allow_empty=True) == ("scripted", True, "")
    assert calls == [("transform_llm_output", "")]
    # idempotent per turn: the recorded outcome comes back, the hook is not fired again
    assert apply_llm_output_transform(agent, "scripted", turn_id="t", logger=LOG)[1] is True
    assert len(calls) == 1


def test_empty_reply_and_no_hook_string_is_not_recorded(monkeypatch):
    """Negative control: nothing recorded, so real text later in the same turn still reaches the hook."""
    calls = []
    _hooks(monkeypatch, None, calls)
    agent = _agent()
    assert apply_llm_output_transform(agent, "", turn_id="t", logger=LOG, allow_empty=True) == ("", False, None)
    assert not hasattr(agent, "_llm_output_transform")
    _hooks(monkeypatch, "rewritten", calls)
    assert apply_llm_output_transform(agent, "real text", turn_id="t", logger=LOG)[0] == "rewritten"


def test_empty_hook_string_is_ignored(monkeypatch):
    _hooks(monkeypatch, "", [])
    assert apply_llm_output_transform(_agent(), "", turn_id="t", logger=LOG, allow_empty=True) == ("", False, None)


def _finalize(final_response):
    agent = FakeAgent()
    agent.platform = "telegram"
    messages = [{"role": "user", "content": "do it"}]
    return finalize_turn(
        agent, final_response=final_response, api_call_count=2, interrupted=False, failed=False,
        messages=messages, conversation_history=[], effective_task_id="task", turn_id="turn",
        user_message="do it", original_user_message="do it", _should_review_memory=False,
        _turn_exit_reason="text_response(finish_reason=stop)",
    )


def test_finalize_turn_delivers_the_scripted_text_when_the_model_says_nothing(monkeypatch):
    calls = []
    _hooks(monkeypatch, "SCRIPTED CARD", calls)
    result = _finalize("")
    assert result["final_response"] == "SCRIPTED CARD"
    assert result["response_transformed"] is True
    # the hook fired once for the turn, not twice
    assert [c for c in calls if c[0] == "transform_llm_output"] == [("transform_llm_output", "")]


def test_finalize_turn_empty_reply_without_a_plugin_string_stays_empty(monkeypatch):
    calls = []
    _hooks(monkeypatch, None, calls)
    result = _finalize("")
    assert not result["final_response"]
    assert result["response_transformed"] is False


def test_finalize_turn_nonempty_reply_still_fires_the_hook_once(monkeypatch):
    calls = []
    _hooks(monkeypatch, "REWRITTEN", calls)
    result = _finalize("model words")
    assert result["final_response"] == "REWRITTEN"
    assert [c for c in calls if c[0] == "transform_llm_output"] == [("transform_llm_output", "model words")]
