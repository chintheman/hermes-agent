"""Tests for the turn-end promise check: a reply that ends by promising later work nobody will run."""

from __future__ import annotations

import pytest

from agent.promise_stop import build_promise_stop_nudge, find_deferred_promise, promise_check_enabled


@pytest.fixture(autouse=True)
def _clear_env(monkeypatch):
    monkeypatch.delenv("HERMES_PROMISE_CHECK", raising=False)


# Real replies that promised work after the turn ended (nothing runs until the next message).
FLAGGED = [
    # Telegram topic 5, 2026-10-21 22:56 (Chin: "Feels wrong")
    "Config says deepseek-flash, but this session's header says claude-haiku-5-5 on anthropic. "
    "I can't confirm which one is serving this chat from here. I'll check the live session record "
    "next and tell you what it shows.",
    "The pin should move at 07:30. I'll check it then and let you know.",
    "Restart is done. Starting now.",
    "Got it. Let me check the latest data.",
    "Logged. I'll keep an eye on it and report back.",
    "That looks off. I'll look into it shortly.",
    "I will verify the totals and get back to you.",
]

# Replies that must pass: the work is done, waits on his reply, or is a scheduled job.
PASSED = [
    "Logged 2 eggs: 140 kcal, 12 g protein. Day total 1,240 kcal.",
    "Send a test message in topic 5 and I'll confirm which model answered.",
    "When you reply with the weight, I'll update the day.",
    "Once you send the photo, I'll log it.",
    "Let me check: covered 2026-10-14. Yes, it was delivered at 08:00.",
    "I checked the log: the 22:55 reply came from claude-haiku-5-5.\n\nNothing else to do.",
    "The brief job sends tomorrow's plan at 07:30.",
    "",
]


@pytest.mark.parametrize("text", FLAGGED)
def test_flags_a_promise_of_later_work(text):
    assert find_deferred_promise(text)


@pytest.mark.parametrize("text", PASSED)
def test_passes_finished_or_conditional_replies(text):
    assert find_deferred_promise(text) is None


def test_only_the_ending_counts():
    early = "I'll check the record next. " + "Here is what it shows. " * 3 + "Sleep was 6 h 23 min, readiness 71."
    assert find_deferred_promise(early) is None


def test_nudge_quotes_the_promise_and_offers_both_ways_out():
    nudge = build_promise_stop_nudge(FLAGGED[0], attempts=0)
    assert "I'll check the live session record next" in nudge
    assert "now" in nudge and "when you reply" in nudge


def test_nudges_once_per_turn():
    assert build_promise_stop_nudge(FLAGGED[0], attempts=1) is None


def test_env_can_disable(monkeypatch):
    monkeypatch.setenv("HERMES_PROMISE_CHECK", "off")
    assert promise_check_enabled() is False
    assert build_promise_stop_nudge(FLAGGED[0], attempts=0) is None


def test_non_text_is_ignored():
    assert build_promise_stop_nudge(None, attempts=0) is None
    assert build_promise_stop_nudge(["I'll check next."], attempts=0) is None


class _Agent:
    """The attributes apply_stop_gates reads, nothing more."""

    def __init__(self):
        self._turn_file_mutation_paths = set()
        self.emitted = []

    def _interim_content_was_streamed(self, text):
        return False

    def _emit_interim_assistant_message(self, msg):
        self.emitted.append(msg)


def test_stop_gate_holds_the_draft_back_and_asks_once(monkeypatch):
    from agent.turn_stop_gates import apply_stop_gates

    monkeypatch.delenv("HERMES_KANBAN_TASK", raising=False)
    agent, messages = _Agent(), [{"role": "user", "content": "what model are you"}]
    draft = {"role": "assistant", "content": FLAGGED[0]}
    verdict = apply_stop_gates(agent, draft, final_response=FLAGGED[0], messages=messages, conversation_history=[],
                               pending_verification_response=None, pending_verification_response_previewed=False)
    assert verdict.continue_turn is True
    assert agent.emitted == []                       # the promise never reaches the chat
    assert draft["_promise_stop_synthetic"] is True  # nor the stored transcript
    assert messages[-1]["role"] == "user" and messages[-1]["_promise_stop_synthetic"] is True

    redo = "I checked the gateway log: the 22:55 reply came from claude-haiku-5-5."
    verdict = apply_stop_gates(agent, {"role": "assistant", "content": FLAGGED[0]}, final_response=FLAGGED[0],
                               messages=messages, conversation_history=[], pending_verification_response=None,
                               pending_verification_response_previewed=False)
    assert verdict.continue_turn is False             # a second promise in the same turn goes out as is
    verdict = apply_stop_gates(_Agent(), {"role": "assistant", "content": redo}, final_response=redo, messages=[],
                               conversation_history=[], pending_verification_response=None,
                               pending_verification_response_previewed=False)
    assert verdict.continue_turn is False and verdict.final_response == redo


def test_synthetic_rows_are_never_persisted():
    from agent.session_persistence import _is_ephemeral_scaffolding

    assert _is_ephemeral_scaffolding({"role": "assistant", "content": "x", "_promise_stop_synthetic": True})
