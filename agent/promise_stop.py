"""Turn-end check for promises of later work. Nothing runs after a reply until the next message
arrives, so a reply that ends with "I'll check X next and tell you" or "Starting now" promises work
no one will do (Chin, 2026-10-21: Haiku in topic 5 ended on exactly that). Policy-only: return one
synthetic nudge so the model either does the work now with its tools or says it waits on the
user's reply. The held-back reply is never shown or stored.
"""

from __future__ import annotations

import os
import re
from typing import Any, Optional

_DEFAULT_MAX_ATTEMPTS = 1
_TAIL_SENTENCES = 2

_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+|\n+")
# First person, future or "let me", then an act of checking or reporting back.
_PROMISE = re.compile(
    r"\b(?:i'?ll|i will|i'm going to|i am going to|let me)\s+(?:\w+\s+){0,2}?"
    r"(?:check|look|verify|confirm|report|follow up|get back|tell you|let you know|update you|ping you|"
    r"monitor|keep an eye|dig|investigate|test|circle back)\b",
    re.IGNORECASE,
)
# Said anywhere in the ending, these always mean "after this reply".
_ALWAYS = re.compile(r"\b(?:starting now|report back|get back to you|stand by|bear with me)\b", re.IGNORECASE)
# Work that waits on the user is honest: "when you reply, I'll ...", "send it and I'll confirm".
_WAITS_ON_USER = re.compile(
    r"\b(?:when|once|after|if|as soon as) you\b|\b(?:send|reply|paste|run|restart|tell me)\b[^.!?]*\band i'?ll\b",
    re.IGNORECASE,
)
# "Let me check: <result>" carries its own answer.
_ANSWERED = re.compile(r"\blet me \w+\s*:\s*\S", re.IGNORECASE)


def promise_check_enabled() -> bool:
    """On unless ``HERMES_PROMISE_CHECK`` is set to a false value."""
    return (os.environ.get("HERMES_PROMISE_CHECK") or "").strip().lower() not in {"0", "false", "no", "off"}


def find_deferred_promise(text: Any) -> Optional[str]:
    """The sentence in the reply's ending that promises later work, or None."""
    if not isinstance(text, str) or not text.strip():
        return None
    sentences = [s.strip() for s in _SENTENCE_SPLIT.split(text.strip()) if s.strip()]
    for sentence in sentences[-_TAIL_SENTENCES:]:
        if _WAITS_ON_USER.search(sentence) or _ANSWERED.search(sentence):
            continue
        if _PROMISE.search(sentence) or _ALWAYS.search(sentence):
            return sentence
    return None


def build_promise_stop_nudge(final_response: Any, *, attempts: int = 0,
                             max_attempts: int = _DEFAULT_MAX_ATTEMPTS) -> Optional[str]:
    """A one-time continuation message when the reply ends on a promise of later work."""
    if attempts >= max_attempts or not promise_check_enabled():
        return None
    promise = find_deferred_promise(final_response)
    if promise is None:
        return None
    return (
        "[Promise check] Your reply ended with: \"%s\" Nothing runs after this reply until the user "
        "sends another message, so that would not happen. Do the work now with your tools and give the "
        "result, or, if it truly needs the user, say it happens when you reply (or name the scheduled job "
        "that does it). Then write the whole reply again; the earlier draft was not sent." % promise
    )
