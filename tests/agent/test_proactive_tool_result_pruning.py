"""Tests for proactive tool-result pruning.

``ContextCompressor.prune_tool_results_only`` runs the cheap, deterministic
Phase-1 prune (summarize old tool outputs, dedup repeats) on a cost-oriented
trigger that is INDEPENDENT of the full-compression threshold. On large-window
models ``should_compress()`` (~50% of the window) rarely fires, so without this
the old tool outputs ride in history and are re-sent verbatim every turn.

Mirrors the construction/patching conventions in test_context_compressor.py.
"""

from unittest.mock import patch

from agent.context_compressor import (
    ContextCompressor,
    _PRUNED_TOOL_PLACEHOLDER,
    _estimate_msg_budget_tokens,
    _LEAN_TAIL_DEMOTE_MIN_CHARS,
    _PRUNE_MIN_CHARS,
    _lean_recovery_stub,
    salvage_grown_transcript,
)

LARGE_WINDOW = 1_000_000


def _compressor(**kw):
    defaults = dict(
        model="test",
        quiet_mode=True,
        threshold_percent=0.50,
        protect_first_n=2,
        protect_last_n=4,
    )
    defaults.update(kw)
    with patch(
        "agent.context_compressor.get_model_context_length",
        return_value=LARGE_WINDOW,
    ):
        return ContextCompressor(**defaults)


def _assistant_call(cid, name="terminal", args='{"cmd":"ls"}'):
    return {
        "role": "assistant",
        "content": "",
        "tool_calls": [
            {"id": cid, "type": "function",
             "function": {"name": name, "arguments": args}}
        ],
    }


def _tool_msg(cid, content):
    return {"role": "tool", "tool_call_id": cid, "content": content}


def _build(n_pairs, big_indices, big_chars=9000, small="ok"):
    """system + n_pairs of (assistant tool_call, tool result).

    Tool results whose pair index is in ``big_indices`` get a distinct payload
    of ``big_chars`` characters; the rest get a tiny payload.
    """
    msgs = [{"role": "system", "content": "sys"}]
    for i in range(n_pairs):
        cid = f"call_{i}"
        msgs.append(_assistant_call(cid))
        if i in big_indices:
            msgs.append(_tool_msg(cid, chr(65 + (i % 26)) * big_chars))
        else:
            msgs.append(_tool_msg(cid, small))
    return msgs


def _tool_by_id(msgs, cid):
    return [m for m in msgs if m.get("role") == "tool" and m.get("tool_call_id") == cid][0]


def test_prunes_below_compression_threshold():
    """The whole point: prune fires at 120k tokens, far below the ~500k
    (50% of 1M) full-compression trigger that would otherwise never run."""
    c = _compressor(proactive_prune_tokens=48_000, proactive_prune_min_result_chars=8_000)
    assert c.should_compress(prompt_tokens=120_000) is False  # compression would NOT run
    msgs = _build(8, big_indices={0, 1, 2})
    result, pruned = c.prune_tool_results_only(msgs, current_tokens=120_000)
    assert pruned >= 3
    assert len(result) == len(msgs)
    for cid in ("call_0", "call_1", "call_2"):
        m = _tool_by_id(result, cid)
        assert len(m["content"]) < 9000                       # summarized
        assert m["content"] != _PRUNED_TOOL_PLACEHOLDER       # informative, not a blank placeholder












def test_idempotent():
    c = _compressor(proactive_prune_tokens=48_000, proactive_prune_min_result_chars=8_000)
    msgs = _build(8, big_indices={0, 1, 2})
    first, n1 = c.prune_tool_results_only(msgs, current_tokens=120_000)
    assert n1 >= 3
    # No usage reading bypasses the token gate and exercises prune idempotence.
    second, n2 = c.prune_tool_results_only(first, current_tokens=None)
    assert n2 == 0
    assert [m.get("content") for m in second] == [m.get("content") for m in first]


def test_rearms_only_after_reclaimed_token_runway():
    """A prune boundary must earn back its cache break before the next one."""
    c = _compressor(
        proactive_prune_tokens=48_000,
        proactive_prune_min_result_chars=8_000,
    )
    msgs = _build(8, big_indices={0, 1, 2, 6, 7})

    first, n1 = c.prune_tool_results_only(msgs, current_tokens=120_000)
    assert n1 >= 3
    rearm_tokens = sum(map(_estimate_msg_budget_tokens, first)) + 48_000

    # Age the two protected large results out of the tail.  They are now a
    # valid >=4K-token prune candidate, but the post-prune prompt has not yet
    # regrown the tokens reclaimed at the first cache-breaking boundary.
    grown = first + [
        _assistant_call("call_8"),
        _tool_msg("call_8", "ok"),
        _assistant_call("call_9"),
        _tool_msg("call_9", "ok"),
    ]
    assert sum(map(_estimate_msg_budget_tokens, grown)) < rearm_tokens
    # Below the full-compression threshold, where the runway is pure
    # prompt-cache hysteresis. (Above it the runway is bypassed on the
    # provider-billed reading instead — see
    # tests/agent/test_proactive_prune_rearm_threshold.py, #101889.)
    _under_threshold = c.threshold_tokens - 1
    blocked, n2 = c.prune_tool_results_only(grown, current_tokens=_under_threshold)
    assert n2 == 0
    assert blocked is grown
    assert len(_tool_by_id(blocked, "call_6")["content"]) == 9000
    assert len(_tool_by_id(blocked, "call_7")["content"]) == 9000

    missing = rearm_tokens - sum(map(_estimate_msg_budget_tokens, grown))
    regrown = grown + [{"role": "user", "content": "x" * (missing * 4)}]
    assert sum(map(_estimate_msg_budget_tokens, regrown)) >= rearm_tokens
    rearmed, n3 = c.prune_tool_results_only(regrown, current_tokens=_under_threshold)
    assert n3 >= 2
    assert rearmed is not regrown


def test_successful_full_compression_resets_proactive_runway():
    """A full compression establishes a fresh cache boundary and baseline."""
    c = _compressor(
        proactive_prune_tokens=48_000,
        proactive_prune_min_result_chars=8_000,
    )
    first, n1 = c.prune_tool_results_only(
        _build(8, big_indices={0, 1, 2}), current_tokens=120_000,
    )
    assert n1 >= 3

    history = [{"role": "system", "content": "sys"}]
    for i in range(12):
        history.append({
            "role": "user" if i % 2 == 0 else "assistant",
            "content": f"turn {i} " + ("x" * 1000),
        })
    c.tail_token_budget = 50
    with patch.object(c, "_generate_summary", return_value="summary"):
        compressed = c.compress(history, current_tokens=500_000, force=True)
    assert c._last_compression_made_progress is True
    assert len(compressed) < len(history)

    # The successful full boundary supersedes the old proactive-prune runway.
    fresh = _build(8, big_indices={0, 1, 2})
    result, pruned = c.prune_tool_results_only(fresh, current_tokens=48_000)
    assert pruned >= 3
    assert result is not fresh






# ---------------------------------------------------------------------------
# Salvage follow-ups: no-op caller contract, prompt-cache hysteresis gate,
# no-orphan pairing invariant, and the default-off behavior pin.
# ---------------------------------------------------------------------------








def test_min_reclaim_gate_default_and_clamp():
    """Default 4096; negative/None coerce to disabled (0)."""
    assert _compressor().proactive_prune_min_reclaim_tokens == 4096
    assert _compressor(proactive_prune_min_reclaim_tokens=0).proactive_prune_min_reclaim_tokens == 0
    assert _compressor(proactive_prune_min_reclaim_tokens=-5).proactive_prune_min_reclaim_tokens == 0
    assert _compressor(proactive_prune_min_reclaim_tokens=None).proactive_prune_min_reclaim_tokens == 0


def test_no_orphans_both_directions():
    """tool_call_id pairing survives the prune in BOTH directions: every
    surviving tool result has its assistant call, and every assistant tool_call
    has its result row (the #69830 test-pin rule — never assert exact surviving
    pair counts, only the pairing invariant)."""
    c = _compressor(
        proactive_prune_tokens=48_000,
        proactive_prune_min_result_chars=8_000,
        proactive_prune_min_reclaim_tokens=0,
    )
    msgs = _build(10, big_indices={0, 1, 2, 3, 4})
    result, pruned = c.prune_tool_results_only(msgs, current_tokens=120_000)
    assert pruned >= 1
    call_ids = set()
    for m in result:
        if m.get("role") == "assistant":
            for tc in m.get("tool_calls") or []:
                call_ids.add(tc["id"] if isinstance(tc, dict) else tc.id)
    result_ids = {m["tool_call_id"] for m in result if m.get("role") == "tool"}
    assert result_ids <= call_ids, "orphan tool results without a matching call"
    assert call_ids <= result_ids, "orphan tool calls without a matching result"


def test_unset_config_zero_behavior_change():
    """Pin: with the config knobs unset, the compressor behaves byte-identically
    to pre-feature main — the prune path is dead code and the full-compression
    Phase-1 caller keeps its 200-char floor."""
    c = _compressor()  # nothing configured
    assert c.proactive_prune_tokens == 0
    msgs = _build(8, big_indices={0, 1, 2})
    import copy
    snapshot = copy.deepcopy(msgs)
    result, pruned = c.prune_tool_results_only(msgs, current_tokens=10_000_000)
    assert pruned == 0
    assert result is msgs
    assert msgs == snapshot  # input never mutated
    # And the compression-path caller still prunes at the 200-char default floor
    # (min_prune_chars default unchanged).
    import inspect
    sig = inspect.signature(c._prune_old_tool_results)
    assert sig.parameters["min_prune_chars"].default == 200


# --- failure carve-out (Context Discipline W4 / §12 G4) --------------------------------------
# Contract: a tool result that records a FAILURE is never replaced by a 1-line summary, while an
# equally large SUCCESS result still is. Without this, a >min_prune_chars traceback in a non-tail
# position is reduced to "[terminal] ran `x` -> exit 1, 1 lines output" and the failure is gone.

_FAIL_BODY = (
    '{"exit_code": 1, "stderr": "' + ("frame locals dumped " * 400) + '", '
    '"traceback": "Traceback (most recent call last):\\n  File \\"app.py\\", line 7, in <module>'
    '\\nValueError: boom"}'
)
_OK_BODY = "column_a  column_b\n" * 700


def _prune_only(msgs):
    comp = _compressor()
    return comp._prune_old_tool_results(msgs, protect_tail_count=1, min_prune_chars=8000)


def test_failure_outlives_prune_where_equally_large_success_does_not():
    """The whole contract in one assertion pair: failure kept whole, success still summarized."""
    assert len(_FAIL_BODY) > 8000 and len(_OK_BODY) > 8000, "fixtures must exceed min_prune_chars"
    msgs = _build(0, set())
    msgs = [
        {"role": "system", "content": "sys"},
        _assistant_call("call_fail", args='{"command": "python app.py"}'),
        _tool_msg("call_fail", _FAIL_BODY),
        _assistant_call("call_ok", args='{"command": "ls"}'),
        _tool_msg("call_ok", _OK_BODY),
        {"role": "user", "content": "tail"},
    ]
    out, pruned = _prune_only(msgs)
    bodies = [m.get("content") for m in out]
    assert _FAIL_BODY in bodies, "a failing tool result was summarized away"
    assert _OK_BODY not in bodies, "an oversized success result should still be pruned"
    assert pruned >= 1


def test_failure_markers_cover_the_shapes_tools_actually_emit():
    """The predicate is the mechanism, so assert it directly on real-world failure shapes."""
    from agent.context_compressor import _looks_like_failure

    for shape in [
        '{"exit_code": 1, "stdout": ""}',
        '{"isError": true, "content": "nope"}',
        "Traceback (most recent call last):\n  File \"a.py\"",
        "KeyError: 'missing_key'",
        "subprocess.CalledProcessError: Command 'x' returned non-zero exit status 1",
        "FAILED tests/test_a.py::test_b - AssertionError",
        "bash: hermes: command not found",
        "sh: 1: Permission denied",
    ]:
        assert _looks_like_failure(shape), f"not detected as a failure: {shape!r}"
    for shape in [
        '{"exit_code": 0, "stdout": "all good"}',
        "column_a  column_b\n" * 50,
        "wrote 12 files; 0 errors",
    ]:
        assert not _looks_like_failure(shape), f"false positive on a success: {shape!r}"


# --- failure carve-out reaches the other two prune paths --------------------------------------
# Contract is the same as above (failure kept whole, equally-large success still shrunk), but
# these two paths are separate call sites from ``_prune_old_tool_results`` and each needed its
# own guard: salvage (last-resort shrink when a compression candidate grew) and the lean tail
# demotion (the default path, run on every compress() call).


def test_salvage_spares_a_failure_but_still_shrinks_an_equal_success():
    """salvage_grown_transcript fires when the candidate is bigger than the original (the real
    caller's trigger, agent/conversation_compression.py ~2857). Outside the newest 2 tool-result
    indices it must keep a failure verbatim and still placeholder an equally large success."""
    assert len(_FAIL_BODY) > _PRUNE_MIN_CHARS and len(_OK_BODY) > _PRUNE_MIN_CHARS
    candidate = [
        {"role": "system", "content": "sys"},
        _assistant_call("call_fail"), _tool_msg("call_fail", _FAIL_BODY),
        _assistant_call("call_ok"), _tool_msg("call_ok", _OK_BODY),
        _assistant_call("call_r1"), _tool_msg("call_r1", "ok"),
        _assistant_call("call_r2"), _tool_msg("call_r2", "ok"),
        {"role": "user", "content": "tail"},
    ]
    original = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "hello " * 3000},
        {"role": "assistant", "content": "hi"},
    ]
    from agent.model_metadata import estimate_messages_tokens_rough
    assert estimate_messages_tokens_rough(candidate) > estimate_messages_tokens_rough(original), (
        "candidate must be larger than original -- that is what triggers salvage in the real caller"
    )

    out = salvage_grown_transcript(original, candidate)
    assert out is not None, "salvage should have found a smaller candidate"
    bodies = [m.get("content") for m in out]
    assert _FAIL_BODY in bodies, "a failing tool result was salvaged away"
    assert _OK_BODY not in bodies, "an oversized success result should still be salvaged"
    assert _PRUNED_TOOL_PLACEHOLDER in bodies


def test_lean_tail_demotion_spares_a_failure_but_still_stubs_an_equal_success():
    """_demote_stale_tail_tools is the default lean-tail path run on every compress() call.
    Past the newest _LEAN_TAIL_KEEP_TOOL_ROUNDS rounds it must keep a failure verbatim and still
    demote an equally large success to a _lean_recovery_stub line."""
    assert len(_FAIL_BODY) > _LEAN_TAIL_DEMOTE_MIN_CHARS and len(_OK_BODY) > _LEAN_TAIL_DEMOTE_MIN_CHARS
    c = _compressor()
    msgs = [{"role": "system", "content": "sys"}]
    for i in range(9):  # more than the 6 protected rounds, so round 0 and 1 are demotion-eligible
        cid = f"call_{i}"
        msgs.append(_assistant_call(cid))
        if i == 0:
            msgs.append(_tool_msg(cid, _FAIL_BODY))
        elif i == 1:
            msgs.append(_tool_msg(cid, _OK_BODY))
        else:
            msgs.append(_tool_msg(cid, "ok"))

    out = c._demote_stale_tail_tools(msgs, tail_start=1)
    bodies = [m.get("content") for m in out if m.get("role") == "tool"]
    assert _FAIL_BODY in bodies, "a failing tool result was demoted away"
    assert _OK_BODY not in bodies, "an oversized success result should still be demoted"
    expected_stub = _lean_recovery_stub("", len(_OK_BODY), getattr(c, "_session_id", "") or "")
    assert expected_stub in bodies
