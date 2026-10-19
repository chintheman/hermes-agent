"""Pure-function tests for the file-driven pin request module (no Telegram, no adapter)."""
import json
import multiprocessing
from datetime import datetime, timedelta, timezone

import pytest

from plugins.platforms.telegram import pin_requests as pr

NOW = datetime(2026, 10, 19, 7, 30, tzinfo=timezone.utc)
CHAT, THREAD = "-1004389869012", "5"


def iso(delta_minutes):
    return (NOW + timedelta(minutes=delta_minutes)).isoformat()


@pytest.fixture(autouse=True)
def paths(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_PIN_REQUESTS", str(tmp_path / "pin-requests.jsonl"))
    monkeypatch.setenv("HERMES_PIN_LEDGER", str(tmp_path / "pins.json"))
    return tmp_path


def req(key="op-brief", match="Morning brief", chat=CHAT, thread=THREAD, expires=30, unpin_at=None):
    return {"op": "pin_next", "key": key, "chat_id": chat, "thread_id": thread, "match": match,
            "unpin_at": unpin_at, "expires_at": iso(expires), "requested_at": iso(0)}


def write_requests(*items):
    with open(pr.requests_path(), "a") as fh:
        for item in items:
            fh.write(json.dumps(item) + "\n")


def pending_lines():
    p = pr.requests_path()
    return [json.loads(x) for x in p.read_text().splitlines() if x.strip()] if p.exists() else []


def test_normalize_strips_markup_and_folds_case():
    assert pr.normalize("**Morning** _Brief_ `x`\n  <b>Today</b>~") == "morning brief x today"


def test_match_and_pin_returns_request_and_consumes_it():
    write_requests(req())
    got = pr.take_pin_for_send(CHAT, THREAD, "*Morning brief* for Monday", now=NOW)
    assert got["key"] == "op-brief"
    assert pending_lines() == []


def test_non_matching_chat_or_thread_or_text_not_taken():
    write_requests(req())
    assert pr.take_pin_for_send("-100other", THREAD, "Morning brief", now=NOW) is None
    assert pr.take_pin_for_send(CHAT, "9", "Morning brief", now=NOW) is None
    assert pr.take_pin_for_send(CHAT, None, "Morning brief", now=NOW) is None
    assert pr.take_pin_for_send(CHAT, THREAD, "something else", now=NOW) is None
    assert len(pending_lines()) == 1


def test_expired_request_is_dropped_not_matched():
    write_requests(req(expires=-1))
    assert pr.take_pin_for_send(CHAT, THREAD, "Morning brief", now=NOW) is None
    assert pending_lines() == []


def test_missing_file_is_inert(paths):
    assert pr.take_pin_for_send(CHAT, THREAD, "Morning brief", now=NOW) is None
    assert pr.collect_due_unpins(now=NOW) == []
    assert not pr.requests_path().exists()
    assert not pr.ledger_path().exists()


def test_unpin_request_returns_ledger_entry_and_drops_it():
    pr.ledger_put({"key": "op-workout", "chat_id": CHAT, "thread_id": THREAD, "message_id": "77",
                   "pinned_at": iso(0), "unpin_at": None})
    write_requests({"op": "unpin", "key": "op-workout", "requested_at": iso(1)})
    due = pr.collect_due_unpins(now=NOW)
    assert [d["message_id"] for d in due] == ["77"]
    assert pr.ledger_load() == {}
    assert pending_lines() == []


def test_unpin_request_for_unknown_key_touches_nothing():
    write_requests({"op": "unpin", "key": "guide", "requested_at": iso(1)})
    assert pr.collect_due_unpins(now=NOW) == []


def test_unpin_at_passed_is_due_and_future_is_not():
    pr.ledger_put({"key": "a", "chat_id": CHAT, "thread_id": THREAD, "message_id": "1",
                   "pinned_at": iso(-100), "unpin_at": iso(-1)})
    pr.ledger_put({"key": "b", "chat_id": CHAT, "thread_id": THREAD, "message_id": "2",
                   "pinned_at": iso(-100), "unpin_at": iso(60)})
    pr.ledger_put({"key": "c", "chat_id": CHAT, "thread_id": THREAD, "message_id": "3",
                   "pinned_at": iso(-100), "unpin_at": None})
    write_requests(req(key="zzz"))  # make file exist
    due = pr.collect_due_unpins(now=NOW)
    assert [d["message_id"] for d in due] == ["1"]
    assert sorted(pr.ledger_load()) == ["b", "c"]


def test_only_ledger_entries_are_ever_returned():
    """An unpin request names a key; a message id that is not in the ledger can never come back."""
    pr.ledger_put({"key": "op-brief", "chat_id": CHAT, "thread_id": THREAD, "message_id": "5",
                   "pinned_at": iso(0), "unpin_at": None})
    write_requests({"op": "unpin", "key": "op-brief", "message_id": "999", "requested_at": iso(1)})
    due = pr.collect_due_unpins(now=NOW)
    assert [d["message_id"] for d in due] == ["5"]


def test_replace_returns_previous_entry_for_same_key():
    pr.ledger_put({"key": "op-brief", "chat_id": CHAT, "thread_id": THREAD, "message_id": "5",
                   "pinned_at": iso(0), "unpin_at": None})
    old = pr.ledger_pop("op-brief")
    assert old["message_id"] == "5"
    assert pr.ledger_pop("op-brief") is None


def test_ledger_write_is_atomic_replace(paths):
    pr.ledger_put({"key": "k", "chat_id": CHAT, "thread_id": THREAD, "message_id": "1",
                   "pinned_at": iso(0), "unpin_at": None})
    assert json.loads(pr.ledger_path().read_text())["k"]["message_id"] == "1"
    assert [p.name for p in paths.iterdir() if p.name.endswith(".tmp")] == []


def test_corrupt_lines_skipped():
    with open(pr.requests_path(), "w") as fh:
        fh.write("not json\n")
        fh.write(json.dumps(req()) + "\n")
    assert pr.take_pin_for_send(CHAT, THREAD, "Morning brief", now=NOW)["key"] == "op-brief"


def _appender(path, n, tag):
    import os
    os.environ["HERMES_PIN_REQUESTS"] = path
    from plugins.platforms.telegram import pin_requests as m
    for i in range(n):
        m.append_request({"op": "pin_next", "key": f"{tag}{i}", "chat_id": "c", "thread_id": "t",
                          "match": "zzz", "unpin_at": None, "expires_at": "2999-01-01T00:00:00+00:00",
                          "requested_at": "x"})


def test_concurrent_append_during_processing_is_not_lost():
    ctx = multiprocessing.get_context("fork")
    procs = [ctx.Process(target=_appender, args=(str(pr.requests_path()), 40, t)) for t in ("a", "b")]
    for p in procs:
        p.start()
    for _ in range(30):  # process repeatedly while writers append
        pr.collect_due_unpins(now=NOW)
    for p in procs:
        p.join()
    keys = {r["key"] for r in pending_lines()}
    assert keys == {f"{t}{i}" for t in ("a", "b") for i in range(40)}
