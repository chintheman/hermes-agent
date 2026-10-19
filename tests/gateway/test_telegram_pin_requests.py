"""Adapter side of pin requests: pin after a successful send, ledger-only unpin, errors swallowed."""
import asyncio
import json
from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, MagicMock

import pytest

from gateway.config import PlatformConfig
from gateway.platforms.base import SendResult
from plugins.platforms.telegram import pin_requests as pr
from plugins.platforms.telegram.adapter import TelegramAdapter

CHAT, THREAD = "-1004389869012", "5"


def _now():
    return datetime.now(timezone.utc)


def _iso(minutes):
    return (_now() + timedelta(minutes=minutes)).isoformat()


@pytest.fixture(autouse=True)
def paths(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_PIN_REQUESTS", str(tmp_path / "pin-requests.jsonl"))
    monkeypatch.setenv("HERMES_PIN_LEDGER", str(tmp_path / "pins.json"))
    return tmp_path


def _adapter(msg_ids=("101", "102", "103")):
    adapter = TelegramAdapter(PlatformConfig(enabled=True, token="***"))
    adapter._rich_send_disabled = True
    adapter._bot = MagicMock()
    ids = iter(msg_ids)

    async def send_message(text, **_kw):
        return MagicMock(message_id=int(next(ids)))

    adapter._bot.send_message = AsyncMock(side_effect=send_message)
    adapter._bot.pin_chat_message = AsyncMock(return_value=True)
    adapter._bot.unpin_chat_message = AsyncMock(return_value=True)
    return adapter


def _req(key="op-brief", match="Morning brief", chat=CHAT, thread=THREAD, expires=30, unpin_at=None):
    return {"op": "pin_next", "key": key, "chat_id": chat, "thread_id": thread, "match": match,
            "unpin_at": unpin_at, "expires_at": _iso(expires), "requested_at": _iso(0)}


def _write(*items):
    with open(pr.requests_path(), "a") as fh:
        for i in items:
            fh.write(json.dumps(i) + "\n")


META = {"thread_id": THREAD}


async def _send(a, *args, **kw):
    """send(), then wait for the fire-and-forget pin task (send itself must not wait for it)."""
    res = await a.send(*args, **kw)
    await a._drain_pin_tasks()
    return res


@pytest.mark.asyncio
async def test_matching_send_is_pinned_silently_and_ledgered():
    a = _adapter()
    _write(_req())
    res = await _send(a, CHAT, "**Morning brief** Monday", metadata=META)
    assert res.success and res.message_id == "101"
    a._bot.pin_chat_message.assert_awaited_once()
    args, kw = a._bot.pin_chat_message.await_args
    assert (args and args[0] == CHAT or kw.get("chat_id") == CHAT)
    assert kw["disable_notification"] is True
    entry = pr.ledger_load()["op-brief"]
    assert entry["message_id"] == "101" and entry["chat_id"] == CHAT and entry["thread_id"] == THREAD
    assert pr.requests_path().read_text().strip() == ""


@pytest.mark.asyncio
async def test_replacing_same_key_unpins_old_first():
    a = _adapter()
    order = []
    a._bot.unpin_chat_message = AsyncMock(side_effect=lambda *x, **k: order.append(("unpin", k)))
    a._bot.pin_chat_message = AsyncMock(side_effect=lambda *x, **k: order.append(("pin", k)))
    _write(_req())
    await _send(a, CHAT, "Morning brief one", metadata=META)
    _write(_req())
    await _send(a, CHAT, "Morning brief two", metadata=META)
    assert [o[0] for o in order] == ["pin", "unpin", "pin"]
    assert order[1][1]["message_id"] == 101
    assert pr.ledger_load()["op-brief"]["message_id"] == "102"


@pytest.mark.asyncio
async def test_non_matching_thread_not_pinned_and_request_kept():
    a = _adapter()
    _write(_req())
    await _send(a, CHAT, "Morning brief", metadata={"thread_id": "9"})
    a._bot.pin_chat_message.assert_not_awaited()
    assert len([x for x in pr.requests_path().read_text().splitlines() if x.strip()]) == 1


@pytest.mark.asyncio
async def test_expired_request_dropped_not_pinned():
    a = _adapter()
    _write(_req(expires=-5))
    await _send(a, CHAT, "Morning brief", metadata=META)
    a._bot.pin_chat_message.assert_not_awaited()
    assert pr.requests_path().read_text().strip() == ""


@pytest.mark.asyncio
async def test_unpin_request_unpins_ledger_entry_only():
    a = _adapter()
    pr.ledger_put({"key": "op-workout", "chat_id": CHAT, "thread_id": THREAD, "message_id": "77",
                   "pinned_at": _iso(-5), "unpin_at": None})
    _write({"op": "unpin", "key": "op-workout", "requested_at": _iso(0)})
    await a._pin_maintenance()
    a._bot.unpin_chat_message.assert_awaited_once()
    assert a._bot.unpin_chat_message.await_args.kwargs["message_id"] == 77
    assert pr.ledger_load() == {}


@pytest.mark.asyncio
async def test_unpin_at_passed_unpins():
    a = _adapter()
    pr.ledger_put({"key": "op-workout", "chat_id": CHAT, "thread_id": THREAD, "message_id": "77",
                   "pinned_at": _iso(-500), "unpin_at": _iso(-1)})
    _write(_req(key="other"))
    await a._pin_maintenance()
    a._bot.unpin_chat_message.assert_awaited_once()


@pytest.mark.asyncio
async def test_message_not_in_ledger_is_never_unpinned():
    a = _adapter()
    _write({"op": "unpin", "key": "guide", "message_id": "1", "requested_at": _iso(0)})
    await a._pin_maintenance()
    a._bot.unpin_chat_message.assert_not_awaited()


@pytest.mark.asyncio
async def test_pin_api_exception_does_not_break_send():
    a = _adapter()
    a._bot.pin_chat_message = AsyncMock(side_effect=RuntimeError("boom"))
    _write(_req())
    res = await _send(a, CHAT, "Morning brief", metadata=META)
    assert res.success and res.message_id == "101"
    assert "op-brief" not in pr.ledger_load()


@pytest.mark.asyncio
async def test_missing_files_inert_no_pin_calls(paths):
    a = _adapter()
    res = await _send(a, CHAT, "Morning brief", metadata=META)
    assert res.success
    a._bot.pin_chat_message.assert_not_awaited()
    a._bot.unpin_chat_message.assert_not_awaited()
    assert not pr.requests_path().exists() and not pr.ledger_path().exists()


@pytest.mark.asyncio
async def test_rich_path_success_also_pins():
    a = _adapter()
    a._should_attempt_rich = MagicMock(return_value=True)
    a._try_send_rich = AsyncMock(return_value=SendResult(success=True, message_id="555"))
    _write(_req())
    res = await _send(a, CHAT, "Morning brief", metadata=META)
    assert res.message_id == "555"
    assert a._bot.pin_chat_message.await_args.kwargs["message_id"] == 555


@pytest.mark.asyncio
async def test_chunked_send_pins_first_chunk():
    a = _adapter(msg_ids=("201", "202", "203", "204"))
    long = "Morning brief\n" + "\n".join(" ".join(f"w{i * 20 + j}" for j in range(20)) for i in range(80))
    _write(_req())
    await _send(a, CHAT, long, metadata=META)
    assert a._bot.pin_chat_message.await_args.kwargs["message_id"] == 201


@pytest.mark.asyncio
async def test_failed_send_does_not_consume_request():
    a = _adapter()
    a._bot.send_message = AsyncMock(side_effect=RuntimeError("down"))
    _write(_req())
    res = await _send(a, CHAT, "Morning brief", metadata=META)
    assert not res.success
    a._bot.pin_chat_message.assert_not_awaited()
    assert len([x for x in pr.requests_path().read_text().splitlines() if x.strip()]) == 1


@pytest.mark.asyncio
async def test_periodic_task_starts_and_stops():
    a = _adapter()
    a._start_pin_loop(interval=0.01)
    assert a._pin_loop_task is not None and not a._pin_loop_task.done()
    pr.ledger_put({"key": "k", "chat_id": CHAT, "thread_id": THREAD, "message_id": "9",
                   "pinned_at": _iso(-9), "unpin_at": _iso(-1)})
    _write(_req(key="other"))
    await asyncio.sleep(0.1)
    a._bot.unpin_chat_message.assert_awaited()
    await a._stop_pin_loop()
    assert a._pin_loop_task is None


@pytest.mark.asyncio
async def test_send_returns_before_the_pin_sequence_finishes():
    a = _adapter()
    gate = asyncio.Event()

    async def slow_pin(*_a, **_k):
        await gate.wait()

    a._bot.pin_chat_message = AsyncMock(side_effect=slow_pin)
    _write(_req())
    res = await asyncio.wait_for(a.send(CHAT, "Morning brief", metadata=META), timeout=2)
    assert res.success and res.message_id == "101"
    assert a._pin_tasks  # still running, referenced
    gate.set()
    await a._drain_pin_tasks()
    assert not a._pin_tasks
    assert pr.ledger_load()["op-brief"]["message_id"] == "101"


@pytest.mark.asyncio
async def test_ledger_entry_is_written_before_the_pin_call():
    a = _adapter()
    seen = {}

    async def pin(*_a, **kw):
        seen["ledger"] = dict(pr.ledger_load())

    a._bot.pin_chat_message = AsyncMock(side_effect=pin)
    _write(_req())
    await _send(a, CHAT, "Morning brief", metadata=META)
    assert seen["ledger"]["op-brief"]["message_id"] == "101"


@pytest.mark.asyncio
async def test_failed_pin_rolls_the_new_ledger_entry_back():
    a = _adapter()
    a._bot.pin_chat_message = AsyncMock(side_effect=RuntimeError("no rights"))
    _write(_req())
    await _send(a, CHAT, "Morning brief", metadata=META)
    assert pr.ledger_load() == {}


@pytest.mark.asyncio
async def test_old_unpin_failure_restores_old_entry_and_skips_new_pin():
    a = _adapter()
    pr.ledger_put({"key": "op-brief", "chat_id": CHAT, "thread_id": THREAD, "message_id": "50",
                   "pinned_at": _iso(-60), "unpin_at": None})
    a._bot.unpin_chat_message = AsyncMock(side_effect=RuntimeError("Timed out"))
    _write(_req())
    await _send(a, CHAT, "Morning brief", metadata=META)
    assert pr.ledger_load()["op-brief"]["message_id"] == "50"
    a._bot.pin_chat_message.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize("err", ["Bad Request: message to unpin not found", "Bad Request: message is not pinned"])
async def test_old_message_already_gone_is_fine_and_new_one_pins(err):
    a = _adapter()
    pr.ledger_put({"key": "op-brief", "chat_id": CHAT, "thread_id": THREAD, "message_id": "50",
                   "pinned_at": _iso(-60), "unpin_at": None})
    a._bot.unpin_chat_message = AsyncMock(side_effect=RuntimeError(err))
    _write(_req())
    await _send(a, CHAT, "Morning brief", metadata=META)
    assert pr.ledger_load()["op-brief"]["message_id"] == "101"


@pytest.mark.asyncio
async def test_corrupt_ledger_means_no_pin_and_file_untouched(paths):
    a = _adapter()
    pr.ledger_path().write_text("garbage")
    _write(_req())
    res = await _send(a, CHAT, "Morning brief", metadata=META)
    assert res.success
    a._bot.pin_chat_message.assert_not_awaited()
    assert pr.ledger_path().read_text() == "garbage"


@pytest.mark.asyncio
async def test_quoting_reply_does_not_take_the_pin():
    a = _adapter()
    _write(_req())
    await _send(a, CHAT, "Re: the Morning brief you sent", metadata=META)
    a._bot.pin_chat_message.assert_not_awaited()


@pytest.mark.asyncio
async def test_unpin_request_cancels_pending_pin_next_so_the_card_never_pins():
    a = _adapter()
    _write(_req(key="op-workout", match="Workout A"), {"op": "unpin", "key": "op-workout", "requested_at": _iso(0)})
    await _send(a, CHAT, "Workout A", metadata=META)
    a._bot.pin_chat_message.assert_not_awaited()


@pytest.mark.asyncio
async def test_final_edit_pins_the_edited_message():
    a = _adapter()
    a._bot.edit_message_text = AsyncMock(return_value=MagicMock(message_id=900))
    _write(_req())
    await a.edit_message(CHAT, "900", "Morning brief Monday", finalize=True, metadata=META)
    await a._drain_pin_tasks()
    a._bot.pin_chat_message.assert_awaited_once()
    assert a._bot.pin_chat_message.await_args.kwargs["message_id"] == 900
    assert pr.ledger_load()["op-brief"]["message_id"] == "900"


@pytest.mark.asyncio
async def test_interim_edit_does_not_pin():
    a = _adapter()
    a._bot.edit_message_text = AsyncMock(return_value=MagicMock(message_id=900))
    _write(_req())
    await a.edit_message(CHAT, "900", "Morning brief Mon", finalize=False, metadata=META)
    await a._drain_pin_tasks()
    a._bot.pin_chat_message.assert_not_awaited()
