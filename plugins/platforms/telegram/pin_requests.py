"""File-driven pin requests for the Telegram adapter.

Local scripts cannot call Telegram, so they append JSON lines to a requests file; the gateway acts on them.
Only messages the bot itself pinned (recorded in the ledger) are ever unpinned.

Requests (``~/.hermes/data/pin-requests.jsonl``, env ``HERMES_PIN_REQUESTS``):
  {"op": "pin_next", "key", "chat_id", "thread_id", "match", "unpin_at", "expires_at", "requested_at"}
  {"op": "unpin", "key", "requested_at"}
Ledger (``~/.hermes/data/pins.json``, env ``HERMES_PIN_LEDGER``): ``{key: {key, chat_id, thread_id,
message_id, pinned_at, unpin_at}}``.

Every function is inert when the files do not exist and never raises into the caller's send path
(callers wrap in try/except as well).
"""
from __future__ import annotations

import fcntl
import json
import logging
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)

_MARKUP_CHARS = re.compile(r"[*_`~]")
_TAGS = re.compile(r"<[^>]+>")


def _data_dir() -> Path:
    from hermes_constants import get_hermes_home
    return get_hermes_home() / "data"


def requests_path() -> Path:
    env = os.environ.get("HERMES_PIN_REQUESTS")
    return Path(env) if env else _data_dir() / "pin-requests.jsonl"


def ledger_path() -> Path:
    env = os.environ.get("HERMES_PIN_LEDGER")
    return Path(env) if env else _data_dir() / "pins.json"


def normalize(text: str) -> str:
    """Strip markdown/HTML markup, collapse whitespace, casefold."""
    text = _TAGS.sub("", text or "")
    text = _MARKUP_CHARS.sub("", text)
    return " ".join(text.split()).casefold()


def _parse(ts: Any) -> Optional[datetime]:
    try:
        dt = datetime.fromisoformat(str(ts))
    except (TypeError, ValueError):
        return None
    return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)


def _now(now: Optional[datetime]) -> datetime:
    return now or datetime.now(timezone.utc)


def _expired(req: dict, now: datetime) -> bool:
    exp = _parse(req.get("expires_at"))
    return exp is None or exp <= now


# --- requests file (flock on the file itself; rewritten in place so the inode, hence the lock, is stable)

def append_request(req: dict) -> None:
    path = requests_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as fh:
        fcntl.flock(fh, fcntl.LOCK_EX)
        try:
            fh.write(json.dumps(req) + "\n")
            fh.flush()
        finally:
            fcntl.flock(fh, fcntl.LOCK_UN)


def _locked_rewrite(fn):
    """Run ``fn(requests: list[dict]) -> (keep: list[dict], result)`` under the file lock. Returns result,
    or None when the file does not exist."""
    path = requests_path()
    if not path.exists():
        return None
    with open(path, "r+") as fh:
        fcntl.flock(fh, fcntl.LOCK_EX)
        try:
            items = []
            for line in fh.read().splitlines():
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except ValueError:
                    continue
                if isinstance(obj, dict):
                    items.append(obj)
            keep, result = fn(items)
            fh.seek(0)
            fh.truncate()
            fh.write("".join(json.dumps(k) + "\n" for k in keep))
            fh.flush()
            return result
        finally:
            fcntl.flock(fh, fcntl.LOCK_UN)


# --- ledger (single writer: the gateway; atomic replace)

def ledger_load() -> dict:
    try:
        data = json.loads(ledger_path().read_text())
        return data if isinstance(data, dict) else {}
    except (OSError, ValueError):
        return {}


def _ledger_save(data: dict) -> None:
    path = ledger_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".{os.getpid()}.tmp")
    tmp.write_text(json.dumps(data, indent=1))
    os.replace(tmp, path)


def ledger_put(entry: dict) -> None:
    data = ledger_load()
    data[entry["key"]] = entry
    _ledger_save(data)


def ledger_pop(key: str) -> Optional[dict]:
    data = ledger_load()
    entry = data.pop(key, None)
    if entry is not None:
        _ledger_save(data)
    return entry


# --- processing

def take_pin_for_send(chat_id: str, thread_id: Optional[str], content: str,
                      now: Optional[datetime] = None) -> Optional[dict]:
    """Find, consume and return the pending pin_next request matching a message just sent to
    (chat_id, thread_id). Expired requests are dropped on the way. None when nothing matches."""
    now = _now(now)
    norm = normalize(content)

    def fn(items):
        keep, found = [], None
        for r in items:
            if r.get("op") == "pin_next" and _expired(r, now):
                continue
            if (found is None and r.get("op") == "pin_next"
                    and str(r.get("chat_id")) == str(chat_id)
                    and thread_id is not None and str(r.get("thread_id")) == str(thread_id)
                    and normalize(r.get("match", "")) and normalize(r["match"]) in norm):
                found = r
                continue
            keep.append(r)
        return keep, found

    return _locked_rewrite(fn)


def collect_due_unpins(now: Optional[datetime] = None) -> list[dict]:
    """Process unpin requests, ledger entries past ``unpin_at`` and expired pin_next requests. Returns the
    ledger entries (and only those) the caller must now unpin; they are already removed from the ledger."""
    now = _now(now)
    due: list[dict] = []

    def fn(items):
        keep = []
        for r in items:
            op = r.get("op")
            if op == "unpin":
                entry = ledger_pop(str(r.get("key")))
                if entry is not None:
                    due.append(entry)
            elif op == "pin_next":
                if not _expired(r, now):
                    keep.append(r)
            # unknown ops are dropped
        return keep, None

    if requests_path().exists():
        _locked_rewrite(fn)
    ledger = ledger_load()
    for key, entry in list(ledger.items()):
        at = _parse(entry.get("unpin_at")) if entry.get("unpin_at") else None
        if at is not None and at <= now:
            popped = ledger_pop(key)
            if popped is not None:
                due.append(popped)
    return due


def make_entry(req: dict, message_id: str, now: Optional[datetime] = None) -> dict:
    return {"key": req["key"], "chat_id": str(req["chat_id"]), "thread_id": str(req.get("thread_id")),
            "message_id": str(message_id), "pinned_at": _now(now).isoformat(), "unpin_at": req.get("unpin_at")}
