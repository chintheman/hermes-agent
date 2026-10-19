"""File-driven pin requests for the Telegram adapter.

Local scripts cannot call Telegram, so they append JSON lines to a requests file; the gateway acts on them.
Only messages the bot itself pinned (recorded in the ledger) are ever unpinned.

Requests (``~/.hermes/data/pin-requests.jsonl``, env ``HERMES_PIN_REQUESTS``):
  {"op": "pin_next", "key", "chat_id", "thread_id", "match", "unpin_at", "expires_at", "requested_at"}
  {"op": "unpin", "key", "requested_at"}
Ledger (``~/.hermes/data/pins.json``, env ``HERMES_PIN_LEDGER``): ``{key: {key, chat_id, thread_id,
message_id, pinned_at, unpin_at}}``.

Single-writer rule: the LEDGER is written by the gateway process only (one adapter, serialized by the
adapter's pin lock). Scripts must never write it. The REQUESTS file has many writers (scripts append, the
gateway consumes); every access to it holds an flock on the file.

A ledger that cannot be parsed is never treated as empty: it is copied to ``<ledger>.bak`` (once) and every
write is refused with ``LedgerCorrupt`` until a human fixes it.

Every function is inert when the files do not exist.
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


_corrupt_logged = False   # the corrupt-ledger error is logged once per process, not once per call


class LedgerCorrupt(Exception):
    """pins.json exists but is not a JSON object. Never overwritten; a .bak copy is kept."""


def _data_dir() -> Path:
    # The gateway and every script that writes requests MUST share the same HERMES_HOME (the gateway's launchd
    # plist sets /Users/0xsteamboat/.hermes; scripts default to ~/.hermes). A mismatch means requests are
    # written to a file the gateway never reads.
    home = os.environ.get("HERMES_HOME") or os.path.expanduser("~/.hermes")
    return Path(home) / "data"


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
    or None when the file is missing or empty. The file is rewritten only when its content changes."""
    path = requests_path()
    try:
        if not path.exists() or path.stat().st_size == 0:
            return None
    except OSError:
        return None
    with open(path, "r+") as fh:
        fcntl.flock(fh, fcntl.LOCK_EX)
        try:
            raw = fh.read()
            items = []
            for line in raw.splitlines():
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
            new = "".join(json.dumps(k) + "\n" for k in keep)
            if new != raw:
                fh.seek(0)
                fh.truncate()
                fh.write(new)
                fh.flush()
            return result
        finally:
            fcntl.flock(fh, fcntl.LOCK_UN)


def requeue_request(req: dict) -> None:
    """Put a consumed pin_next request back unchanged (same text, same original expiry) after the pin did not
    happen. It can only match a later send to the same chat and thread with the same start, and it still
    expires on schedule."""
    append_request(req)


# --- ledger (single writer: the gateway; atomic replace)

def ledger_load() -> dict:
    """The ledger, ``{}`` when the file does not exist or is empty. Raises LedgerCorrupt otherwise-bad content
    (after keeping a .bak copy)."""
    path = ledger_path()
    try:
        raw = path.read_text()
    except FileNotFoundError:
        return {}
    except OSError as exc:
        raise LedgerCorrupt("cannot read %s: %s" % (path, exc))
    if not raw.strip():
        return {}
    try:
        data = json.loads(raw)
        if isinstance(data, dict):
            return data
    except ValueError:
        pass
    bak = path.with_name(path.name + ".bak")
    try:
        if not bak.exists():
            bak.write_text(raw)
    except OSError:
        pass
    global _corrupt_logged
    if not _corrupt_logged:
        _corrupt_logged = True
        logger.error("pin ledger %s is corrupt; refusing to overwrite it (copy kept at %s)", path, bak)
    raise LedgerCorrupt("pin ledger %s is corrupt" % path)


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
    # the match may open the message or any of its first 3 non-empty lines (a short model preamble)
    heads = [norm] + [n for n in (normalize(l) for l in (content or "").splitlines()) if n][:3]

    def fn(items):
        keep, found = [], None
        for r in items:
            if r.get("op") == "pin_next" and _expired(r, now):
                continue
            if (found is None and r.get("op") == "pin_next"
                    and str(r.get("chat_id")) == str(chat_id)
                    and thread_id is not None and str(r.get("thread_id")) == str(thread_id)
                    and normalize(r.get("match", "")) and any(h.startswith(normalize(r["match"])) for h in heads)):
                found = r
                continue
            keep.append(r)
        return keep, found

    return _locked_rewrite(fn)


def collect_due_unpins(now: Optional[datetime] = None) -> list[dict]:
    """Process unpin requests, ledger entries past ``unpin_at`` and expired pin_next requests. Returns the
    ledger entries (and only those) the caller must now unpin; they are already removed from the ledger.

    An unpin request also deletes every pending pin_next with the same key that was requested EARLIER in the
    file (a session logged before its card was pinned). A corrupt ledger leaves everything untouched."""
    now = _now(now)
    due: list[dict] = []
    try:
        ledger_load()
    except LedgerCorrupt:
        def drop_expired(items):
            return [r for r in items if not (r.get("op") == "pin_next" and _expired(r, now))], None
        _locked_rewrite(drop_expired)
        return []

    def fn(items):
        keep = []
        for r in items:
            op = r.get("op")
            if op == "unpin":
                key = str(r.get("key"))
                keep = [k for k in keep if not (k.get("op") == "pin_next" and str(k.get("key")) == key)]
                entry = ledger_pop(key)
                if entry is not None:
                    due.append(entry)
            elif op == "pin_next":
                if not _expired(r, now):
                    keep.append(r)
            # unknown ops are dropped
        return keep, None

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
