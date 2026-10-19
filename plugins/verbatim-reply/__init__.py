"""verbatim-reply: scripted messages reach Chin as written, enforced in code.

Why: the model (deepseek-flash) ran the coach scripts, which print the exact message Chin should get, and then
reworded it: it dropped warnings from a session card, invented a rating Chin never gave and saved it as a
note, and saved a food photo as a workout note. Prompt rules were ignored, so the rules live here.

How it fits together (hook contracts are in hermes_cli/plugins.py VALID_HOOKS):

* A coach script writes the message to $HERMES_HOME/data/op/verbatim/<id>.md and prints the id, either as a
  JSON field ("verbatim_id") or as a marker line `[verbatim <id>]` (first AND last line of plain text).
* transform_tool_result scans each tool result for ids, reads the staged text and keeps it, keyed by
  (session_id, turn_id). It never changes the result.
* transform_llm_output: if the turn has staged texts, the model's final text is replaced by them, joined by a
  blank line. Otherwise, on Telegram only, em and en dashes are scrubbed from the text.
* pre_llm_call remembers the latest user message per session. pre_tool_call then blocks a terminal call to
  coach_log.py add-note unless its --text is a substring of that message.
* pre_tool_call also blocks edits of the coach skills (skill_manage, write_file, patch) for the interactive
  agent and for the background self-improvement review, which dispatches through the same hook.

Everything fails open except the two blocks, which fail closed on a parse error.
"""

from __future__ import annotations

import logging
import os
import re
import shlex
import threading
import time
from collections import OrderedDict
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)

ID_PATTERN = r"[0-9A-Za-z_-]{8,64}"
ID_RE = re.compile(r"^%s$" % ID_PATTERN)
# {"verbatim_id": "<id>"}, also when the JSON sits escaped inside another JSON string (\"verbatim_id\": \"<id>\").
_JSON_ID = re.compile(r'verbatim_id\\*"\s*:\s*\\*"(%s)\\*"' % ID_PATTERN)
_MARKER_ID = re.compile(r"\[verbatim (%s)\]" % ID_PATTERN)

MAX_TEXT_BYTES = 200_000
MAX_FILE_AGE_S = 3600
PENDING_TTL_S = 30 * 60
PENDING_MAX = 256
USER_MSG_MAX = 256

NOTE_BLOCK_MESSAGE = "add-note saves Chin's own words only. Quote his message exactly, or do not save a note."
SKILL_LOCK_MESSAGE = "This skill is locked. Changes to it go through Claude Code."

LOCKED_SKILLS = {"health-wellness-coach", "body-composition-blocks"}
_LOCKED_TEXT_PATH = re.compile(
    r"(^|/)skills/(productivity/health-wellness-coach|health/body-composition-blocks)(/|$)"
)

_LOCK = threading.Lock()
# (session_id, turn_id) -> {"ts": float, "ids": [..], "texts": [..]}
_PENDING: "OrderedDict[tuple, dict]" = OrderedDict()
# session_id -> (ts, latest user message)
_LATEST_USER: "OrderedDict[str, tuple]" = OrderedDict()


# ---------------------------------------------------------------------------
# Staged text
# ---------------------------------------------------------------------------

def verbatim_dir() -> Path:
    env = os.environ.get("HERMES_VERBATIM_DIR")
    if env:
        return Path(env)
    home = os.environ.get("HERMES_HOME") or os.path.expanduser("~/.hermes")
    return Path(home) / "data" / "op" / "verbatim"


def find_ids(result: str) -> list:
    """Verbatim ids in a tool result, in order of appearance, without repeats."""
    hits = []
    for rx in (_JSON_ID, _MARKER_ID):
        hits += [(m.start(), m.group(1)) for m in rx.finditer(result)]
    seen, out = set(), []
    for _pos, vid in sorted(hits):
        if vid not in seen:
            seen.add(vid)
            out.append(vid)
    return out


def read_staged(vid: str) -> Optional[str]:
    """The staged text for an id, or None. The real path must stay inside the verbatim dir."""
    if not isinstance(vid, str) or not ID_RE.match(vid):
        return None
    base = os.path.realpath(verbatim_dir())
    path = os.path.realpath(os.path.join(base, vid + ".md"))
    if os.path.dirname(path) != base:
        logger.info("verbatim-reply: id %r resolves outside the verbatim dir, ignored", vid)
        return None
    try:
        st = os.stat(path)
        if not os.path.isfile(path) or st.st_size > MAX_TEXT_BYTES:
            return None
        if time.time() - st.st_mtime > MAX_FILE_AGE_S:
            return None
        with open(path, encoding="utf-8") as fh:
            text = fh.read()
    except (OSError, UnicodeDecodeError):
        return None
    return text.rstrip("\n") if text.strip() else None


def _prune(now: float) -> None:
    for key in [k for k, v in _PENDING.items() if now - v["ts"] > PENDING_TTL_S]:
        _PENDING.pop(key, None)
    while len(_PENDING) > PENDING_MAX:
        _PENDING.popitem(last=False)


def _on_transform_tool_result(
    tool_name: str = "", args: Any = None, result: Any = None,
    session_id: str = "", turn_id: str = "", **_: Any,
) -> Optional[str]:
    """Collect staged texts named in this result. Never changes the result."""
    try:
        if not isinstance(result, str) or "verbatim" not in result:
            return None
        ids = find_ids(result)
        if not ids:
            return None
        key = (session_id or "", turn_id or "")
        now = time.time()
        with _LOCK:
            _prune(now)
            entry = _PENDING.get(key)
            known = set(entry["ids"]) if entry else set()
        for vid in ids:
            if vid in known:
                continue
            text = read_staged(vid)
            if text is None:
                logger.info("verbatim-reply: no staged text for id %s (tool %s)", vid, tool_name)
                continue
            with _LOCK:
                entry = _PENDING.setdefault(key, {"ts": now, "ids": [], "texts": []})
                if vid not in entry["ids"]:
                    entry["ids"].append(vid)
                    entry["texts"].append(text)
                    entry["ts"] = now
                known.add(vid)
    except Exception as exc:  # noqa: BLE001
        logger.debug("verbatim-reply transform_tool_result failed (non-fatal): %s", exc)
    return None


# ---------------------------------------------------------------------------
# Dash scrub (Telegram only, when there is no scripted text)
# ---------------------------------------------------------------------------

_CODE = re.compile(r"(```.*?(?:```|\Z)|~~~.*?(?:~~~|\Z)|`{1,2}[^`\n]+`{1,2})", re.DOTALL)
_DASH = "—–"


def _scrub_plain(s: str) -> str:
    s = re.sub(r"(?m)^([ \t]*)[%s][ \t]+" % _DASH, r"\1- ", s)          # a dash used as a bullet
    s = re.sub(r"(?m)[ \t]*[%s][ \t]*$" % _DASH, "", s)                 # a dash that ends a line
    s = re.sub(r"(?<=\d)[ \t]*[%s][ \t]*(?=\d)" % _DASH, "-", s)        # 5-7, 5 - 7
    s = re.sub(r"[ \t]+[%s][ \t]+" % _DASH, ", ", s)                    # " - " as a pause
    s = re.sub(r"[ \t]*[%s][ \t]*" % _DASH, ", ", s)                    # any other
    return re.sub(r",(?:[ \t]*,)+", ",", s)


def scrub_dashes(text: str) -> str:
    """Replace em and en dashes outside inline code and fenced code blocks."""
    parts = _CODE.split(text)
    return "".join(p if i % 2 else _scrub_plain(p) for i, p in enumerate(parts))


def _on_transform_llm_output(
    response_text: str = "", session_id: str = "", model: str = "", platform: str = "",
    turn_id: str = "", **_: Any,
) -> Optional[str]:
    try:
        key = (session_id or "", turn_id or "")
        with _LOCK:
            _prune(time.time())
            entry = _PENDING.pop(key, None)
        if entry and entry["texts"]:
            logger.info("verbatim-reply: replaced model text with %d scripted message(s)", len(entry["texts"]))
            return "\n\n".join(entry["texts"])
        if (platform or "").lower() == "telegram" and isinstance(response_text, str):
            fixed = scrub_dashes(response_text)
            if fixed != response_text:
                return fixed
    except Exception as exc:  # noqa: BLE001
        logger.debug("verbatim-reply transform_llm_output failed (non-fatal): %s", exc)
    return None


# ---------------------------------------------------------------------------
# add-note guard
# ---------------------------------------------------------------------------

def _message_text(msg: Any) -> str:
    if isinstance(msg, str):
        return msg
    if isinstance(msg, list):
        out = []
        for part in msg:
            if isinstance(part, dict) and isinstance(part.get("text"), str):
                out.append(part["text"])
            elif isinstance(part, str):
                out.append(part)
        return "\n".join(out)
    return ""


def _on_pre_llm_call(session_id: str = "", user_message: Any = None, **_: Any) -> None:
    try:
        text = _message_text(user_message)
        if not session_id or not text.strip():
            return None
        with _LOCK:
            _LATEST_USER[session_id] = (time.time(), text)
            _LATEST_USER.move_to_end(session_id)
            while len(_LATEST_USER) > USER_MSG_MAX:
                _LATEST_USER.popitem(last=False)
    except Exception as exc:  # noqa: BLE001
        logger.debug("verbatim-reply pre_llm_call failed (non-fatal): %s", exc)
    return None


def normalize(s: str) -> str:
    s = re.sub(r"[^\w\s]|_", "", str(s).lower())
    return " ".join(s.split())


_OPERATORS = {";", "&", "&&", "|", "||", "(", ")", ";;", "|&"}


def _add_note_texts(command: str) -> Optional[list]:
    """The --text values of every coach_log.py add-note call in `command`.
    [] when the command has none; None when it has one that cannot be parsed."""
    if "add-note" not in command or "coach_log" not in command:
        return []
    try:
        lex = shlex.shlex(command, posix=True, punctuation_chars=True)
        lex.whitespace_split = True
        lex.commenters = ""
        tokens = list(lex)
    except ValueError:
        return None
    segments, cur = [], []
    for t in tokens:
        if t in _OPERATORS or (t and set(t) <= set(";&|()")):
            segments.append(cur)
            cur = []
        else:
            cur.append(t)
    segments.append(cur)
    texts = []
    for seg in segments:
        if not any(t.endswith("coach_log.py") or t == "coach_log" for t in seg) or "add-note" not in seg:
            continue
        found = None
        for i, t in enumerate(seg):
            if t == "--text":
                if i + 1 >= len(seg):
                    return None
                found = seg[i + 1]
            elif t.startswith("--text="):
                found = t[len("--text="):]
        if found is None:
            return None
        texts.append(found)
    if not texts:
        return None   # the words are there but no segment is a call we understand: fail closed
    return texts


def check_add_note(command: str, latest_user: str) -> Optional[str]:
    """None to allow; the block message otherwise."""
    texts = _add_note_texts(command)
    if texts == []:
        return None
    if texts is None:
        return NOTE_BLOCK_MESSAGE
    said = normalize(latest_user or "")
    for t in texts:
        n = normalize(t)
        if not n or n not in said:
            return NOTE_BLOCK_MESSAGE
    return None


# ---------------------------------------------------------------------------
# Skill lock
# ---------------------------------------------------------------------------

def _locked_roots() -> list:
    home = os.environ.get("HERMES_HOME") or os.path.expanduser("~/.hermes")
    roots = [
        os.path.join(home, "skills", "productivity", "health-wellness-coach"),
        os.path.join(home, "skills", "health", "body-composition-blocks"),
        os.path.expanduser("~/projects/health-wellness-coach/skill"),
    ]
    return roots + [os.path.realpath(r) for r in roots]


def _under(path: str, root: str) -> bool:
    return path == root or path.startswith(root.rstrip(os.sep) + os.sep)


def path_is_locked(path: Any) -> bool:
    if not isinstance(path, str) or not path.strip():
        return False
    p = os.path.expanduser(path.strip())
    # 1. The path as written (cheap, no filesystem access).
    plain = os.path.normpath(p).replace(os.sep, "/")
    if _LOCKED_TEXT_PATH.search(plain):
        return True
    # 2. The path with symlinks resolved, against the locked roots (the repo's skill/ is a symlink target).
    candidates = set()
    for make in (os.path.realpath, lambda x: os.path.realpath(os.path.abspath(x))):
        try:
            candidates.add(make(p).replace(os.sep, "/"))
        except Exception:  # noqa: BLE001 - an unresolvable path is judged on its text alone
            pass
    roots = [r.replace(os.sep, "/") for r in _locked_roots()]
    for c in candidates | {plain}:
        if _LOCKED_TEXT_PATH.search(c) or any(_under(c, r) for r in roots):
            return True
    return False


_V4A_PATH = re.compile(r"^\*\*\*\s*(?:Update File|Add File|Delete File|Move to):\s*(.+?)\s*$", re.MULTILINE)


def _skill_manage_locked(args: dict) -> bool:
    ops = args.get("operations")
    ops = [o for o in ops if isinstance(o, dict)] if isinstance(ops, list) else [args]
    for op in ops:
        name = str(op.get("name") or "").strip().lower().replace("\\", "/").split("/")[-1]
        if name in LOCKED_SKILLS:
            return True
    return False


def is_locked_call(tool_name: str, args: Any) -> bool:
    if not isinstance(args, dict):
        return False
    if tool_name == "skill_manage":
        return _skill_manage_locked(args)
    if tool_name in ("write_file", "patch"):
        if path_is_locked(args.get("path")) or path_is_locked(args.get("file_path")):
            return True
        patch = args.get("patch")
        if isinstance(patch, str):
            return any(path_is_locked(m) for m in _V4A_PATH.findall(patch))
    return False


def _on_pre_tool_call(
    tool_name: str = "", args: Any = None, session_id: str = "", **_: Any,
) -> Optional[dict]:
    try:
        if is_locked_call(tool_name, args):
            logger.info("verbatim-reply: blocked %s on a locked skill", tool_name)
            return {"action": "block", "message": SKILL_LOCK_MESSAGE}
    except Exception as exc:  # noqa: BLE001
        logger.debug("verbatim-reply skill lock check failed: %s", exc)
    try:
        if tool_name == "terminal" and isinstance(args, dict) and isinstance(args.get("command"), str):
            if "add-note" in args["command"] and "coach_log" in args["command"]:
                with _LOCK:
                    latest = _LATEST_USER.get(session_id or "", (0, ""))[1]
                msg = check_add_note(args["command"], latest)
                if msg:
                    logger.info("verbatim-reply: blocked an add-note that is not Chin's words")
                    return {"action": "block", "message": msg}
    except Exception as exc:  # noqa: BLE001
        logger.info("verbatim-reply: add-note check failed, blocking: %s", exc)
        return {"action": "block", "message": NOTE_BLOCK_MESSAGE}
    return None


def register(ctx) -> None:
    ctx.register_hook("transform_tool_result", _on_transform_tool_result)
    ctx.register_hook("transform_llm_output", _on_transform_llm_output)
    ctx.register_hook("pre_llm_call", _on_pre_llm_call)
    ctx.register_hook("pre_tool_call", _on_pre_tool_call)
