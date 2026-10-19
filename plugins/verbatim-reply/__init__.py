"""verbatim-reply: scripted messages reach Chin as written, enforced in code.

Why: the model (deepseek-flash) ran the coach scripts, which print the exact message Chin should get, and then
reworded it: it dropped warnings from a session card, invented a rating Chin never gave and saved it as a
note, and saved a food photo as a workout note. Prompt rules were ignored, so the rules live here.

How it fits together (hook contracts are in hermes_cli/plugins.py VALID_HOOKS):

* A coach script writes the message to $HWC_VERBATIM_DIR (default <hermes home>/data/op/verbatim)/<id>.md
  and prints the id, either as a JSON field ("verbatim_id") or as a marker line `[verbatim <id>]` (first AND
  last line of plain text). The same env var name is used on both sides.
* transform_tool_result scans results of the terminal, process and execute_code tools for ids (results of
  read_file, session_search or web tools may echo an OLD id and are ignored), reads the staged text and keeps
  it, keyed by (session_id, turn_id). It never changes the result. An id is delivered at most once, and a text
  already held for the turn is not added twice.
* transform_llm_output: if the turn has staged texts, the model's final text is replaced by them, joined by a
  blank line. The entry is looked up by (session_id, turn_id), then by turn_id alone when exactly one entry has
  it (context compression can change session_id mid-turn). Otherwise, on Telegram only, em and en dashes are
  scrubbed. `[verbatim <id>]` marker lines are stripped from every final text.
* The core calls the hook with an empty reply too (agent/turn_finalizer.py, allow_empty), so a turn where the
  model says nothing still delivers the scripted text.
* pre_llm_call remembers the latest user message per (session, turn) and per session. pre_tool_call then blocks
  a terminal call `python3 .../coach_log.py add-note` unless its --text (two or more words) is a substring of
  that message.
* pre_tool_call also blocks edits of the coach skills (skill_manage, write_file, patch) for the interactive
  agent and for the background self-improvement review, which dispatches through the same hook.

Known limit of the skill lock: it does NOT inspect `terminal` or `execute_code`. Those tools also run the
skill's own scripts, so a command is not judged by the path it mentions. A shell edit (sed -i, tee, a redirect)
can still change the coach skill; changes to that skill are meant to go through Claude Code.

Everything fails open except the two blocks, which fail closed when a coach_log add-note call cannot be parsed.
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
_MARKER_LINE = re.compile(r"(?m)^[ \t]*\[verbatim %s\][ \t]*(?:\n|$)" % ID_PATTERN)
_MARKER_INLINE = re.compile(r"[ \t]?\[verbatim %s\]" % ID_PATTERN)

# Only these tools can carry a fresh id: they run the coach scripts. Other tools may echo an old one.
ID_TOOLS = {"terminal", "process", "process_manage", "execute_code"}

MAX_TEXT_BYTES = 200_000
MAX_FILE_AGE_S = 3600
PENDING_TTL_S = 30 * 60
PENDING_MAX = 256
USER_MSG_MAX = 256
DELIVERED_MAX = 500

NOTE_BLOCK_MESSAGE = "add-note saves Chin's own words only. Quote his message exactly, or do not save a note."
SKILL_LOCK_MESSAGE = "This skill is locked. Changes to it go through Claude Code."

LOCKED_SKILLS = {"health-wellness-coach", "body-composition-blocks"}
_LOCKED_TEXT_PATH = re.compile(
    r"(^|/)skills/(productivity/health-wellness-coach|health/body-composition-blocks)(/|$)"
)

_LOCK = threading.Lock()
# (session_id, turn_id) -> {"ts": float, "ids": [..], "texts": [..]}
_PENDING: "OrderedDict[tuple, dict]" = OrderedDict()
# (session_id, turn_id) -> latest user message of that turn; session_id -> latest user message of the session
_USER_BY_TURN: "OrderedDict[tuple, str]" = OrderedDict()
_USER_BY_SESSION: "OrderedDict[str, str]" = OrderedDict()
# ids already delivered (never delivered twice)
_DELIVERED: "OrderedDict[str, float]" = OrderedDict()


# ---------------------------------------------------------------------------
# Staged text
# ---------------------------------------------------------------------------

def _hermes_home() -> Path:
    try:
        from hermes_constants import get_hermes_home

        return Path(get_hermes_home())
    except Exception:  # noqa: BLE001 - the plugin must load even when the helper cannot
        return Path(os.environ.get("HERMES_HOME") or os.path.expanduser("~/.hermes"))


def verbatim_dir() -> Path:
    env = os.environ.get("HWC_VERBATIM_DIR")
    if env:
        return Path(env)
    return _hermes_home() / "data" / "op" / "verbatim"


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


def strip_markers(text: str) -> str:
    """Remove `[verbatim <id>]` marker lines (and inline markers); the markers are for the gateway only."""
    out = _MARKER_LINE.sub("", text)
    return _MARKER_INLINE.sub("", out)


def _prune(now: float) -> None:
    """Caller holds _LOCK. Entries that expire undelivered are logged at WARNING."""
    for key in [k for k, v in _PENDING.items() if now - v["ts"] > PENDING_TTL_S]:
        entry = _PENDING.pop(key, None)
        if entry:
            logger.warning(
                "verbatim-reply: %d scripted message(s) for session %r turn %r expired undelivered",
                len(entry["texts"]), key[0], key[1])
    while len(_PENDING) > PENDING_MAX:
        key, entry = _PENDING.popitem(last=False)
        logger.warning("verbatim-reply: dropped %d undelivered scripted message(s) for session %r turn %r (map full)",
                       len(entry["texts"]), key[0], key[1])


def _remember_delivered(ids: list, now: float) -> None:
    """Caller holds _LOCK."""
    for vid in ids:
        _DELIVERED[vid] = now
        _DELIVERED.move_to_end(vid)
    while len(_DELIVERED) > DELIVERED_MAX:
        _DELIVERED.popitem(last=False)


def _on_transform_tool_result(
    tool_name: str = "", args: Any = None, result: Any = None,
    session_id: str = "", turn_id: str = "", **_: Any,
) -> Optional[str]:
    """Collect staged texts named in this result. Never changes the result."""
    try:
        if tool_name not in ID_TOOLS:
            return None
        if not isinstance(result, str) or "verbatim" not in result:
            return None
        ids = find_ids(result)
        if not ids:
            return None
        key = (session_id or "", turn_id or "")
        now = time.time()
        for vid in ids:
            with _LOCK:
                _prune(now)
                if vid in _DELIVERED:
                    logger.info("verbatim-reply: id %s was already delivered, ignored", vid)
                    continue
                entry = _PENDING.get(key)
                if entry and vid in entry["ids"]:
                    continue
            text = read_staged(vid)
            if text is None:
                logger.info("verbatim-reply: no staged text for id %s (tool %s)", vid, tool_name)
                continue
            with _LOCK:
                entry = _PENDING.setdefault(key, {"ts": now, "ids": [], "texts": []})
                entry["ids"].append(vid)   # the id is spent for this turn even when the text repeats
                if text.strip() in [t.strip() for t in entry["texts"]]:
                    logger.info("verbatim-reply: id %s repeats a text already held for this turn, skipped", vid)
                    continue
                entry["texts"].append(text)
                entry["ts"] = now
    except Exception as exc:  # noqa: BLE001
        logger.debug("verbatim-reply transform_tool_result failed (non-fatal): %s", exc)
    return None


# ---------------------------------------------------------------------------
# Dash scrub (Telegram only, when there is no scripted text)
# ---------------------------------------------------------------------------

# Fenced code, inline code and URLs are left untouched.
_PROTECTED = re.compile(
    r"```.*?(?:```|\Z)|~~~.*?(?:~~~|\Z)|`{1,2}[^`\n]+`{1,2}|\S*://\S*|\bt\.me/\S*", re.DOTALL)
_DASH = "—–"
_MON = r"(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec)[a-z]*\.?"
_DATE = r"(?:\d{4}-\d{2}-\d{2}|\d{1,2}[ \t]+%s|%s[ \t]+\d{1,2})" % (_MON, _MON)


def _scrub_plain(s: str) -> str:
    s = re.sub(r"(\|[ \t]*)[%s]([ \t]*(?=\|))" % _DASH, r"\1-\2", s)                 # a table cell that is only a dash
    s = re.sub(r"(%s)[ \t]*[%s][ \t]*(?=%s)" % (_DATE, _DASH, _DATE), r"\1 to ", s)   # date to date
    s = re.sub(r"(%s[ \t]+\d{1,2})[ \t]*[%s][ \t]*(?=\d{1,2}\b)" % (_MON, _DASH), r"\1 to ", s)   # Oct 5 - 7
    s = re.sub(r"(?m)^([ \t]*)[%s][ \t]+" % _DASH, r"\1- ", s)          # a dash used as a bullet
    s = re.sub(r"(?m)[ \t]*[%s][ \t]*$" % _DASH, "", s)                 # a dash that ends a line
    s = re.sub(r"(?<=\d)[ \t]*[%s][ \t]*(?=\d)" % _DASH, "-", s)        # 5-7, 5 - 7
    s = re.sub(r"[ \t]+[%s][ \t]+" % _DASH, ", ", s)                    # " - " as a pause
    s = re.sub(r"[ \t]*[%s][ \t]*" % _DASH, ", ", s)                    # any other
    return re.sub(r",(?:[ \t]*,)+", ",", s)


def scrub_dashes(text: str) -> str:
    """Replace em and en dashes outside code and URLs. Protected spans are swapped for placeholders
    first, so the text around them is scrubbed as one string."""
    kept = []

    def hold(m):
        kept.append(m.group(0))
        return "\ue000%d\ue001" % (len(kept) - 1)

    out = _scrub_plain(_PROTECTED.sub(hold, text))
    return re.sub("\ue000(\\d+)\ue001", lambda m: kept[int(m.group(1))], out)


def _take_entry(session_id: str, turn_id: str) -> Optional[dict]:
    """Pop the turn's entry: exact (session, turn) first, then the single entry with that turn_id."""
    key = (session_id or "", turn_id or "")
    with _LOCK:
        _prune(time.time())
        entry = _PENDING.pop(key, None)
        if entry is not None:
            return entry
        if turn_id:
            same_turn = [k for k in _PENDING if k[1] == turn_id]
            if len(same_turn) == 1:
                logger.info("verbatim-reply: session id changed mid-turn (%r -> %r), using the turn_id match",
                            same_turn[0][0], key[0])
                return _PENDING.pop(same_turn[0])
        if _PENDING:
            logger.warning(
                "verbatim-reply: no scripted entry for session %r turn %r while %d other entr%s wait",
                key[0], key[1], len(_PENDING), "y" if len(_PENDING) == 1 else "ies")
    return None


def _on_transform_llm_output(
    response_text: str = "", session_id: str = "", model: str = "", platform: str = "",
    turn_id: str = "", **_: Any,
) -> Optional[str]:
    try:
        entry = _take_entry(session_id, turn_id)
        if entry and entry["texts"]:
            logger.info("verbatim-reply: replaced model text with %d scripted message(s)", len(entry["texts"]))
            with _LOCK:
                _remember_delivered(entry["ids"], time.time())
            return strip_markers("\n\n".join(entry["texts"])) or None
        if not isinstance(response_text, str) or not response_text:
            return None
        fixed = strip_markers(response_text)
        if (platform or "").lower() == "telegram":
            fixed = scrub_dashes(fixed)
        if fixed != response_text and fixed.strip():
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


def _on_pre_llm_call(session_id: str = "", turn_id: str = "", user_message: Any = None, **_: Any) -> None:
    try:
        text = _message_text(user_message)
        if not session_id or not text.strip():
            return None
        with _LOCK:
            _USER_BY_SESSION[session_id] = text
            _USER_BY_SESSION.move_to_end(session_id)
            while len(_USER_BY_SESSION) > USER_MSG_MAX:
                _USER_BY_SESSION.popitem(last=False)
            if turn_id:
                _USER_BY_TURN[(session_id, turn_id)] = text
                while len(_USER_BY_TURN) > USER_MSG_MAX:
                    _USER_BY_TURN.popitem(last=False)
    except Exception as exc:  # noqa: BLE001
        logger.debug("verbatim-reply pre_llm_call failed (non-fatal): %s", exc)
    return None


def _latest_user(session_id: str, turn_id: str) -> str:
    """The user message for this turn; else the session's latest; else the one for the same turn_id."""
    with _LOCK:
        if turn_id and (session_id or "", turn_id) in _USER_BY_TURN:
            return _USER_BY_TURN[(session_id or "", turn_id)]
        if session_id in _USER_BY_SESSION:
            return _USER_BY_SESSION[session_id]
        if turn_id:
            same = [v for (s, t), v in _USER_BY_TURN.items() if t == turn_id]
            if len(same) == 1:
                return same[0]
    return ""


def normalize(s: str) -> str:
    s = re.sub(r"[^\w\s]|_", "", str(s).lower())
    return " ".join(s.split())


_OPERATORS = {";", "&", "&&", "|", "||", "(", ")", ";;", "|&"}
_NOTE_CALL = re.compile(r"\bpython3?\b\s+(?:-\S+\s+)*\S*coach_log\.py\b\s+(?:\S+\s+)*?add-note\b")
_NOTE_HINT = re.compile(r"\bcoach_log\.py\b.*\badd-note\b", re.DOTALL)


def _is_script_token(t: str) -> bool:
    return t.endswith("coach_log.py")


def _add_note_texts(command: str) -> Optional[list]:
    """The --text values of every `python[3] .../coach_log.py add-note` call in `command`.
    [] when the command has none; None when it has one that cannot be parsed."""
    if not _NOTE_HINT.search(command):
        return []
    try:
        lex = shlex.shlex(command, posix=True, punctuation_chars=True)
        lex.whitespace_split = True
        lex.commenters = ""
        tokens = list(lex)
    except ValueError:
        return None if _NOTE_CALL.search(command) else []
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
        call = None
        for i, t in enumerate(seg):
            if os.path.basename(t).startswith("python"):
                j = i + 1
                while j < len(seg) and seg[j].startswith("-"):
                    j += 1
                if j < len(seg) and _is_script_token(seg[j]):
                    call = seg[j + 1:]
                    break
        if call is None or "add-note" not in call:
            continue
        found = None
        for i, t in enumerate(call):
            if t == "--text":
                if i + 1 >= len(call):
                    return None
                found = call[i + 1]
            elif t.startswith("--text="):
                found = t[len("--text="):]
        if found is None:
            return None
        texts.append(found)
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
        if len(n.split()) < 2 or n not in said:
            return NOTE_BLOCK_MESSAGE
    return None


# ---------------------------------------------------------------------------
# Skill lock
# ---------------------------------------------------------------------------

def _locked_roots() -> list:
    home = str(_hermes_home())
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
    tool_name: str = "", args: Any = None, session_id: str = "", turn_id: str = "", **_: Any,
) -> Optional[dict]:
    try:
        if is_locked_call(tool_name, args):
            logger.info("verbatim-reply: blocked %s on a locked skill", tool_name)
            return {"action": "block", "message": SKILL_LOCK_MESSAGE}
    except Exception as exc:  # noqa: BLE001
        logger.debug("verbatim-reply skill lock check failed: %s", exc)
    try:
        if tool_name == "terminal" and isinstance(args, dict) and isinstance(args.get("command"), str):
            cmd = args["command"]
            if _NOTE_HINT.search(cmd):
                msg = check_add_note(cmd, _latest_user(session_id or "", turn_id or ""))
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
