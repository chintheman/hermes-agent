"""Deterministic pre-send formatter: collapses every labelled MINE / YOURS ownership block in an
outbound message into ONE block at the very end.

Chin, 2026-09-27: "All this mine your mine yours and repeated outputs is killing me... I need 1
summary. 1 yours and mine. How can we ensure that is baked into you and Hermes deterministically?"
Prompting the model (``SKIM_MASTER_RULES_GUIDANCE`` in ``agent/prompt_builder.py``) has a non-zero
failure rate; this module is the deterministic backstop applied to every outbound message text,
regardless of what the model actually produced.

The rule lives in ONE spec file shared with Claude Code's own Stop hook
(``~/.hermes/hooks/one-ownership-block.py``): ``~/wiki/_system/ownership-format.json``. Loaded at
call time and cached by mtime so an edit to the spec takes effect on the next send without a
restart; falls back to built-in defaults (identical to the spec's current contents) when the file
is missing or unparseable.

Hooked into ``gateway/platforms/base.py::BasePlatformAdapter.__init__`` — see
``_install_ownership_formatter`` there — which wraps the concrete subclass's ``send()`` and
``edit_message()`` once per adapter instance. That is the single choke point: every platform
adapter (~30 of them) implements its own ``send``, so wrapping the bound method after subclass
construction covers all of them without touching a single adapter file.

Never fails a send: any internal error is caught and logged, and the ORIGINAL text is returned
unchanged.
"""

from __future__ import annotations

import json
import logging
import re
import threading
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

_SPEC_PATH = Path.home() / "wiki" / "_system" / "ownership-format.json"

# Built-in fallback — kept byte-identical in shape to ~/wiki/_system/ownership-format.json so a
# missing/unparseable spec file degrades to the same behavior, not a different one.
_DEFAULT_SPEC: Dict[str, Any] = {
    "labels": {"mine": "MINE", "yours": "YOURS"},
    "label_pattern": r"^\s*[#>\-*\s]*\**\s*{LABEL}\s*\**\s*:",
    "rules": {
        "max_blocks_per_turn": 1,
        "block_position": "end",
        "single_label_allowed": True,
        "drop_empty_labels": True,
        "force_full_resend": False,
        "correction_scope": "block_only",
    },
    "empty_values": [
        "nothing", "nothing.", "none", "none.", "n/a",
        "nothing running", "nothing running.",
        "nothing pending", "nothing pending.",
        "nothing yet", "nothing for now", "nothing for now.",
    ],
}

_CODE_FENCE_RE = re.compile(r"^\s*(```|~~~)")

_spec_lock = threading.Lock()
_spec_cache: Dict[str, Any] = {"mtime": None, "spec": None}


def _load_spec() -> Dict[str, Any]:
    """Read ``~/wiki/_system/ownership-format.json``, cached by mtime. Falls back to
    ``_DEFAULT_SPEC`` (never raises) when the file is missing, unreadable or malformed."""
    try:
        mtime = _SPEC_PATH.stat().st_mtime
    except OSError:
        return _DEFAULT_SPEC
    with _spec_lock:
        if _spec_cache["spec"] is not None and _spec_cache["mtime"] == mtime:
            return _spec_cache["spec"]
    try:
        data = json.loads(_SPEC_PATH.read_text(encoding="utf-8"))
        if not isinstance(data, dict) or not isinstance(data.get("labels"), dict):
            raise ValueError("ownership-format.json missing a 'labels' object")
        # Merge over defaults so a partial/older spec file still has every key this module reads.
        merged = {**_DEFAULT_SPEC, **data}
        merged["labels"] = {**_DEFAULT_SPEC["labels"], **data.get("labels", {})}
        merged["rules"] = {**_DEFAULT_SPEC["rules"], **data.get("rules", {})}
        if not isinstance(data.get("empty_values"), list):
            merged["empty_values"] = _DEFAULT_SPEC["empty_values"]
    except Exception:
        logger.warning("ownership-format.json unreadable/invalid at %s; using built-in defaults",
                       _SPEC_PATH, exc_info=True)
        return _DEFAULT_SPEC
    with _spec_lock:
        _spec_cache["mtime"], _spec_cache["spec"] = mtime, merged
    return merged


def _compile_label_pattern(pattern: str, label: str) -> re.Pattern:
    return re.compile(pattern.replace("{LABEL}", re.escape(label)))


def _protected_line_flags(lines: list) -> list:
    """One bool per line: True while inside a fenced code block (``` or ~~~), inclusive of the
    fence lines themselves. A stray unclosed fence protects everything after it — the safer
    failure mode for "never touch labels inside code"."""
    flags = []
    in_fence = False
    for line in lines:
        is_fence_line = bool(_CODE_FENCE_RE.match(line))
        if is_fence_line and not in_fence:
            in_fence = True
            flags.append(True)
            continue
        if is_fence_line and in_fence:
            flags.append(True)
            in_fence = False
            continue
        flags.append(in_fence)
    return flags


_CLOSING_MARKUP_RE = re.compile(r"^([*_]+)(\s|$)")


def _extract_label_value(rest_of_line: str) -> str:
    """The value after a matched label prefix, trimmed. ``label_pattern`` spans everything up to
    and including the colon, so a bold/italic label written as ``**MINE:** fixed it`` — the exact
    canonical form in Hermes's own Skim Master Rules guidance prompt — leaves the CLOSING ``**``
    stuck to the front of the value (the pattern only accounts for markers *before* the colon).
    Strip a marker run immediately glued to the colon (no space before it) with no accompanying
    space, so it doesn't get mistaken for an intentional leading ``*italic*`` value, which always
    has a space before it."""
    raw = rest_of_line.rstrip("\r\n")
    m = _CLOSING_MARKUP_RE.match(raw)
    return (raw[m.end(1):] if m else raw).strip()


def _is_empty_value(value: str, empty_values: list) -> bool:
    v = value.strip().lower().rstrip(".")
    return any(v == str(ev).strip().lower().rstrip(".") for ev in empty_values)


def format_ownership_block(text: str, spec: Optional[Dict[str, Any]] = None) -> str:
    """Collapse every labelled MINE/YOURS line in ``text`` into ONE block at the end, per
    ``spec`` (loaded from the shared spec file when omitted). Never raises: any internal error
    returns ``text`` unchanged, logged.
    """
    if not text or not isinstance(text, str):
        return text
    try:
        return _format(text, spec or _load_spec())
    except Exception:
        logger.error("ownership formatter failed; sending original text unchanged", exc_info=True)
        return text


def _format(text: str, spec: Dict[str, Any]) -> str:
    labels: Dict[str, str] = spec.get("labels") or _DEFAULT_SPEC["labels"]
    pattern = spec.get("label_pattern") or _DEFAULT_SPEC["label_pattern"]
    empty_values = spec.get("empty_values") or _DEFAULT_SPEC["empty_values"]
    mine_label, yours_label = labels.get("mine", "MINE"), labels.get("yours", "YOURS")
    mine_re = _compile_label_pattern(pattern, mine_label)
    yours_re = _compile_label_pattern(pattern, yours_label)

    lines = text.splitlines(keepends=True)
    protected = _protected_line_flags(lines)

    mine_values: list = []
    yours_values: list = []
    keep_mask = [True] * len(lines)

    for i, line in enumerate(lines):
        if protected[i]:
            continue
        m = mine_re.match(line)
        y = None if m else yours_re.match(line)
        match = m or y
        if not match:
            continue
        value = _extract_label_value(line[match.end():])
        (mine_values if m else yours_values).append(value)
        keep_mask[i] = False

    if not mine_values and not yours_values:
        return text  # nothing labelled — leave byte-identical

    # A removed label line sitting alone between two blank lines (the common shape: blank line,
    # "MINE: ...", blank line) leaves a doubled blank line behind. Consume one adjacent blank line
    # at each removal site — surgical, not a global collapse, so prose/code elsewhere that never
    # touched a label line stays byte-identical.
    def _is_blank_and_kept(idx: int) -> bool:
        return 0 <= idx < len(lines) and keep_mask[idx] and lines[idx].strip("\r\n") == ""

    for i in range(len(lines)):
        if keep_mask[i]:
            continue
        if _is_blank_and_kept(i - 1) and _is_blank_and_kept(i + 1):
            keep_mask[i + 1] = False

    def _merge(values: list) -> str:
        seen, merged = set(), []
        for v in values:
            key = v.strip()
            if key in seen:
                continue
            seen.add(key)
            merged.append(v.strip())
        return "; ".join(merged)

    mine_merged = _merge(mine_values) if mine_values else ""
    yours_merged = _merge(yours_values) if yours_values else ""

    drop_empty = bool((spec.get("rules") or {}).get("drop_empty_labels", True))
    if drop_empty:
        if mine_merged and _is_empty_value(mine_merged, empty_values):
            mine_merged = ""
        if yours_merged and _is_empty_value(yours_merged, empty_values):
            yours_merged = ""

    body = "".join(line for line, keep in zip(lines, keep_mask) if keep)
    body = body.strip("\n").rstrip()

    block_lines = []
    if mine_merged:
        block_lines.append(f"{mine_label}: {mine_merged}")
    if yours_merged:
        block_lines.append(f"{yours_label}: {yours_merged}")

    if not block_lines:
        return body

    block = "\n".join(block_lines)
    return f"{body}\n\n{block}" if body else block
