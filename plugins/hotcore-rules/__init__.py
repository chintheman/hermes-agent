"""Deliver tool-scoped hot-core rules at the moment their tool runs.

Part of hotcore-retrieval-split. MEMORY.md blocks carrying `@when:tool=<glob>`
are filtered out of the frozen system-prompt snapshot; this hook is what brings
them back, appended to the result of the tool they are about. A rule like
"Garmin restarts strength activity per block — re-query for the new ID" is then
paid for on Garmin calls and on no other turn.

Why a separate plugin rather than part of spine: `plugins/memory/` is excluded
from hook discovery (`skip_names` in hermes_cli/plugins.py), so a memory
provider cannot register a `transform_tool_result` hook. The selection logic is
imported from spine.rule_scope rather than duplicated.

Two things this deliberately does NOT do:

* **No process-global flag cache.** The config is read through the house loader
  on every call, which caches on the config file's mtime. A hand-rolled global
  in the sibling code memoised `False` at gateway start and never re-read it,
  which silently emptied the hot core of 27 blocks for three live tests on
  2026-09-05.
* **No process-global "already sent" set.** Dedupe is keyed by session_id, which
  the dispatcher passes. A global keyed on tool name alone would deliver to the
  first session after a restart and to nobody afterwards.

Fails open in both directions: any error returns None, leaving the tool result
untouched, and if the flag is off the rules are already in the snapshot so this
stays silent.
"""

from __future__ import annotations

import logging
import os
from collections import OrderedDict
from typing import Any, Optional

logger = logging.getLogger(__name__)

HOTCORE = os.path.expanduser("~/.hermes/memories/MEMORY.md")

# (session_id, tool_name) pairs already served this process. Bounded, because a
# gateway runs for weeks; oldest entries are evicted rather than growing without
# limit. Keyed by session so a new conversation always gets its rules.
_SENT: "OrderedDict[tuple, bool]" = OrderedDict()
_SENT_MAX = 512


def _scope_filter_enabled() -> bool:
    """Read memory.scope_filter live. See the module docstring on caching."""
    try:
        from hermes_cli.config import cfg_get, load_config_readonly

        return bool(
            cfg_get(load_config_readonly(), "memory", "scope_filter", default=False)
        )
    except Exception:
        return False


def _already_sent(session_id: str, tool_name: str) -> bool:
    key = (session_id or "", tool_name or "")
    if key in _SENT:
        return True
    _SENT[key] = True
    while len(_SENT) > _SENT_MAX:
        _SENT.popitem(last=False)
    return False


def _rules_for_tool(tool_name: str) -> list:
    """Hot-core blocks this tool triggers that the snapshot no longer carries."""
    import sys

    plugins_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    memory_dir = os.path.join(plugins_dir, "memory")
    if memory_dir not in sys.path:
        sys.path.insert(0, memory_dir)
    from spine.rule_scope import TurnContext, deliverable, split_blocks

    if not os.path.exists(HOTCORE):
        return []
    with open(HOTCORE, encoding="utf-8", errors="ignore") as fh:
        blocks = split_blocks(fh.read())
    return deliverable(blocks, TurnContext(tool_name=tool_name))


def _on_transform_tool_result(
    tool_name: str = "",
    args: Any = None,
    result: Any = None,
    session_id: str = "",
    **_: Any,
) -> Optional[str]:
    """Append this tool's rules to its result. None leaves the result alone."""
    try:
        if not tool_name or not isinstance(result, str):
            return None
        if not _scope_filter_enabled():
            return None
        rules = _rules_for_tool(tool_name)
        if not rules:
            return None
        if _already_sent(session_id, tool_name):
            return None
        body = "\n".join(f"- {r}" for r in rules)
        logger.info(
            "hotcore-rules: delivered %d rule(s) with the %s result",
            len(rules), tool_name,
        )
        return (
            f"{result}\n\n[MEMORY — rules about {tool_name}]\n{body}"
        )
    except Exception as exc:  # noqa: BLE001
        logger.debug("hotcore-rules hook failed (non-fatal): %s", exc)
        return None


def register(ctx) -> None:
    ctx.register_hook("transform_tool_result", _on_transform_tool_result)
