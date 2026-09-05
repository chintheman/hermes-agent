"""Spine — Hermes Memory Provider v2 (plugins/memory/spine).

Implements the MemoryProvider ABC with canonical JSONL logs, a derived
SQLite FTS5+vec index, and four agent tools (remember, recall, reflect, forget).
Three loops (observer, consolidation, activation manifest) run via plugin hooks.

Spec: memory-system-v2.1.0-spec.md
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Dict, List, Optional

from agent.memory_provider import MemoryProvider

logger = logging.getLogger(__name__)

__version__ = "2.1.0"


def _send_secrets_alert(target: str, text: str) -> None:
    """Best-effort real Telegram send for a secrets-block alert.

    Previously this path only did `print(...)`, which looked like a real
    notification but just wrote to whatever process's stdout happened to be
    calling this hook — the user would never actually see it. `target` is
    "telegram:<chat_id>[:<thread_id>]".

    This hook fires synchronously from inside the live agent's call chain,
    which may already be running inside an event loop — schedule on it if so,
    otherwise run one directly.
    """
    parts = target.split(":")
    if len(parts) < 2 or parts[0] != "telegram":
        logger.warning("Unrecognized secrets-alert target %r — not sent", target)
        return
    chat_id, thread_id = parts[1], (parts[2] if len(parts) > 2 else None)

    import hermes_cli.gateway as gateway_mod
    from tools.send_message_tool import _send_telegram

    token = gateway_mod.get_env_value("TELEGRAM_BOT_TOKEN") or ""
    if not token:
        logger.warning("No TELEGRAM_BOT_TOKEN available — secrets alert not sent")
        return

    async def _send():
        await _send_telegram(token, chat_id, text, thread_id=thread_id)

    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None
    if loop is not None:
        loop.create_task(_send())
    else:
        asyncio.run(_send())


class SpineProvider(MemoryProvider):
    """Hermes Memory v2 — "The Sleeping Brain" provider."""

    # ------------------------------------------------------------------
    # MemoryProvider ABC
    # ------------------------------------------------------------------

    @property
    def name(self) -> str:
        return "spine"

    def is_available(self) -> bool:
        """Spine is local-only — always available if config is present."""
        return True

    def initialize(self, session_id: str, **kwargs) -> None:
        """Read config, open index, warm embedder if needed."""
        from .config import load_spine_config

        self._session_id = session_id
        self._config = load_spine_config(kwargs.get("hermes_home", ""))
        logger.info("Spine initialized — session=%s", session_id)

    def system_prompt_block(self) -> str:
        """Spine contributes no static system prompt text."""
        return ""

    # ------------------------------------------------------------------
    # Per-turn context injection (MemoryProvider ABC)
    #
    # prefetch() is called before every API call and its return value is
    # injected into the API copy of this turn's user message. queue_prefetch()
    # is called after a turn completes, so the work happens off the critical
    # path and prefetch() only reads a cached string — the contract the ABC
    # docstring asks for and the pattern hindsight already uses.
    #
    # PHASE 1: deliberately returns "". The pipe is being proven before
    # anything flows through it. Rule selection lands in phase 2.
    # ------------------------------------------------------------------

    def prefetch(self, query: str, *, session_id: str = "") -> str:
        """Return context to inject for the upcoming turn.

        Must be fast: read the cache, do not recall here.
        """
        cache = getattr(self, "_prefetch_cache", "")
        # One-shot: injected context is stamped into api_content and replayed on
        # later turns, so returning the same string every turn would duplicate
        # it rather than refresh it.
        self._prefetch_cache = ""
        return cache

    def queue_prefetch(self, query: str, *, session_id: str = "") -> None:
        """Prepare context for the NEXT turn, off the critical path."""
        # Phase 1 no-op. Phase 2 fills _prefetch_cache with the subject-triggered
        # rules selected by rule_scope.select().
        return None

    def get_tool_schemas(self) -> List[Dict[str, Any]]:
        """Return the spine tool schemas."""
        from .tools import REMEMBER_SCHEMA, RECALL_SCHEMA, RECALL_AT_SCHEMA, REFLECT_SCHEMA, FORGET_SCHEMA, EXPLAIN_SCHEMA

        return [REMEMBER_SCHEMA, RECALL_SCHEMA, RECALL_AT_SCHEMA, REFLECT_SCHEMA, FORGET_SCHEMA, EXPLAIN_SCHEMA]

    def handle_tool_call(self, tool_name: str, args: Dict[str, Any], **kwargs) -> str:
        """Dispatch tool call to the appropriate handler."""
        from .tools import handle_remember, handle_recall, handle_recall_at, handle_reflect, handle_forget, handle_explain

        handlers = {
            "remember": handle_remember,
            "recall": handle_recall,
            "recall_at": handle_recall_at,
            "reflect": handle_reflect,
            "forget": handle_forget,
            "explain": handle_explain,
        }
        handler = handlers.get(tool_name)
        if handler is None:
            return '{"error": "Unknown tool: ' + tool_name + '"}'
        return handler(args, config=self._config)

    def shutdown(self) -> None:
        """Unload embedder, close index."""
        from .embedder import unload_embedder

        unload_embedder()
        logger.info("Spine shutdown complete.")

    # ------------------------------------------------------------------
    # Optional hooks
    # ------------------------------------------------------------------

    def on_turn_start(self, turn_number: int, message: str, **kwargs) -> None:
        """Per-turn bookkeeping. Injection happens in prefetch(), not here.

        2026-09-05: this method used to build an "activation manifest" and store
        it on self._manifest_text, which NOTHING ever read — two writes, zero
        reads, confirmed by grep. system_prompt_block() returns "" and this hook
        returns None, so spine had no route to inject anything at all. Every
        session since it shipped built that manifest and threw it away, and its
        rule_checklist was four hard-coded rules matching none of the ten real
        [R] rules in the hot core. Dead and wrong, so both are gone.

        The ABC's actual per-turn injection path is prefetch(), which returns a
        string. That is implemented below.
        """
        self._turn_number = turn_number

    def on_session_reset(self, **kwargs) -> None:
        """Clear per-session caches on /reset."""
        self._turn_number = 0
        self._prefetch_cache = ""

    def on_session_end(self, messages: List[Dict[str, Any]]) -> None:
        """Observer pass — extract durable observations (spec §5.1)."""
        from .loops import run_observer

        run_observer(messages, config=self._config)

    def on_memory_write(
        self,
        action: str,
        target: str,
        content: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Post-hoc secrets scan on builtin memory writes (HANDOFF-REVIEW P1-6).

        Builtin memory() bypasses spine's remember() gate. This hook scans
        every builtin write and strips secrets from MEMORY.md if found.
        Not ideal (secret lands briefly), but the safest plug-compatible fix.
        """
        from .secrets_detector import detect_secrets
        from .config import SpineConfig

        result = detect_secrets(content)
        if result.get("blocked"):
            logger.warning("Builtin memory write blocked by secrets detector: %s", result.get("reason"))
            # Strip the line from MEMORY.md — match the line's actual content
            # exactly (lstrip("-*") only matters for bullet-style entries;
            # verified 2026-07-30 that real MEMORY.md is actually §-separated
            # free-text paragraphs, not bullets — harmless no-op against that
            # format, kept for any bullet-style entries that do occur), not
            # "any line containing this as a substring anywhere" — the old
            # substring check could delete unrelated lines that merely
            # happened to contain the same short fragment (e.g. a recurring
            # URL) elsewhere in the file.
            import os as _os
            mem_path = _os.path.expanduser("~/.hermes/memories/MEMORY.md")
            if _os.path.exists(mem_path):
                with open(mem_path, "r", encoding="utf-8") as f:
                    lines = f.readlines()
                target_content = content.strip()
                cleaned = [
                    l for l in lines
                    if l.strip().lstrip("-*").strip() != target_content
                ]
                with open(mem_path, "w", encoding="utf-8") as f:
                    f.writelines(cleaned)
                logger.info("Stripped secret-bearing content from MEMORY.md")
            # Alert — actually send it, don't just log locally
            from .config import load_spine_config
            target_ch = "telegram:-1004389869012:11"
            try:
                cfg = load_spine_config()
                target_ch = getattr(cfg, "report_target", target_ch)
                _send_secrets_alert(
                    target_ch,
                    f"🔒 Secrets alert: builtin memory write blocked — {result.get('reason')}",
                )
            except Exception:
                logger.warning("Failed to send secrets alert to %s", target_ch)
