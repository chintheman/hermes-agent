"""Spine — Hermes Memory Provider v2 (plugins/memory/spine).

Implements the MemoryProvider ABC with canonical JSONL logs, a derived
SQLite FTS5+vec index, and four agent tools (remember, recall, reflect, forget).
Three loops (observer, consolidation, activation manifest) run via plugin hooks.

Spec: memory-system-v2.1.0-spec.md
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import threading
import time
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


def _scope_filter_enabled() -> bool:
    """Read memory.scope_filter from the live Hermes config.

    Defaults to False. Both halves of scope filtering must read the same flag
    at the same moment, or the feature is half-on: the snapshot filtered with
    nothing delivering the rest, or context delivered on top of a snapshot that
    already carries it.

    This used to hold the value in a process-global set on the first prefetch
    of the process's life and never re-read. Half 1 (MemoryStore in
    tools/memory_tool.py) reads the flag afresh for every agent the gateway
    builds, so turning the flag on against a RUNNING gateway filtered the
    snapshot while prefetch() went on short-circuiting to "" on a stale False.
    The deferred rules were then in no prompt at all. Measured live on
    2026-09-05: gateway up at 11:10:57, a cron turn at 11:15:26 cached False,
    the flag went on at 11:34:38, and the 11:34:45 turn lost all 27 deferred
    blocks. Three separate live tests failed this way and read as "prefetch is
    broken".

    load_config_readonly() is the house loader: it caches on the config file's
    (mtime_ns, size), so an edit invalidates it, and it resolves HERMES_HOME,
    where the hardcoded ~/.hermes/config.yaml read the wrong file under a
    profile. Cache-hit cost is ~130us, against ~30ms of selection work behind
    it.
    """
    try:
        from hermes_cli.config import cfg_get, load_config_readonly

        return bool(
            cfg_get(load_config_readonly(), "memory", "scope_filter", default=False)
        )
    except Exception:
        return False


def _root_spine_config() -> bool | None:
    """observer_enabled from the INSTALL's own config, whatever scope is bound.

    Path.home()/.hermes/config.yaml, deliberately not get_hermes_home(): the entire
    point is to reach a config the bound profile scope cannot see.
    """
    try:
        import yaml  # type: ignore
        from pathlib import Path

        data = yaml.safe_load((Path.home() / ".hermes" / "config.yaml").read_text()) or {}
        spine = ((data.get("memory") or {}) if isinstance(data, dict) else {}).get("spine") or {}
        if isinstance(spine, dict) and "observer_enabled" in spine:
            return bool(spine["observer_enabled"])
    except Exception as exc:
        # Returning None here lets the caller default to ON, so a broken root config
        # must leave a trace — this is the "root says off, pass runs anyway" case.
        logger.warning(
            "Spine observer: root config unreadable (%s: %s) — switch falls back to ON",
            type(exc).__name__, exc,
        )
    return None


def _observer_enabled() -> bool:
    """Read memory.spine.observer_enabled from the live Hermes config.

    Defaults to True — this flag exists so the pass can be turned OFF, and it is
    read fresh each session end for the same reason _scope_filter_enabled() is
    (a stale process-global read makes a flag look broken).

    Why the opt-out exists (measured 2026-10-04, ~/.hermes/memory.db):
    7,476 observations written, 28 ever retrieved (0.4% read rate), 5,350 rows
    already superseded or demoted, and 7 reached MEMORY.md — a 0.33% yield. The
    pass spends one LLM call per session end to feed a store the agent does not
    read. Set memory.spine.observer_enabled: false to stop that; unset or true
    restores the previous behaviour exactly.

    PROFILE FALLBACK, added 2026-10-04 after review. `load_config_readonly()`
    resolves through whatever HERMES_HOME is bound to, and a profile home carries
    no `memory.spine` block — so under a profile-scoped session end this read
    returned the *default* True and the switch quietly did nothing. Measured live:
    HERMES_HOME=profiles/librarian -> observer_enabled True, while the root config
    says false. The store this gate protects is the single root-owned
    ~/.hermes/memory.db that every profile shares, so the switch governing it cannot
    be profile-local: a profile that sets the key itself still wins (its value is
    read first), otherwise the root config decides. This is deliberately NOT general
    config inheritance — it is one flag, for one shared resource.
    """
    try:
        from hermes_cli.config import cfg_get, load_config_readonly

        cfg = load_config_readonly()
        spine = cfg_get(cfg, "memory", "spine", default=None)
        if isinstance(spine, dict) and "observer_enabled" in spine:
            return bool(spine["observer_enabled"])
        root = _root_spine_config()
        if root is not None:
            return root
        return True
    except Exception as exc:
        # Fail OPEN — preserve prior behaviour — but never SILENTLY. A read that raises
        # is otherwise indistinguishable from a config that says "on", and the pass's
        # only output is the absence of a message. Measured 2026-10-04: the switch read
        # off for 25 minutes while the observer wrote 24 observations and 2 episodes,
        # and nothing anywhere said the setting had not taken effect.
        logger.warning(
            "Spine observer switch unreadable (%s: %s) — defaulting to ON, so the pass "
            "WILL run. Fix memory.spine.observer_enabled or the config loader.",
            type(exc).__name__, exc,
        )
        return True


_warm_started = False
_warm_lock = threading.Lock()


def _warm_models_once(config) -> None:
    """Load the embedder and (if enabled) the re-ranker in a background thread, once
    per process.

    Measured 2026-10-07: the first recall after a gateway restart took 20.9s while
    both models loaded on demand; the next took 0.55s. Warming at the first session
    init moves that cost off the user's first recall. Daemon thread, never blocks
    init, never raises: a failed warm just leaves the on-demand load in place.
    """
    global _warm_started
    with _warm_lock:
        if _warm_started:
            return
        _warm_started = True

    def _run() -> None:
        t0 = time.monotonic()
        try:
            from .embedder import embedder_available
            embedder_available()
            if getattr(config, "rerank_pool", 0) > 0:
                from .reranker import _load
                _load(config.rerank_model)
            logger.info("Spine models warmed in %.1fs", time.monotonic() - t0)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Spine model warm-up failed (%s: %s); loading on demand",
                           type(exc).__name__, exc)

    threading.Thread(target=_run, name="spine-warm", daemon=True).start()


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
        _warm_models_once(self._config)

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
        """Rules this turn needs that the frozen snapshot does not already hold.

        Computed inline from THIS turn's query, deliberately not from a cache.

        Two ordering bugs live here if you cache. queue_prefetch() runs AFTER a
        turn to prepare the NEXT one, so the first turn of a session has nothing
        cached — and a one-shot run (cron, hermes-run, most headless work) is
        always a first turn. Observed live on 2026-09-05: the snapshot dropped 27
        blocks and prefetch delivered none of them. Then on turn 2+ a cache holds
        the selection for the PREVIOUS query, so it would deliver rules for the
        topic before last.

        Inline is safe here in a way it would not be for the ABC's usual case.
        Selection is pure string matching over a ~24KB file — no network, no
        embedding, no model call, ~30ms. The "be fast, use the cache" guidance
        exists to keep recall off the critical path, not to forbid that.
        """
        try:
            if not _scope_filter_enabled():
                return ""
            return self._build_scoped_context(query)
        except Exception as exc:  # noqa: BLE001
            logger.warning("rule_scope prefetch failed (non-fatal): %s", exc)
            return ""

    def queue_prefetch(self, query: str, *, session_id: str = "") -> None:
        """Prepare context for the NEXT turn, off the critical path.

        SHADOW MODE: computes the selection and logs what it WOULD have
        deferred. Delivery itself lives in prefetch(), which computes from the
        current turn's query.

        The log only says something once blocks carry @when: markers — until
        then every block is universal and deferred is 0 by design, which is
        itself the fail-safe worth confirming in production.
        """
        try:
            self._shadow_record(query)
        except Exception as exc:  # never let shadow logging break a turn
            logger.debug("rule_scope shadow failed (non-fatal): %s", exc)

        # Deliberately does NOT prime a cache for prefetch(): the selection
        # depends on the NEXT turn's query, which is not knowable here.
        return None

    def _build_scoped_context(self, query: str, tool_name: str = "",
                             has_image: bool = False) -> str:
        """Rules this turn needs that the frozen snapshot does not already hold."""
        from .rule_scope import TurnContext, deliverable, split_blocks

        hotcore = os.path.expanduser("~/.hermes/memories/MEMORY.md")
        if not os.path.exists(hotcore):
            return ""
        with open(hotcore, encoding="utf-8", errors="ignore") as fh:
            blocks = split_blocks(fh.read())
        extra = deliverable(blocks, TurnContext(query=query or "",
                                                tool_name=tool_name,
                                                has_image=has_image))
        if not extra:
            return ""
        body = "\n".join(f"- {b}" for b in extra)
        return ("[MEMORY — rules relevant to this turn]\n" + body)

    # Shadow-mode instrumentation ---------------------------------------

    SHADOW_LOG = os.path.expanduser("~/.hermes/state/rule-scope-shadow.jsonl")

    def _shadow_record(self, query: str, tool_name: str = "",
                       has_image: bool = False) -> None:
        """Append one line describing what scoping would have done this turn."""
        from .rule_scope import TurnContext, split_blocks, summarise

        hotcore = os.path.expanduser("~/.hermes/memories/MEMORY.md")
        if not os.path.exists(hotcore):
            return
        with open(hotcore, encoding="utf-8", errors="ignore") as fh:
            blocks = split_blocks(fh.read())

        rec = summarise(blocks, TurnContext(query=query or "",
                                            tool_name=tool_name,
                                            has_image=has_image))
        rec["ts"] = time.strftime("%Y-%m-%dT%H:%M:%S")
        rec["session"] = getattr(self, "_session_id", "")
        os.makedirs(os.path.dirname(self.SHADOW_LOG), exist_ok=True)
        with open(self.SHADOW_LOG, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(rec) + "\n")

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

    def on_session_end(self, messages: List[Dict[str, Any]]) -> None:
        """Observer pass — extract durable observations (spec §5.1).

        Gated by memory.spine.observer_enabled (default True, so this is a no-op
        unless the flag is explicitly turned off). See _observer_enabled() for the
        2026-10-04 measurements that motivated the opt-out.
        """
        if not _observer_enabled():
            logger.info("Spine observer disabled by config — session end extracted nothing")
            return
        from .loops import run_observer

        # The pass's own heartbeat, logged whether or not it finds anything durable.
        # Added 2026-10-04 after review: the only line written was "wrote episode", so a
        # pass that ran and produced nothing — a quiet session, or a broken extraction —
        # left no trace at all, and the audit's switch check could not tell "asked and
        # declined" from "never asked". One INFO per session end is the cheapest thing
        # that makes the gate observable.
        logger.info("Spine observer gate open — session end reached the pass")
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
