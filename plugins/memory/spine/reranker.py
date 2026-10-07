"""Second-pass re-ranker for spine recall.

Hybrid search (FTS5 + vectors, fused by reciprocal rank) scores each channel
independently. A document that is a strong keyword match but a middling vector
match gets credit from one channel only, and loses to documents that are
mediocre in both. Measured 2026-10-07: the answer to eval case
mh-memory-sync-latency was keyword rank 1, vector rank 214, fused rank 33, so
recall (k=25) missed it. Neither recency, pool size, nor a channel weight fixed
that robustly: weighting keyword 1.25x passed it, 1.5x broke two other cases.

A cross-encoder reads the query and each candidate TOGETHER, so it judges the
actual answer rather than two separate similarity proxies. It re-orders the
top `pool` hybrid candidates; the final order blends the hybrid rank with the
cross-encoder rank (same RRF k=60), so recency and entity boosts still count.
Measured on copies of the store: eval 29/29 (was 28/29) on both the current and
the pre-prune store, ~0.7s per query at pool 50 on the M1. Pool 30 is too small
(the answer sat at 33).

Fails OPEN: if the model cannot load or predict, recall returns the hybrid order
unchanged and logs a warning once.
"""
from __future__ import annotations

import logging
import threading
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
_RRF_K = 61          # matches search_hybrid's fusion constant
_MAX_CHARS = 2000    # the model truncates at 512 tokens anyway

_model: Any = None
_model_name: Optional[str] = None
_failed = False
_lock = threading.Lock()


def _load(name: str) -> Any:
    global _model, _model_name, _failed
    if _model is not None and _model_name == name:
        return _model
    if _failed:
        return None
    with _lock:
        if _model is not None and _model_name == name:
            return _model
        try:
            from sentence_transformers import CrossEncoder

            _model = CrossEncoder(name, max_length=512)
            _model_name = name
            logger.info("Spine re-ranker loaded: %s", name)
        except Exception as exc:  # noqa: BLE001
            _failed = True
            logger.warning("Spine re-ranker unavailable (%s: %s); recall uses hybrid order",
                           type(exc).__name__, exc)
            return None
    return _model


def _text(row: Dict[str, Any]) -> str:
    title = row.get("title")
    content = row.get("content") or ""
    return (f"{title}: {content}" if title else content)[:_MAX_CHARS]


def rerank(query: str, candidates: List[Dict[str, Any]],
           model_name: str = DEFAULT_MODEL) -> List[Dict[str, Any]]:
    """Return `candidates` re-ordered by hybrid rank blended with cross-encoder rank."""
    if len(candidates) < 2:
        return candidates
    model = _load(model_name or DEFAULT_MODEL)
    if model is None:
        return candidates
    try:
        scores = model.predict([(query, _text(r)) for r in candidates],
                               batch_size=32, show_progress_bar=False)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Spine re-ranker failed (%s: %s); using hybrid order",
                       type(exc).__name__, exc)
        return candidates
    by_model = sorted(range(len(candidates)), key=lambda i: -float(scores[i]))
    model_rank = {i: n for n, i in enumerate(by_model)}
    fused = sorted(range(len(candidates)),
                   key=lambda i: -(1.0 / (i + _RRF_K) + 1.0 / (model_rank[i] + _RRF_K)))
    return [candidates[i] for i in fused]
