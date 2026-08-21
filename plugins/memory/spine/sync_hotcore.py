#!/usr/bin/env python3
"""Import MEMORY.md blocks that spine has never seen.

MEMORY.md has two writers. Spine's promote pass is one; Hermes's own memory()
tool is the other, and it writes straight into the hot core without ever
touching spine. So the always-loaded file steadily accumulates facts that exist
nowhere else -- 15 of them on 2026-08-20, which a "these are already in spine,
just delete them" cleanup would have destroyed outright.

This closes that leak: anything in the hot core that is not retrievable becomes
a spine observation, so the hot core can be trimmed safely at any time.

Imported rows are written as `demoted`: searchable, but never re-promoted back
into the file by the consolidation pass.

Silent when there is nothing to do.
"""
from __future__ import annotations

import argparse
import json
import os
import sqlite3
import sys

PROFILE = "agent:main"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    # spine/__init__.py imports agent.memory_provider, so the repo root has to
    # be on the path too or a manual run dies with ModuleNotFoundError: agent.
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.dirname(os.path.abspath(__file__))))))
    from spine.config import load_spine_config
    from spine import coverage
    from spine.embedder import embedder_available
    from spine.tools import handle_remember

    config = load_spine_config()
    mem_path = os.path.expanduser("~/.hermes/memories/MEMORY.md")
    con = sqlite3.connect(os.path.expanduser(config.db))
    con.execute("PRAGMA query_only = ON")
    try:
        uncovered = coverage.uncovered_hotcore(mem_path, con)
        total = len(coverage.hotcore_blocks(mem_path))
    finally:
        con.close()

    if not uncovered:
        if args.dry_run:
            print(f"hot core: {total} blocks, all retrievable from spine")
        return

    print(f"hot core: {total} blocks, {len(uncovered)} not retrievable from spine")
    for block, missing in uncovered:
        print(f"  + {block[:88].replace(chr(10), ' ')}")

    if args.dry_run:
        print("\nDRY RUN — nothing written.")
        return

    if not embedder_available():
        # Importing without vectors would put these rows in the store but leave
        # them findable by keyword only, which is the quiet half-failure this
        # whole exercise exists to prevent.
        sys.exit("refusing to import: embedder unavailable, rows would have no vectors")

    ids = []
    for block, _missing in uncovered:
        content = coverage.strip_tag(block)
        r = json.loads(handle_remember({
            "content": content,
            "type": "fact",
            "epistemic": "extracted",
            "confidence": 0.9,
            "topics": [],
            "evidence": ["MEMORY.md hot core, imported by sync_hotcore"],
        }, config))
        if r.get("success"):
            ids.append(r["id"])
        else:
            print(f"  ! failed: {content[:70]} -> {r}")

    # Demoted, not active: these have already had their turn in the hot core and
    # must not be promoted back into the file they came from.
    from spine.index import MemoryIndex
    idx = MemoryIndex(config.db)
    idx.open()
    try:
        for oid in ids:
            idx.update_status(oid, "demoted")   # writes DB + canonical JSONL
        idx.conn.commit()
    finally:
        idx.close()

    # Verify rather than assume: re-run the same coverage test.
    con = sqlite3.connect(os.path.expanduser(config.db))
    con.execute("PRAGMA query_only = ON")
    try:
        still = coverage.uncovered_hotcore(mem_path, con)
    finally:
        con.close()

    print(f"\nimported {len(ids)}, still uncovered: {len(still)}")
    if still:
        sys.exit(f"POST-IMPORT CHECK FAILED: {len(still)} blocks remain unretrievable")
    print("hot core fully retrievable")


if __name__ == "__main__":
    main()
