"""Enforced-scope self-scan wrapper for scripts/check-windows-footguns.py.

scripts/check_subprocess_stdin.py has had a pytest wrapper (see
tests/tools/test_subprocess_stdin_guard.py's test_all_tui_subprocess_calls_
have_stdin) that runs the checker with its default full-scan behavior and
asserts a clean exit — so a normal pytest run of that file catches
regressions even when no one remembers to run the standalone script by hand.
check-windows-footguns.py had no equivalent: only a narrow rule-level test
(tests/scripts/test_footgun_subprocess_encoding.py, scoped to the
text=True/encoding= rule) existed, so a bare ``os.killpg``/``signal.SIGKILL``
regression (caught by CI running the real script with --all, not by any
local pytest run) shipped in the T1-T3 npx-agent-browser hardening commit
before anyone ran the script directly. This closes that gap the same way
the stdin guard already closes its equivalent one.
"""

from __future__ import annotations

import importlib.util
import subprocess
import sys
import tomllib
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = REPO_ROOT / "scripts" / "check-windows-footguns.py"


def _load_scanner():
    """Import scripts/check-windows-footguns.py as a module (it is a script)."""
    spec = importlib.util.spec_from_file_location("check_windows_footguns", SCRIPT)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    # The scanner's dataclasses resolve their module namespace via
    # sys.modules; register the module before executing it.
    sys.modules["check_windows_footguns"] = mod
    spec.loader.exec_module(mod)
    return mod


def test_enforced_scope_scan_has_no_unsuppressed_windows_footguns():
    """Mirrors check_subprocess_stdin.py's wrapper: run the real checker
    against the enforced scope (--all) and require a clean exit, so this test
    file — not just institutional memory — is what catches the next
    bare os.killpg/signal.SIGKILL-style regression."""
    result = subprocess.run(
        [sys.executable, str(SCRIPT), "--all"],
        capture_output=True,
        text=True,
        timeout=60,
        stdin=subprocess.DEVNULL,
    )
    assert result.returncode == 0, (
        f"Windows footgun check failed:\n{result.stdout}\n{result.stderr}"
    )


def test_all_roots_cover_shipped_packages():
    """--all must reach every top-level package pyproject.toml ships.

    Regression guard: the roots list is hand-maintained and drifted from the
    shipped-package set (tui_gateway/, providers/ were missing while the gate
    stayed green — the scan just got quieter, silently). Parse
    ``[tool.setuptools.packages.find].include`` and require each top-level
    package to be reachable from the ``--all`` roots, so the next drift is a
    red gate instead of a review finding.
    """
    scanner = _load_scanner()
    roots = scanner.all_roots(REPO_ROOT)

    with open(REPO_ROOT / "pyproject.toml", "rb") as f:
        pyproject = tomllib.load(f)
    includes = pyproject["tool"]["setuptools"]["packages"]["find"]["include"]

    for entry in includes:
        top_level = entry.split(".")[0]
        assert (REPO_ROOT / top_level) in roots, (
            f"--all roots miss shipped package: {top_level}"
        )
