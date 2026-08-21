"""Regression tests for host_supervisor._pid_alive and orphan reconciliation.

Covers the class of bug fixed in the round-3 CI sweep: ``_pid_alive`` must
treat unknown PIDs as dead (never raise), and a corrupt on-disk ``host_pid``
must self-heal through the normal ``not-running`` path instead of wedging
every subsequent turn.

The out-of-range and garbage-registry tests are direct red tests for the
regression where ``psutil.pid_exists`` raised ``OverflowError`` on a PID
larger than C ``long`` (e.g. ``2**63`` from a corrupt registry), escaping
``reconcile_startup_orphan`` and leaking the registry on disk.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from tui_gateway.host_supervisor import HostSupervisor, _pid_alive


def test_pid_alive_live_pid():
    assert _pid_alive(os.getpid()) is True


def test_pid_alive_reaped_pid():
    """A PID that has exited must report dead — except on Windows, where PIDs
    are recycled from a free pool immediately (so the probe may legitimately
    hit a reused PID)."""
    proc = subprocess.Popen([sys.executable, "-c", "pass"])
    proc.wait(timeout=30)
    pid = proc.pid
    result = _pid_alive(pid)
    if sys.platform == "win32":
        assert result in (True, False)  # PID may already be recycled
    else:
        assert result is False, "a reaped child must report dead"


def test_pid_alive_out_of_range_does_not_raise():
    """A corrupt registry PID (2**63) must resolve to False, not raise.

    Regression: the psutil-direct implementation raised OverflowError
    ("Python int too large to convert to C long") on PIDs beyond C long,
    escaping reconcile_startup_orphan and wedging the compute-host path.
    """
    assert _pid_alive(2**63) is False
    assert _pid_alive(2**31) is False
    assert _pid_alive(-1) is False
    assert _pid_alive(0) is False


def test_reconcile_startup_orphan_garbage_pid_removes_registry(tmp_path):
    """A corrupt host_pid in the registry must self-heal: not-running + removed."""
    registry_path = tmp_path / "registry.json"
    registry_path.write_text(
        json.dumps({"host_pid": 2**63, "hello": {"session_id": "x"}}),
        encoding="utf-8",
    )
    supervisor = HostSupervisor(registry_path=registry_path, autostart=False)

    result = supervisor.reconcile_startup_orphan()

    assert result == "not-running"
    assert not registry_path.exists(), "corrupt registry must be removed, not leaked"


def test_reconcile_startup_orphan_missing_registry_is_none(tmp_path):
    supervisor = HostSupervisor(registry_path=tmp_path / "absent.json", autostart=False)
    assert supervisor.reconcile_startup_orphan() == "none"


def test_reconcile_startup_orphan_invalid_registry_is_removed(tmp_path):
    registry_path = tmp_path / "registry.json"
    registry_path.write_text("not json at all", encoding="utf-8")
    supervisor = HostSupervisor(registry_path=registry_path, autostart=False)

    assert supervisor.reconcile_startup_orphan() == "invalid-registry"
    assert not registry_path.exists()
