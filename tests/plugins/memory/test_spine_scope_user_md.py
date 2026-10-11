"""USER.md honours @when: scoping: deferred out of the snapshot, delivered by spine prefetch."""
from pathlib import Path

from tools.memory_tool_store import MemoryStore

MEM = "[R] always rule\n§\nuniversal memory fact\n§\ngarmin cron note @when:subject=garmin,cron"
USER = ("identity: Chin lives in Singapore\n§\nsquat 5x5 at 100kg @when:subject=squat,sleep,gym"
        "\n§\n[R] user rule @when:subject=never")


def _setup(tmp_path, monkeypatch):
    mem = Path(tmp_path) / "memories"
    mem.mkdir(parents=True)
    (mem / "MEMORY.md").write_text(MEM, encoding="utf-8")
    (mem / "USER.md").write_text(USER, encoding="utf-8")
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    return mem


def _ctx(query):
    import plugins.memory.spine as spine
    return spine.SpineProvider._build_scoped_context(object(), query)


def test_user_md_tagged_entry_leaves_snapshot_and_returns_via_prefetch(tmp_path, monkeypatch):
    _setup(tmp_path, monkeypatch)
    s = MemoryStore(scope_filter=True)
    s.load_from_disk()
    user_snap = s.format_for_system_prompt("user")
    assert "identity: Chin" in user_snap
    assert "squat 5x5" not in user_snap           # deferred out of the prompt
    assert "[R] user rule" in user_snap           # [R] never deferred
    assert "squat 5x5" in _ctx("how was my sleep and squat today")
    assert "[USER PROFILE" in _ctx("how was my sleep and squat today")
    assert "squat 5x5" not in _ctx("hello")        # no subject word, nothing delivered


def test_scope_filter_off_keeps_everything_in_both_snapshots(tmp_path, monkeypatch):
    _setup(tmp_path, monkeypatch)
    s = MemoryStore(scope_filter=False)
    s.load_from_disk()
    assert "squat 5x5" in s.format_for_system_prompt("user")
    assert "garmin cron" in s.format_for_system_prompt("memory")
