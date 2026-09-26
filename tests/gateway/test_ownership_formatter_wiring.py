"""Tests that the ownership-block formatter (gateway/ownership_format.py) is actually wired into
the single choke point every platform adapter's outbound send passes through:
``BasePlatformAdapter.__init__`` wraps the concrete subclass's ``send``/``edit_message`` (and
``send_for_platform`` when the adapter defines one, e.g. Relay). Covers the config flag
(``display.ownership_formatter``, default ON) and the never-fail-a-send guarantee at the
dispatch layer, on top of the pure-function tests in test_ownership_format.py.
"""

import pytest

from gateway.config import Platform, PlatformConfig
from gateway.platforms.base import BasePlatformAdapter, SendResult


class _RecordingAdapter(BasePlatformAdapter):
    """Minimal concrete adapter: records exactly what content it was asked to deliver."""

    def __init__(self):
        super().__init__(PlatformConfig(enabled=True, token="***"), Platform.TELEGRAM)
        self.sent_content = []
        self.edited_content = []

    async def connect(self, *, is_reconnect: bool = False):
        return True

    async def disconnect(self):
        return None

    async def send(self, chat_id, content, reply_to=None, metadata=None):
        self.sent_content.append(content)
        return SendResult(success=True, message_id="1")

    async def edit_message(self, chat_id, message_id, content, *, finalize=False):
        self.edited_content.append(content)
        return SendResult(success=True, message_id=message_id)

    async def get_chat_info(self, chat_id):
        return {"id": chat_id}


class _RelayLikeAdapter(_RecordingAdapter):
    """Adds send_for_platform — the third egress door DeliveryTransport.send() calls directly
    for a Relay transport fronting a logical platform (gateway/delivery.py)."""

    def __init__(self):
        super().__init__()
        self.sent_for_platform_content = []

    async def send_for_platform(self, logical_platform, chat_id, content, reply_to=None, metadata=None):
        self.sent_for_platform_content.append(content)
        return SendResult(success=True, message_id="2")


def _write_config(tmp_path, monkeypatch, body: str) -> None:
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    (tmp_path / "config.yaml").write_text(body)


@pytest.mark.asyncio
async def test_send_collapses_ownership_block():
    adapter = _RecordingAdapter()
    await adapter.send("123", "Body.\n\nMINE: shipped it\nYOURS: none\n\nMINE: shipped it")
    assert adapter.sent_content == ["Body.\n\nMINE: shipped it"]


@pytest.mark.asyncio
async def test_send_with_keyword_content_is_formatted():
    adapter = _RecordingAdapter()
    await adapter.send(chat_id="123", content="Body.\n\nMINE: nothing running\nYOURS: approve")
    assert adapter.sent_content == ["Body.\n\nYOURS: approve"]


@pytest.mark.asyncio
async def test_edit_message_collapses_ownership_block():
    adapter = _RecordingAdapter()
    await adapter.edit_message("123", "msg1", "Body.\n\nYOURS: nothing\nMINE: shipped it", finalize=True)
    assert adapter.edited_content == ["Body.\n\nMINE: shipped it"]


@pytest.mark.asyncio
async def test_send_for_platform_collapses_ownership_block():
    """The Relay-style third egress door gets the same treatment."""
    adapter = _RelayLikeAdapter()
    await adapter.send_for_platform(Platform.SLACK, "123", "Body.\n\nMINE: a\n\nMINE: a")
    assert adapter.sent_for_platform_content == ["Body.\n\nMINE: a"]


@pytest.mark.asyncio
async def test_send_final_ledgered_path_is_formatted():
    """The real turn-final delivery path (send_final_ledgered -> _send_with_retry ->
    self.send) resolves ``self.send`` to the wrapped instance attribute, not the raw subclass
    method — proving the wrap covers the path every interactive chat reply actually takes, not
    just a direct ``adapter.send(...)`` call."""
    adapter = _RecordingAdapter()
    result = await adapter._send_with_retry(
        chat_id="123", content="Body.\n\nMINE: shipped it\n\nMINE: shipped it")
    assert result.success is True
    assert adapter.sent_content == ["Body.\n\nMINE: shipped it"]


@pytest.mark.asyncio
async def test_send_untouched_when_no_labels_present():
    adapter = _RecordingAdapter()
    await adapter.send("123", "Just a normal reply.")
    assert adapter.sent_content == ["Just a normal reply."]


@pytest.mark.asyncio
async def test_config_flag_disables_formatter(tmp_path, monkeypatch):
    """display.ownership_formatter: false must leave the raw text untouched — the exact
    duplicated-block text Chin complained about, if an operator needs to debug the formatter
    itself."""
    _write_config(tmp_path, monkeypatch, "display:\n  ownership_formatter: false\n")
    adapter = _RecordingAdapter()
    raw = "Body.\n\nMINE: shipped it\n\nMINE: shipped it"
    await adapter.send("123", raw)
    assert adapter.sent_content == [raw]


@pytest.mark.asyncio
async def test_config_flag_default_is_on(tmp_path, monkeypatch):
    _write_config(tmp_path, monkeypatch, "{}\n")
    adapter = _RecordingAdapter()
    await adapter.send("123", "Body.\n\nMINE: shipped it\n\nMINE: shipped it")
    assert adapter.sent_content == ["Body.\n\nMINE: shipped it"]


@pytest.mark.asyncio
async def test_formatter_exception_never_blocks_a_send(monkeypatch):
    """A raising formatter must degrade to the ORIGINAL text — a send is never lost because of
    this feature."""
    import gateway.platforms.base as gw_base

    def _boom(_text):
        raise RuntimeError("boom")

    monkeypatch.setattr(gw_base, "format_ownership_block", _boom)
    adapter = _RecordingAdapter()
    raw = "Body.\n\nMINE: shipped it"
    result = await adapter.send("123", raw)
    assert result.success is True
    assert adapter.sent_content == [raw]


@pytest.mark.asyncio
async def test_disabled_flag_check_failure_never_blocks_a_send(monkeypatch):
    """A raising config check must also degrade safely rather than dropping the send."""
    import gateway.platforms.base as gw_base

    def _boom():
        raise RuntimeError("boom")

    monkeypatch.setattr(gw_base, "_ownership_formatter_enabled", _boom)
    adapter = _RecordingAdapter()
    raw = "Body.\n\nMINE: shipped it"
    result = await adapter.send("123", raw)
    assert result.success is True
    assert adapter.sent_content == [raw]
