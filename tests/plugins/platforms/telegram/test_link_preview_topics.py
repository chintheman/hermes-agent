"""Per-topic link previews (Chin, 2026-10-22): topic 5's workout card links every movement to its GIF,
and a preview under each card is noise there, while other topics keep their previews."""

from gateway.config import PlatformConfig
from plugins.platforms.telegram.adapter import TelegramAdapter
from tools.send_message_senders import telegram_previews_off

GROUP = "-1004389869012"


def adapter(extra):
    return TelegramAdapter(PlatformConfig(enabled=True, token="t", extra=extra))


def test_a_listed_topic_has_previews_off_and_another_topic_keeps_them():
    a = adapter({"disable_link_previews_topics": [GROUP + ":5"]})
    assert a._previews_off(GROUP, "5") is True
    assert a._link_preview_kwargs(GROUP, 5)
    assert a._previews_off(GROUP, "4") is False
    assert a._link_preview_kwargs(GROUP, "4") == {}
    assert a._previews_off(GROUP, None) is False


def test_the_rich_payload_carries_the_topic_switch():
    a = adapter({"disable_link_previews_topics": GROUP + ":5"})
    assert a._rich_payload_base(GROUP, "x", "5")["link_preview_options"] == {"is_disabled": True}
    assert "link_preview_options" not in a._rich_payload_base(GROUP, "x", "4")


def test_the_global_switch_still_covers_every_topic():
    a = adapter({"disable_link_previews": True})
    assert a._previews_off(GROUP, "4") is True
    assert a._previews_off() is True


def test_the_out_of_gateway_sender_reads_the_same_list():
    extra = {"disable_link_previews_topics": [GROUP + ":5"]}
    assert telegram_previews_off(extra, GROUP, 5) is True
    assert telegram_previews_off(extra, GROUP, 4) is False
    assert telegram_previews_off(extra, GROUP) is False
    assert telegram_previews_off({"disable_link_previews": True}, GROUP) is True
    assert telegram_previews_off(None, GROUP, 5) is False
