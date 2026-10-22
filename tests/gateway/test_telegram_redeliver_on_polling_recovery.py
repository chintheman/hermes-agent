"""A message refused while Telegram polling was down is resent when polling recovers (Chin, 2026-10-22).
The 09:12 workout card was queued for redelivery, but an in-place polling restart keeps the same
adapter, so the runner's reconnect sweep never ran and the card waited for the 12:56 restart."""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from gateway.config import Platform, PlatformConfig
from plugins.platforms.telegram.adapter import TelegramAdapter


def adapter_with_runner():
    a = TelegramAdapter(PlatformConfig(enabled=True, token="test-token"))
    a.gateway_runner = SimpleNamespace(_redeliver_failed_obligations_for_platform=AsyncMock(return_value=1))
    a._polling_progress_accepting = True
    return a


@pytest.mark.asyncio
async def test_recovery_from_a_degraded_send_path_resends_what_was_refused():
    a = adapter_with_runner()
    a._send_path_degraded = True
    assert a._record_polling_progress(a._polling_generation)
    await asyncio.sleep(0)
    a.gateway_runner._redeliver_failed_obligations_for_platform.assert_awaited_once()
    assert a.gateway_runner._redeliver_failed_obligations_for_platform.await_args.args == (Platform.TELEGRAM,)


@pytest.mark.asyncio
async def test_ordinary_progress_does_not_sweep():
    a = adapter_with_runner()
    a._send_path_degraded = False
    a._record_polling_progress(a._polling_generation)
    a._record_polling_progress(a._polling_generation)
    await asyncio.sleep(0)
    a.gateway_runner._redeliver_failed_obligations_for_platform.assert_not_awaited()


def test_no_running_loop_is_harmless():
    a = adapter_with_runner()
    a._send_path_degraded = True
    assert a._record_polling_progress(a._polling_generation)
    a.gateway_runner._redeliver_failed_obligations_for_platform.assert_not_called()
