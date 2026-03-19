"""
reconnect_handler.py — Automatic reconnection and state recovery for ΨBot.

Handles IB Gateway disconnects during trading hours gracefully:
- Exponential backoff reconnect attempts
- State snapshot before disconnect (preserved in memory)
- Re-subscribe options streams on reconnect
- Notify orchestrator to switch to Scout mode during outage
- Resume normal mode after confirmed reconnect

CCDR Expectation Field Architecture — Version 1.0
"""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime
from enum import Enum
from typing import Callable, Optional

log = logging.getLogger("psibot.reconnect_handler")

# Reconnect timing (seconds)
INITIAL_WAIT = 5.0
MAX_WAIT = 300.0     # 5 minutes max backoff
MAX_ATTEMPTS = 20    # give up after ~1 hour of cumulative wait


class ConnectionState(str, Enum):
    CONNECTED = "Connected"
    DISCONNECTED = "Disconnected"
    RECONNECTING = "Reconnecting"
    FAILED = "Failed"          # exhausted all attempts


class ReconnectHandler:
    """
    Manages automatic reconnection to IB Gateway.

    Integrates with IBKRBroker and OptionsStreamer to restore full
    streaming capability after a disconnect event.

    Usage:
        handler = ReconnectHandler(broker, options_streamer)
        handler.on_disconnect = lambda: orchestrator.set_mode("Scout")
        handler.on_reconnect = lambda: orchestrator.restore_mode()

        # Register with broker disconnect event
        broker.ib.disconnectedEvent += handler.handle_disconnect
    """

    def __init__(self, broker, options_streamer=None):
        self.broker = broker
        self.options_streamer = options_streamer
        self._state = ConnectionState.CONNECTED
        self._reconnect_task: Optional[asyncio.Task] = None
        self._attempts = 0
        self._last_disconnect: Optional[datetime] = None
        self._last_reconnect: Optional[datetime] = None

        # Callbacks — set by orchestrator
        self.on_disconnect: Optional[Callable] = None    # () → None; switch to Scout
        self.on_reconnect: Optional[Callable] = None     # () → None; restore mode
        self.on_give_up: Optional[Callable] = None       # () → None; page operator

        # Saved stream state for restoration
        self._stream_callback: Optional[Callable] = None
        self._stream_symbols: Optional[list] = None
        self._stream_interval: float = 30.0

    def register_stream_state(
        self,
        callback: Callable,
        symbols: list,
        interval_seconds: float = 30.0,
    ) -> None:
        """
        Save options stream parameters so they can be restored on reconnect.
        Call this after starting the OptionsStreamer.
        """
        self._stream_callback = callback
        self._stream_symbols = symbols
        self._stream_interval = interval_seconds

    async def handle_disconnect(self) -> None:
        """
        Called by broker.ib.disconnectedEvent.
        Switches orchestrator to Scout mode and starts reconnect loop.
        """
        if self._state == ConnectionState.RECONNECTING:
            return  # already handling

        self._state = ConnectionState.DISCONNECTED
        self._last_disconnect = datetime.utcnow()
        self._attempts = 0
        log.warning("IB Gateway disconnected at %s", self._last_disconnect.isoformat())

        # Notify orchestrator to switch to conservative Scout mode
        if self.on_disconnect:
            await self._safe_callback(self.on_disconnect)

        # Stop options streaming (stale data is worse than no data)
        if self.options_streamer and self.options_streamer.is_running:
            await self.options_streamer.stop()
            log.info("Options streamer stopped during disconnect")

        # Start reconnect loop
        self._reconnect_task = asyncio.create_task(self._reconnect_loop())

    async def _reconnect_loop(self) -> None:
        """Exponential backoff reconnect loop."""
        self._state = ConnectionState.RECONNECTING
        wait = INITIAL_WAIT

        while self._attempts < MAX_ATTEMPTS:
            self._attempts += 1
            log.info(
                "Reconnect attempt %d/%d (waiting %.1fs)…",
                self._attempts, MAX_ATTEMPTS, wait,
            )
            await asyncio.sleep(wait)

            try:
                await self.broker.connect()
                await self._on_successful_reconnect()
                return
            except Exception as exc:
                log.warning(
                    "Reconnect attempt %d failed: %s",
                    self._attempts, exc,
                )
                wait = min(wait * 2, MAX_WAIT)

        # Exhausted all attempts
        self._state = ConnectionState.FAILED
        log.critical(
            "RECONNECT FAILED after %d attempts — manual intervention required",
            MAX_ATTEMPTS,
        )
        if self.on_give_up:
            await self._safe_callback(self.on_give_up)

    async def _on_successful_reconnect(self) -> None:
        """Restore state after successful reconnection."""
        self._state = ConnectionState.CONNECTED
        self._last_reconnect = datetime.utcnow()
        downtime_seconds = (
            self._last_reconnect - self._last_disconnect
        ).total_seconds() if self._last_disconnect else 0

        log.info(
            "Reconnected to IB Gateway after %.0fs downtime (%d attempts)",
            downtime_seconds, self._attempts,
        )

        # Re-subscribe options streams if we have saved state
        if (
            self.options_streamer
            and self._stream_callback
            and self._stream_symbols
        ):
            try:
                await self.options_streamer.start(
                    callback=self._stream_callback,
                    symbols=self._stream_symbols,
                    update_interval_seconds=self._stream_interval,
                )
                log.info("Options streams restored for %d symbols", len(self._stream_symbols))
            except Exception as exc:
                log.error("Failed to restore options streams: %s", exc)

        # Notify orchestrator that connection is restored
        if self.on_reconnect:
            await self._safe_callback(self.on_reconnect)

    @property
    def connection_state(self) -> ConnectionState:
        return self._state

    @property
    def is_connected(self) -> bool:
        return self._state == ConnectionState.CONNECTED

    @property
    def downtime_seconds(self) -> float:
        """Current or last downtime duration in seconds."""
        if self._last_disconnect is None:
            return 0.0
        end = self._last_reconnect or datetime.utcnow()
        return (end - self._last_disconnect).total_seconds()

    async def _safe_callback(self, cb: Callable, *args) -> None:
        try:
            result = cb(*args)
            if asyncio.iscoroutine(result):
                await result
        except Exception as exc:
            log.error("ReconnectHandler callback error: %s", exc)
