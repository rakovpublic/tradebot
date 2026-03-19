"""
options_streamer.py — Manages continuous IV surface streaming for all
instruments in the options universe, feeding data to the L1 ψ_exp agent.

CCDR Expectation Field Architecture — Version 1.0
"""

import asyncio
import logging
from typing import Callable, Optional

log = logging.getLogger("psibot.options_streamer")


class OptionsStreamer:
    """
    Manages streaming IV surfaces for all instruments in the options universe.

    The L1 ψ_exp Agent calls this; it subscribes to updates and reconstructs
    ψ_exp on every surface update that exceeds the vol-shift threshold (0.5 vega).

    Usage:
        streamer = OptionsStreamer(broker)
        await streamer.start(callback=psi_agent.on_surface_update)
        # ... later ...
        await streamer.stop()
    """

    def __init__(self, broker):
        self.broker = broker
        self._stream_tasks: dict[str, asyncio.Task] = {}
        self._running = False

    async def start(
        self,
        callback: Callable,
        symbols: list = None,
        update_interval_seconds: float = 30.0,
    ) -> None:
        """
        Start streaming IV surfaces for all symbols in the options universe.

        Args:
            callback: async callable invoked with surface dict on each update
            symbols: list of underlying symbols; defaults to config.options_universe
            update_interval_seconds: minimum interval between surface updates
        """
        if symbols is None:
            symbols = self.broker.config.options_universe

        self._running = True
        for symbol in symbols:
            task = await self.broker.stream_options_surface(
                underlying_symbol=symbol,
                callback=callback,
                update_interval_seconds=update_interval_seconds,
            )
            self._stream_tasks[symbol] = task
            log.info("Started IV surface stream: %s", symbol)

        log.info("Options streamer started for %d instruments", len(symbols))

    async def stop(self) -> None:
        """Cancel all streaming tasks gracefully."""
        self._running = False
        for symbol, task in self._stream_tasks.items():
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
            log.info("Stopped IV surface stream: %s", symbol)
        self._stream_tasks.clear()
        log.info("Options streamer stopped")

    async def restart_symbol(self, symbol: str, callback: Callable) -> None:
        """Restart stream for a single symbol after error."""
        if symbol in self._stream_tasks:
            self._stream_tasks[symbol].cancel()
            try:
                await self._stream_tasks[symbol]
            except asyncio.CancelledError:
                pass

        task = await self.broker.stream_options_surface(symbol, callback)
        self._stream_tasks[symbol] = task
        log.info("Restarted IV surface stream: %s", symbol)

    @property
    def active_symbols(self) -> list:
        """Return list of symbols with active stream tasks."""
        return [s for s, t in self._stream_tasks.items() if not t.done()]

    @property
    def is_running(self) -> bool:
        return self._running and bool(self._stream_tasks)
