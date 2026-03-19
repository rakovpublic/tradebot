"""
account_monitor.py — Portfolio positions, P&L, and margin monitoring for ΨBot.

Tracks account equity, rolling drawdown, margin usage, and daily P&L.
Feeds AGT-07 (Risk Agent) and AGT-10 (Monitor Agent).

CCDR Expectation Field Architecture — Version 1.0
"""

from __future__ import annotations

import asyncio
import logging
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

log = logging.getLogger("psibot.account_monitor")

# Rolling window for drawdown computation (business days)
DRAWDOWN_WINDOW_DAYS = 10


@dataclass
class AccountSnapshot:
    """Point-in-time account state record."""
    timestamp: datetime
    net_liquidation: float
    equity_with_loan: float
    initial_margin: float
    maintenance_margin: float
    available_funds: float
    unrealized_pnl: float
    realized_pnl_today: float
    open_position_count: int = 0

    @property
    def margin_utilisation(self) -> float:
        """Fraction of equity consumed by initial margin."""
        if self.equity_with_loan <= 0:
            return 1.0
        return self.initial_margin / self.equity_with_loan

    @property
    def today_pnl(self) -> float:
        return self.unrealized_pnl + self.realized_pnl_today


class AccountMonitor:
    """
    Continuous account monitoring for ΨBot.

    Responsibilities:
    - Fetch account summary from IBKR on configurable interval
    - Maintain rolling equity curve for drawdown calculation
    - Fire callbacks when risk thresholds are breached
    - Provide data to AGT-07 drawdown circuit breaker

    Usage:
        monitor = AccountMonitor(broker, poll_interval_seconds=60)
        monitor.on_drawdown_breach = guardian_agent.activate_guardian
        monitor.on_margin_warning = lambda: log.warning("Margin high")

        await monitor.start()
        # runs in background

        snap = monitor.latest_snapshot
        dd = monitor.rolling_drawdown_pct
    """

    def __init__(
        self,
        broker,
        poll_interval_seconds: float = 60.0,
        drawdown_circuit_breaker_pct: float = 5.0,
        margin_warning_pct: float = 0.70,
    ):
        self.broker = broker
        self.poll_interval = poll_interval_seconds
        self.drawdown_circuit_breaker_pct = drawdown_circuit_breaker_pct
        self.margin_warning_pct = margin_warning_pct

        self._snapshots: deque[AccountSnapshot] = deque(maxlen=DRAWDOWN_WINDOW_DAYS * 24)
        self._peak_equity: float = 0.0
        self._running = False
        self._monitor_task: Optional[asyncio.Task] = None

        # Callbacks — set by orchestrator
        self.on_drawdown_breach: Optional[callable] = None  # (drawdown_pct: float) → None
        self.on_margin_warning: Optional[callable] = None   # (margin_pct: float) → None
        self.on_snapshot: Optional[callable] = None         # (snapshot: AccountSnapshot) → None

    async def start(self) -> None:
        """Start background monitoring loop."""
        self._running = True
        self._monitor_task = asyncio.create_task(self._monitor_loop())
        log.info("Account monitor started (interval=%.0fs)", self.poll_interval)

    async def stop(self) -> None:
        """Stop background monitoring."""
        self._running = False
        if self._monitor_task:
            self._monitor_task.cancel()
            try:
                await self._monitor_task
            except asyncio.CancelledError:
                pass
        log.info("Account monitor stopped")

    async def _monitor_loop(self) -> None:
        while self._running:
            try:
                await self._poll()
            except Exception as exc:
                log.error("Account monitor poll error: %s", exc)
            await asyncio.sleep(self.poll_interval)

    async def _poll(self) -> None:
        """Fetch account state and check thresholds."""
        summary = await self.broker.get_account_summary()
        positions = await self.broker.get_open_positions()

        snap = AccountSnapshot(
            timestamp=datetime.utcnow(),
            net_liquidation=summary.get("net_liquidation", 0.0),
            equity_with_loan=summary.get("equity_with_loan", 0.0),
            initial_margin=summary.get("initial_margin", 0.0),
            maintenance_margin=summary.get("maintenance_margin", 0.0),
            available_funds=summary.get("available_funds", 0.0),
            unrealized_pnl=summary.get("unrealized_pnl", 0.0),
            realized_pnl_today=summary.get("realized_pnl_today", 0.0),
            open_position_count=len([p for p in positions if p.position != 0]),
        )

        self._snapshots.append(snap)

        # Update peak equity
        if snap.net_liquidation > self._peak_equity:
            self._peak_equity = snap.net_liquidation

        # Drawdown check
        dd_pct = self.rolling_drawdown_pct
        if dd_pct >= self.drawdown_circuit_breaker_pct:
            log.critical(
                "DRAWDOWN CIRCUIT BREAKER: %.1f%% ≥ %.1f%%",
                dd_pct, self.drawdown_circuit_breaker_pct,
            )
            if self.on_drawdown_breach:
                await self._safe_callback(self.on_drawdown_breach, dd_pct)

        # Margin warning
        if snap.margin_utilisation >= self.margin_warning_pct:
            log.warning(
                "Margin utilisation high: %.1f%% of equity",
                snap.margin_utilisation * 100,
            )
            if self.on_margin_warning:
                await self._safe_callback(self.on_margin_warning, snap.margin_utilisation)

        if self.on_snapshot:
            await self._safe_callback(self.on_snapshot, snap)

        log.debug(
            "Account: NLV=%.0f margin=%.1f%% dd=%.2f%% positions=%d",
            snap.net_liquidation,
            snap.margin_utilisation * 100,
            dd_pct,
            snap.open_position_count,
        )

    @property
    def latest_snapshot(self) -> Optional[AccountSnapshot]:
        """Most recent account snapshot."""
        return self._snapshots[-1] if self._snapshots else None

    @property
    def rolling_drawdown_pct(self) -> float:
        """
        Rolling drawdown from peak equity over the monitoring window.
        Used by AGT-07 drawdown circuit breaker.
        Returns 0.0 if insufficient history.
        """
        if not self._snapshots or self._peak_equity <= 0:
            return 0.0
        current = self._snapshots[-1].net_liquidation
        return max(0.0, (self._peak_equity - current) / self._peak_equity * 100)

    @property
    def today_pnl(self) -> float:
        snap = self.latest_snapshot
        return snap.today_pnl if snap else 0.0

    @property
    def net_liquidation(self) -> float:
        snap = self.latest_snapshot
        return snap.net_liquidation if snap else 0.0

    def equity_history(self) -> list[tuple[datetime, float]]:
        """Return list of (timestamp, net_liquidation) tuples for charting."""
        return [(s.timestamp, s.net_liquidation) for s in self._snapshots]

    async def _safe_callback(self, cb, *args) -> None:
        try:
            result = cb(*args)
            if asyncio.iscoroutine(result):
                await result
        except Exception as exc:
            log.error("Account monitor callback error: %s", exc)
