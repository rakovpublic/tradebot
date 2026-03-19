"""
state/portfolio_state.py — Portfolio and Position State
========================================================
Tracks open positions, P&L, and portfolio-level risk metrics.

CCDR Expectation Field Architecture — Version 1.0
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional
import numpy as np
import pandas as pd

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from helpers import (
    SignalClass, SignalDirection, MarketPhase, PsiShape,
    BotMode, CCDR_THRESHOLDS,
)


@dataclass
class Position:
    """A single open position opened by the signal layer."""
    position_id: str
    signal_class: SignalClass
    direction: SignalDirection
    instrument: str

    # Entry state snapshot — used for structural stop computation
    entry_timestamp: datetime = field(default_factory=datetime.utcnow)
    entry_gbp: float = 0.0
    entry_phase: MarketPhase = MarketPhase.UNKNOWN
    entry_psi_shape: PsiShape = PsiShape.UNKNOWN
    entry_d_eff: float = 5.0
    entry_price: float = 0.0

    # Sizing
    notional_usd: float = 0.0
    size_multiplier: float = 0.0

    # P&L tracking
    current_price: float = 0.0
    unrealised_pnl_usd: float = 0.0
    realised_pnl_usd: float = 0.0

    # Lifecycle
    is_open: bool = True
    close_timestamp: Optional[datetime] = None
    close_reason: str = ""

    # Historical prices for portfolio D_eff computation (last 30 bars)
    pnl_history: list = field(default_factory=list)

    def update_pnl(self, current_price: float) -> None:
        self.current_price = current_price
        multiplier = 1.0 if self.direction == SignalDirection.LONG else -1.0
        if self.entry_price > 0:
            pct_change = (current_price - self.entry_price) / self.entry_price
            self.unrealised_pnl_usd = multiplier * pct_change * self.notional_usd
            self.pnl_history.append(self.unrealised_pnl_usd)
            if len(self.pnl_history) > 30:
                self.pnl_history = self.pnl_history[-30:]

    def close(self, reason: str, current_price: Optional[float] = None) -> None:
        if current_price is not None:
            self.update_pnl(current_price)
        self.realised_pnl_usd = self.unrealised_pnl_usd
        self.unrealised_pnl_usd = 0.0
        self.is_open = False
        self.close_timestamp = datetime.utcnow()
        self.close_reason = reason


@dataclass
class PortfolioState:
    """Tracks the current portfolio — positions, P&L, and risk metrics."""
    account_equity: float = 100_000.0
    max_risk_usd: float = 10_000.0

    positions: list = field(default_factory=list)

    # Rolling P&L tracking
    daily_pnl_history: list = field(default_factory=list)
    peak_equity: float = field(init=False)

    def __post_init__(self):
        self.peak_equity = self.account_equity

    @property
    def open_positions(self) -> list:
        return [p for p in self.positions if p.is_open]

    @property
    def position_count(self) -> int:
        return len(self.open_positions)

    @property
    def risk_positions(self) -> list:
        """Soliton and Reorder positions — closed in Guardian mode."""
        return [p for p in self.open_positions
                if p.signal_class in (SignalClass.SOLITON, SignalClass.REORDER)]

    @property
    def total_unrealised_pnl(self) -> float:
        return sum(p.unrealised_pnl_usd for p in self.open_positions)

    @property
    def total_notional(self) -> float:
        return sum(p.notional_usd for p in self.open_positions)

    def add_position(self, position: Position) -> bool:
        """Add position if under max concurrent limit."""
        if self.position_count >= CCDR_THRESHOLDS["MAX_POSITIONS"]:
            return False
        self.positions.append(position)
        return True

    def rolling_drawdown(self, days: int = 10) -> float:
        """Rolling P&L drawdown over last `days` periods. Negative = loss."""
        if len(self.daily_pnl_history) < 2:
            return 0.0
        recent = self.daily_pnl_history[-days:]
        peak = max(recent)
        current = recent[-1]
        if self.account_equity > 0:
            return (current - peak) / self.account_equity
        return 0.0

    def update_peak_equity(self) -> None:
        current = self.account_equity + self.total_unrealised_pnl
        if current > self.peak_equity:
            self.peak_equity = current

    def max_drawdown_pct(self) -> float:
        """Max drawdown since peak equity."""
        current = self.account_equity + self.total_unrealised_pnl
        if self.peak_equity > 0:
            return (current - self.peak_equity) / self.peak_equity
        return 0.0

    def get_pnl_matrix(self, days: int = 30) -> np.ndarray:
        """
        Returns P&L series matrix for portfolio D_eff computation.
        Shape: (days, n_positions)
        """
        pnl_series = []
        for pos in self.open_positions:
            if len(pos.pnl_history) >= 3:
                pnl_series.append(pos.pnl_history[-days:])
        if len(pnl_series) < 3:
            return np.array([])
        min_len = min(len(s) for s in pnl_series)
        matrix = np.array([s[-min_len:] for s in pnl_series]).T
        return matrix

    def snapshot(self) -> dict:
        return {
            "timestamp": datetime.utcnow().isoformat(),
            "account_equity": self.account_equity,
            "total_unrealised_pnl": round(self.total_unrealised_pnl, 2),
            "position_count": self.position_count,
            "total_notional": round(self.total_notional, 2),
            "rolling_drawdown_10d": round(self.rolling_drawdown(10), 4),
            "max_drawdown_pct": round(self.max_drawdown_pct(), 4),
            "positions": [
                {
                    "id": p.position_id,
                    "signal_class": p.signal_class.value,
                    "direction": p.direction.value,
                    "instrument": p.instrument,
                    "notional_usd": p.notional_usd,
                    "unrealised_pnl": round(p.unrealised_pnl_usd, 2),
                    "entry_gbp": p.entry_gbp,
                }
                for p in self.open_positions
            ],
        }
