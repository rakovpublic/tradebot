"""
backtesting/backtest_engine.py — Historical Simulation Engine
=============================================================
Runs the full 5-layer CCDR pipeline on historical data.
Used in Phase 0 (validation) and Phase 1-2 (paper trading).

Supports:
  - Full pipeline replay on historical data
  - Paper trade P&L simulation with CCDR structural stops
  - Performance reporting (Sharpe, drawdown, hit rate)

CCDR Expectation Field Architecture — Version 1.0
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Optional
import numpy as np
import pandas as pd

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from helpers import (
    BotMode, MarketPhase, SignalClass, CCDR_THRESHOLDS,
    determine_bot_mode, compute_position_size,
)

log = logging.getLogger("psibot.backtest.engine")


@dataclass
class BacktestTrade:
    """A completed paper trade in backtest."""
    signal_class: SignalClass
    entry_date: datetime
    exit_date: Optional[datetime] = None
    entry_price: float = 0.0
    exit_price: float = 0.0
    entry_gbp: float = 0.0
    entry_d_eff: float = 0.0
    entry_phase: str = ""
    notional_usd: float = 0.0
    pnl_usd: float = 0.0
    exit_reason: str = ""
    is_open: bool = True


@dataclass
class BacktestResult:
    """Summary statistics for a completed backtest."""
    start_date: datetime
    end_date: datetime
    total_trades: int = 0
    winning_trades: int = 0
    total_pnl_usd: float = 0.0
    max_drawdown_pct: float = 0.0
    sharpe_ratio: float = 0.0
    hit_rate: float = 0.0
    avg_risk_reward: float = 0.0
    guardian_activations: int = 0
    trades: list = field(default_factory=list)
    daily_pnl: list = field(default_factory=list)

    def summary(self) -> str:
        return (
            f"Backtest {self.start_date.date()} → {self.end_date.date()}\n"
            f"  Trades: {self.total_trades} | Hit rate: {self.hit_rate:.1%}\n"
            f"  Total P&L: ${self.total_pnl_usd:,.0f}\n"
            f"  Sharpe: {self.sharpe_ratio:.2f} | Max drawdown: {self.max_drawdown_pct:.1%}\n"
            f"  Guardian activations: {self.guardian_activations}\n"
        )


class BacktestEngine:
    """
    Historical pipeline replay engine.
    Simulates the full CCDR pipeline day-by-day on historical data.
    """

    def __init__(self, config: dict = None):
        self.config = config or {}
        self.max_risk_usd = self.config.get("max_risk_usd", 10_000.0)
        self.initial_equity = self.config.get("initial_equity", 100_000.0)
        self._trades: list[BacktestTrade] = []
        self._daily_equity: list[float] = []

    async def run(
        self,
        historical_data: dict,
        start_date: datetime,
        end_date: Optional[datetime] = None,
    ) -> BacktestResult:
        """
        Run full backtest over historical data.

        historical_data dict keys:
          - "options": dict of {date: OptionsSurface}
          - "analyst": dict of {date: AnalystData}
          - "surveys": dict of {date: SurveyData}
          - "dark_pool": dict of {date: DarkPoolData}
          - "prices": pd.DataFrame with multi-asset prices
        """
        from psibot.state.condensate_state import CondensateState
        from psibot.state.portfolio_state import PortfolioState
        from psibot.modes.guardian import GuardianModeController

        if end_date is None:
            end_date = datetime.utcnow()

        log.info("Backtest starting: %s → %s", start_date.date(), end_date.date())

        portfolio = PortfolioState(
            account_equity=self.initial_equity,
            max_risk_usd=self.max_risk_usd,
        )
        guardian = GuardianModeController()

        # Get trading dates from price data
        prices_df = historical_data.get("prices")
        if prices_df is None:
            raise ValueError("historical_data must include 'prices' DataFrame")

        trading_dates = [
            d for d in prices_df.index
            if start_date <= d <= end_date
        ]

        result = BacktestResult(start_date=start_date, end_date=end_date)
        prev_state = None

        for date in trading_dates:
            try:
                state = await self._run_pipeline_for_date(
                    date=date,
                    historical_data=historical_data,
                    prices_df=prices_df,
                )

                if state is None:
                    continue

                # Check Guardian triggers
                guardian.check_guardian_triggers(state, portfolio)
                if guardian.guardian_active:
                    result.guardian_activations += 1
                    state.active_mode = BotMode.GUARDIAN

                # Update portfolio positions (simplified P&L update)
                self._update_portfolio_pnl(portfolio, prices_df, date)

                # Check structural stops
                exits = self._check_stops(state, portfolio)
                for exit_info in exits:
                    self._close_trade(portfolio, exit_info, date, prices_df)

                # Generate signals (if not in Guardian mode)
                if state.active_mode == BotMode.HUNTER and not guardian.guardian_active:
                    self._generate_paper_signals(state, portfolio, date, prices_df)

                # Guardian exit check
                guardian.check_guardian_exit(state)

                # Record daily equity
                equity = portfolio.account_equity + portfolio.total_unrealised_pnl
                portfolio.daily_pnl_history.append(equity)
                self._daily_equity.append(equity)
                prev_state = state

            except Exception as e:
                log.error("Backtest error on %s: %s", date, e)
                continue

        # Compute performance metrics
        result.trades = self._trades
        result.total_trades = len(self._trades)
        result.winning_trades = sum(1 for t in self._trades if t.pnl_usd > 0)
        result.total_pnl_usd = sum(t.pnl_usd for t in self._trades)
        result.daily_pnl = self._daily_equity

        if result.total_trades > 0:
            result.hit_rate = result.winning_trades / result.total_trades

        result.sharpe_ratio = self._compute_sharpe(self._daily_equity)
        result.max_drawdown_pct = self._compute_max_drawdown(self._daily_equity)

        log.info("Backtest complete:\n%s", result.summary())
        return result

    async def _run_pipeline_for_date(self, date, historical_data, prices_df):
        """Run the 5-layer pipeline for a single historical date."""
        from psibot.state.condensate_state import CondensateState
        from psibot.data.options_feed import OptionsSurface
        from psibot.data.cross_asset_feed import CrossAssetData
        from psibot.pipeline import (
            l1_psi_reconstruction, l2_phase_detector,
            l3_holo_monitor, l4_grain_boundary, l5_acoustic_parser
        )
        from psibot.execution.sizing import compute_size_multiplier

        state = CondensateState(timestamp=date)

        # Get options surface for this date
        options = historical_data.get("options", {}).get(date)

        # Get analyst/survey data (use most recent available)
        analyst_data = self._get_most_recent(historical_data.get("analyst", {}), date)
        survey_data = self._get_most_recent(historical_data.get("surveys", {}), date)

        # Get dark pool data
        dp_data = self._get_most_recent(historical_data.get("dark_pool", {}), date)
        dark_pool_ratio = dp_data.dark_pool_ratio if dp_data else 1.0

        # Get returns matrix up to this date
        available_prices = prices_df[prices_df.index <= date].tail(61)
        if len(available_prices) < 10:
            return None

        from helpers import rolling_window_returns
        returns_matrix = rolling_window_returns(available_prices, window=60)

        # Synthetic cross-asset data wrapper
        cross_asset_data = _SimpleCAData(
            returns_matrix=returns_matrix,
            prices=available_prices,
        )

        # Run pipeline
        state = await l1_psi_reconstruction.run(state, options)
        state = await l2_phase_detector.run(state, analyst_data, survey_data)
        state = await l3_holo_monitor.run(state, cross_asset_data)
        state = await l4_grain_boundary.run(state, dark_pool_ratio)
        state = await l5_acoustic_parser.run(state, cross_asset_data)

        # Determine mode and size multiplier
        from helpers import determine_bot_mode
        state.active_mode = determine_bot_mode(state.d_eff, state.gbp, state.phase)
        state.signal_size_multiplier = compute_size_multiplier(state)

        return state

    def _get_most_recent(self, data_dict: dict, date) -> Optional[object]:
        """Get most recent data entry at or before date."""
        if not data_dict:
            return None
        dates = [d for d in data_dict.keys() if d <= date]
        if not dates:
            return None
        return data_dict[max(dates)]

    def _update_portfolio_pnl(self, portfolio, prices_df, date):
        """Update unrealised P&L for open positions using current prices."""
        for pos in portfolio.open_positions:
            if pos.instrument in prices_df.columns:
                current_price = float(prices_df.loc[date, pos.instrument])
                pos.update_pnl(current_price)

    def _check_stops(self, state, portfolio) -> list[dict]:
        """Check structural stops for all positions."""
        from psibot.execution.stops import evaluate_stops
        return evaluate_stops(state, portfolio)

    def _close_trade(self, portfolio, exit_info, date, prices_df):
        """Close a position and record the trade."""
        for pos in portfolio.open_positions:
            if pos.position_id == exit_info["position_id"]:
                current_price = pos.current_price
                pos.close(exit_info["reasons"][0], current_price)
                trade = BacktestTrade(
                    signal_class=pos.signal_class,
                    entry_date=pos.entry_timestamp,
                    exit_date=date,
                    entry_price=pos.entry_price,
                    exit_price=current_price,
                    entry_gbp=pos.entry_gbp,
                    entry_d_eff=pos.entry_d_eff,
                    notional_usd=pos.notional_usd,
                    pnl_usd=pos.realised_pnl_usd,
                    exit_reason="; ".join(exit_info["reasons"]),
                    is_open=False,
                )
                self._trades.append(trade)
                portfolio.account_equity += pos.realised_pnl_usd
                break

    def _generate_paper_signals(self, state, portfolio, date, prices_df):
        """Generate paper trade entries."""
        from psibot.signals.soliton import check_soliton_signal
        from psibot.state.portfolio_state import Position
        from psibot.execution.sizing import size_order

        sig = check_soliton_signal(state, portfolio)
        if sig and portfolio.position_count < CCDR_THRESHOLDS["MAX_POSITIONS"]:
            notional = size_order(sig, state, portfolio, self.config)
            if notional > 0:
                import uuid
                primary_instrument = sig["instruments"][0] if sig["instruments"] else "SPX_FUT"
                entry_price = 100.0  # placeholder
                if primary_instrument in prices_df.columns:
                    entry_price = float(prices_df.loc[date, primary_instrument])
                pos = Position(
                    position_id=str(uuid.uuid4()),
                    signal_class=sig["signal_class"],
                    direction=sig["direction"],
                    instrument=primary_instrument,
                    entry_timestamp=date,
                    entry_gbp=state.gbp,
                    entry_phase=state.phase,
                    entry_psi_shape=state.psi_shape,
                    entry_d_eff=state.d_eff,
                    entry_price=entry_price,
                    current_price=entry_price,
                    notional_usd=notional,
                    size_multiplier=sig["size_multiplier"],
                )
                portfolio.add_position(pos)

    def _compute_sharpe(self, equity_series: list[float], periods_per_year: int = 252) -> float:
        """Annualised Sharpe ratio from daily equity series."""
        if len(equity_series) < 10:
            return 0.0
        returns = np.diff(equity_series) / np.array(equity_series[:-1])
        if np.std(returns) < 1e-10:
            return 0.0
        return float(np.mean(returns) / np.std(returns) * np.sqrt(periods_per_year))

    def _compute_max_drawdown(self, equity_series: list[float]) -> float:
        """Maximum drawdown as fraction of peak equity."""
        if len(equity_series) < 2:
            return 0.0
        arr = np.array(equity_series)
        peak = np.maximum.accumulate(arr)
        drawdown = (arr - peak) / (peak + 1e-10)
        return float(drawdown.min())


class _SimpleCAData:
    """Minimal cross-asset data wrapper for backtest pipeline."""
    def __init__(self, returns_matrix, prices):
        self.returns_matrix = returns_matrix
        self.prices = prices
        self.momentum_20d = 0.01
        self.momentum_60d = 0.02
        self.breadth = 0.6
        self.volume_ratio = 1.0
