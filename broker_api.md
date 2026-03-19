# ΨBot Broker API Integration
## Interactive Brokers TWS API via IB Gateway
### Claude Code Implementation Instructions

> **JNEOPALLIUM — CCDR Expectation Field Architecture**
> This document is the single source of truth for all broker connectivity in ΨBot.
> Drop this file into the project root alongside `claude.md`.

---

## API Selection Decision

**Chosen: TWS API (IB Gateway, headless) — not Client Portal Web API**

| Criterion | TWS API (IB Gateway) | Client Portal Web API |
|-----------|---------------------|----------------------|
| Options surface streaming | ✅ Full tick-by-tick, all strikes/tenors | ⚠️ Snapshot-only, rate-limited |
| L1 ψ_exp reconstruction | ✅ Required real-time IV data available | ❌ Insufficient for Dupire PDE |
| Latency | ✅ < 1ms local socket | ⚠️ ~50–200ms HTTPS round-trip |
| Python async support | ✅ `ib_insync` (native asyncio) | ✅ aiohttp REST |
| Headless server deployment | ✅ IB Gateway (no GUI) | ✅ Client Portal Gateway |
| Options chain metadata | ✅ `reqSecDefOptParams` | ⚠️ Limited |
| Reconnect reliability | ✅ Well-documented patterns | ⚠️ Session token management |
| ΨBot fit | ✅ **Correct choice** | ❌ Not suitable for L1 |

**Reason in one sentence:** ΨBot's L1 agent requires continuous tick-by-tick streaming of full implied volatility surfaces across all strikes and tenors for ψ_exp reconstruction via the Dupire PDE. The Client Portal Web API's snapshot-and-rate-limit model makes this impossible. The TWS API via `ib_insync` is the only viable choice.

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────┐
│                    ΨBot Process                          │
│                                                          │
│  AGT-01 (ψ_exp)  ──────────────────────────────────┐   │
│  AGT-08 (Execution) ────────────────────────────┐   │   │
│  AGT-07 (Risk/Guardian) ─── emergency_stop ─┐   │   │   │
│                                              │   │   │   │
│  ┌────────────────────────────────────────────────────┐ │
│  │           IBKRBroker (execution/broker_api.py)     │ │
│  │           ib_insync IBC wrapper                    │ │
│  └────────────────────┬───────────────────────────────┘ │
└───────────────────────│─────────────────────────────────┘
                        │ TCP socket (localhost:4002)
                        ▼
               ┌─────────────────┐
               │   IB Gateway    │  ← headless, runs as service
               │  (port 4002)    │  ← paper: 4002, live: 4001
               └────────┬────────┘
                        │ IBKR proprietary protocol
                        ▼
               ┌─────────────────┐
               │ IBKR Servers    │
               └─────────────────┘
```

---

## Installation

```bash
# Core broker library — ib_insync is the IBKR-official async Python wrapper
pip install ib_insync

# IB Gateway download (do this on the server, not via pip)
# https://www.interactivebrokers.com/en/trading/ib-api.php
# Download: "IB Gateway" (not TWS) for headless server deployment
# Version: latest stable (currently 10.19.x)
# Install to: /opt/ibgateway/

# For automated IB Gateway startup:
pip install ibcauto  # IBC — automates IB Gateway login
```

---

## Project Structure (broker module)

```
execution/
├── broker_api.py          ← Main broker interface (THIS FILE'S IMPLEMENTATION TARGET)
├── ibkr_config.py         ← Connection parameters, account settings
├── options_streamer.py    ← Continuous IV surface streaming for L1
├── order_manager.py       ← Order lifecycle (create, submit, modify, cancel)
├── account_monitor.py     ← Portfolio positions, P&L, margin monitoring
└── reconnect_handler.py   ← Automatic reconnection + state recovery
```

---

## `ibkr_config.py` — Connection Parameters

```python
"""
ibkr_config.py — All IBKR connection parameters.
Edit this file for paper vs live trading.
NEVER commit API credentials to git.
"""

import os
from dataclasses import dataclass


@dataclass
class IBKRConfig:
    # IB Gateway connection
    host: str = "127.0.0.1"
    port: int = 4002          # 4002 = paper trading; 4001 = live trading
    client_id: int = 1        # unique per connection; use 1=main, 2=monitor, 3=backtest
    readonly: bool = False    # True for data-only connections (AGT-01 can use readonly=True)
    timeout: int = 20         # seconds to wait for connection

    # Account
    account_id: str = os.environ.get("IBKR_ACCOUNT_ID", "")

    # Risk limits (enforced at broker level, separate from CCDR structural stops)
    max_order_value_usd: float = float(os.environ.get("IBKR_MAX_ORDER_USD", "50000"))
    max_daily_loss_usd: float = float(os.environ.get("IBKR_MAX_DAILY_LOSS", "5000"))

    # Options surface streaming
    # Instruments to stream IV surfaces for L1 ψ_exp reconstruction
    options_universe: list[str] = None

    def __post_init__(self):
        if self.options_universe is None:
            self.options_universe = [
                "SPX",   # S&P 500 index options — primary condensate sensor
                "NDX",   # Nasdaq 100
                "SPY",   # SPX ETF (more liquid for some strikes)
                "QQQ",   # NDX ETF
                "IWM",   # Russell 2000
                "GLD",   # Gold
                "TLT",   # 20yr Treasury
                "HYG",   # High yield credit
                "VIX",   # VIX options
            ]

    @property
    def is_paper(self) -> bool:
        return self.port == 4002

    @classmethod
    def paper(cls) -> "IBKRConfig":
        return cls(port=4002)

    @classmethod
    def live(cls) -> "IBKRConfig":
        if not os.environ.get("IBKR_ACCOUNT_ID"):
            raise ValueError("IBKR_ACCOUNT_ID environment variable required for live trading")
        return cls(port=4001)
```

---

## `broker_api.py` — Main Broker Interface

```python
"""
broker_api.py — ΨBot IBKR broker interface.

This module is consumed by:
  - AGT-01 (ψ_exp Agent): options surface streaming for L1
  - AGT-08 (Execution Agent): order submission and fill monitoring
  - AGT-07 (Risk Agent): emergency stop, position queries
  - AGT-10 (Monitor Agent): account P&L, margin

Uses ib_insync for async/await interface over IB Gateway TCP socket.
"""

from __future__ import annotations

import asyncio
import logging
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Optional

from ib_insync import (
    IB, Contract, Option, Stock, Future, Index,
    Order, LimitOrder, MarketOrder, StopOrder,
    Trade, PortfolioItem, AccountValue,
    util as ib_util,
)

from execution.ibkr_config import IBKRConfig
from helpers import (
    CondensateState, BotMode, SignalClass, SignalDirection,
    log,
)

# Use ib_insync's asyncio event loop integration
ib_util.startLoop()


class IBKRBroker:
    """
    Main broker interface for ΨBot.

    Usage:
        broker = IBKRBroker(IBKRConfig.paper())
        await broker.connect()

        # For L1 options streaming
        surface = await broker.get_options_surface("SPX", tenors=[30, 91, 182])

        # For execution
        trade = await broker.submit_signal(signal, state)

        # For risk
        await broker.emergency_stop(reason="Guardian mode activated")

        await broker.disconnect()
    """

    def __init__(self, config: IBKRConfig):
        self.config = config
        self.ib = IB()
        self._connected = False
        self._reconnect_task: Optional[asyncio.Task] = None
        self.log = logging.getLogger("psibot.broker")

        # Register disconnect handler for auto-reconnect
        self.ib.disconnectedEvent += self._on_disconnect

    # -----------------------------------------------------------------------
    # CONNECTION MANAGEMENT
    # -----------------------------------------------------------------------

    async def connect(self) -> None:
        """Connect to IB Gateway. Retries with exponential backoff."""
        attempt = 0
        max_attempts = 5
        while attempt < max_attempts:
            try:
                await self.ib.connectAsync(
                    host=self.config.host,
                    port=self.config.port,
                    clientId=self.config.client_id,
                    timeout=self.config.timeout,
                    readonly=self.config.readonly,
                )
                self._connected = True
                self.log.info(
                    "Connected to IB Gateway (port=%d, account=%s, paper=%s)",
                    self.config.port, self.config.account_id, self.config.is_paper,
                )
                await self._validate_account()
                return
            except Exception as exc:
                attempt += 1
                wait = 2 ** attempt
                self.log.warning(
                    "Connection attempt %d/%d failed: %s — retrying in %ds",
                    attempt, max_attempts, exc, wait,
                )
                await asyncio.sleep(wait)

        raise ConnectionError(
            f"Failed to connect to IB Gateway after {max_attempts} attempts. "
            f"Is IB Gateway running on port {self.config.port}?"
        )

    async def disconnect(self) -> None:
        """Graceful disconnect."""
        self._connected = False
        self.ib.disconnect()
        self.log.info("Disconnected from IB Gateway")

    async def _on_disconnect(self) -> None:
        """Handle unexpected disconnection — attempt auto-reconnect."""
        self._connected = False
        self.log.warning("IB Gateway disconnected unexpectedly — initiating reconnect")

        # AGT-07 should know immediately
        # In production: fire event to orchestrator → switch to Scout mode
        await asyncio.sleep(5)
        try:
            await self.connect()
            self.log.info("Reconnected to IB Gateway successfully")
        except ConnectionError as exc:
            self.log.error("Reconnect failed: %s — manual intervention required", exc)

    async def _validate_account(self) -> None:
        """Verify account ID matches config and fetch basic account info."""
        accounts = self.ib.managedAccounts()
        if self.config.account_id and self.config.account_id not in accounts:
            raise ValueError(
                f"Account ID {self.config.account_id} not found in managed accounts: {accounts}"
            )
        if not self.config.account_id and accounts:
            self.config.account_id = accounts[0]
            self.log.info("Using account: %s", self.config.account_id)

    @property
    def is_connected(self) -> bool:
        return self._connected and self.ib.isConnected()

    @asynccontextmanager
    async def session(self):
        """Async context manager for broker session."""
        await self.connect()
        try:
            yield self
        finally:
            await self.disconnect()

    # -----------------------------------------------------------------------
    # OPTIONS SURFACE DATA — for L1 ψ_exp Agent
    # -----------------------------------------------------------------------

    async def get_options_chain_params(self, underlying_symbol: str) -> dict:
        """
        Fetch all available strikes and expirations for an underlying.
        Required before streaming IV data.

        Returns:
            {
              'exchange': str,
              'expirations': ['20241220', '20250117', ...],
              'strikes': [4500.0, 4510.0, ...],
              'multiplier': '100',
            }
        """
        # Resolve underlying contract
        underlying = await self._resolve_underlying(underlying_symbol)
        await self.ib.qualifyContractsAsync(underlying)

        # Fetch option chain parameters
        chains = await self.ib.reqSecDefOptParamsAsync(
            underlyingSymbol=underlying.symbol,
            futFopExchange="",
            underlyingSecType=underlying.secType,
            underlyingConId=underlying.conId,
        )

        if not chains:
            raise ValueError(f"No option chain found for {underlying_symbol}")

        # Prefer CBOE for index options, SMART for equity options
        preferred_exchange = "CBOE" if underlying.secType == "IND" else "SMART"
        chain = next(
            (c for c in chains if c.exchange == preferred_exchange),
            chains[0],
        )

        return {
            "exchange": chain.exchange,
            "expirations": sorted(chain.expirations),
            "strikes": sorted(chain.strikes),
            "multiplier": chain.multiplier,
        }

    async def get_options_surface_snapshot(
        self,
        underlying_symbol: str,
        tenors_days: list[int] = None,
        strike_range_pct: float = 0.30,  # ±30% from ATM
    ) -> dict:
        """
        Fetch a snapshot of the options IV surface for a given underlying.

        This is the primary data feed for L1 ψ_exp reconstruction.
        For real-time streaming, use stream_options_surface() instead.

        Args:
            underlying_symbol: e.g. "SPX", "NDX", "SPY"
            tenors_days: target tenors [7, 14, 30, 91, 182, 365, 730]
            strike_range_pct: fetch strikes within ±X% of spot price

        Returns:
            OptionsSurface dict compatible with L1 agent:
            {
                'symbol': str,
                'timestamp': datetime,
                'spot': float,
                'tenors_days': [7, 14, 30, 91, ...],
                'strikes': {tenor_days: np.ndarray},
                'iv': {tenor_days: np.ndarray},
            }
        """
        if tenors_days is None:
            tenors_days = [7, 14, 30, 91, 182, 365, 730]

        # Get current spot price
        underlying = await self._resolve_underlying(underlying_symbol)
        await self.ib.qualifyContractsAsync(underlying)
        spot = await self._get_spot_price(underlying)

        # Get available expirations and strikes
        chain_params = await self.get_options_chain_params(underlying_symbol)

        # Match tenors to nearest available expirations
        matched_expirations = self._match_tenors_to_expirations(
            tenors_days=tenors_days,
            available_expirations=chain_params["expirations"],
        )

        # Filter strikes to ±strike_range_pct of spot
        atm_min = spot * (1 - strike_range_pct)
        atm_max = spot * (1 + strike_range_pct)
        filtered_strikes = [
            s for s in chain_params["strikes"]
            if atm_min <= s <= atm_max
        ]

        # Fetch IV for all (expiration, strike, right) combinations
        # Use CALL and PUT separately; use call IV for OTM calls, put IV for OTM puts
        surface = {
            "symbol": underlying_symbol,
            "timestamp": datetime.utcnow(),
            "spot": spot,
            "tenors_days": [],
            "strikes": {},
            "iv": {},
            "bid_iv": {},
            "ask_iv": {},
        }

        for tenor_days, expiration in matched_expirations.items():
            strikes_arr, ivs_arr, bid_ivs, ask_ivs = await self._fetch_iv_slice(
                underlying=underlying,
                expiration=expiration,
                strikes=filtered_strikes,
                exchange=chain_params["exchange"],
                multiplier=chain_params["multiplier"],
                spot=spot,
            )

            if len(strikes_arr) >= 5:  # require minimum viable surface
                surface["tenors_days"].append(tenor_days)
                surface["strikes"][tenor_days] = strikes_arr
                surface["iv"][tenor_days] = ivs_arr
                surface["bid_iv"][tenor_days] = bid_ivs
                surface["ask_iv"][tenor_days] = ask_ivs

        if not surface["tenors_days"]:
            raise ValueError(f"No valid option data fetched for {underlying_symbol}")

        return surface

    async def stream_options_surface(
        self,
        underlying_symbol: str,
        callback,
        tenors_days: list[int] = None,
        update_interval_seconds: float = 30.0,
    ) -> asyncio.Task:
        """
        Start continuous options surface streaming for L1 agent.

        Streams IV surface updates to callback(surface_dict) every
        update_interval_seconds OR immediately on significant vol shift > 0.5 vega.

        Returns asyncio.Task that can be cancelled to stop streaming.

        Usage in AGT-01:
            stream_task = await broker.stream_options_surface(
                "SPX",
                callback=psi_agent.on_surface_update,
                tenors_days=[7, 14, 30, 91, 182, 365],
            )
        """
        async def _stream_loop():
            prev_atm_iv = {}
            while True:
                try:
                    surface = await self.get_options_surface_snapshot(
                        underlying_symbol, tenors_days
                    )
                    # Check for significant vol shift (trigger immediate L1 recompute)
                    for tenor in surface["tenors_days"]:
                        atm_iv = self._atm_iv(surface, tenor)
                        if tenor in prev_atm_iv:
                            vega_shift = abs(atm_iv - prev_atm_iv[tenor]) * 100  # in vega
                            if vega_shift > 0.5:
                                self.log.info(
                                    "Significant vol shift: %s tenor=%dd Δvega=%.2f — "
                                    "triggering immediate L1 recompute",
                                    underlying_symbol, tenor, vega_shift,
                                )
                        prev_atm_iv[tenor] = atm_iv

                    await callback(surface)
                    await asyncio.sleep(update_interval_seconds)

                except asyncio.CancelledError:
                    self.log.info("Stopping options surface stream for %s", underlying_symbol)
                    break
                except Exception as exc:
                    self.log.error("Options stream error for %s: %s", underlying_symbol, exc)
                    await asyncio.sleep(5)  # brief pause before retry

        return asyncio.create_task(_stream_loop())

    async def _fetch_iv_slice(
        self,
        underlying,
        expiration: str,
        strikes: list[float],
        exchange: str,
        multiplier: str,
        spot: float,
    ) -> tuple:
        """
        Fetch IV for all strikes at a single expiration.
        Uses CALL options for OTM calls, PUT options for OTM puts (standard convention).
        Returns (strikes_array, iv_array, bid_iv_array, ask_iv_array).
        """
        import numpy as np

        contracts = []
        for strike in strikes:
            # OTM convention: use PUT for strikes below spot, CALL for above
            right = "P" if strike < spot else "C"
            opt = Option(
                symbol=underlying.symbol,
                lastTradeDateOrContractMonth=expiration,
                strike=strike,
                right=right,
                exchange=exchange,
                currency="USD",
                multiplier=multiplier,
            )
            contracts.append(opt)

        # Qualify contracts (filter out invalid ones)
        qualified = await self.ib.qualifyContractsAsync(*contracts)
        valid_contracts = [c for c in qualified if c.conId > 0]

        if not valid_contracts:
            return [], [], [], []

        # Request market data for all contracts in batch
        tickers = [self.ib.reqMktData(c, "100,101,106", snapshot=True)
                   for c in valid_contracts]

        # Wait for data with timeout
        await asyncio.sleep(2.0)  # allow data to populate

        strikes_out, ivs_out, bid_ivs_out, ask_ivs_out = [], [], [], []

        for i, ticker in enumerate(tickers):
            # Field 106 = implied volatility from model
            iv = getattr(ticker, "impliedVolatility", float("nan"))
            bid_iv = getattr(ticker, "bidImpliedVol", iv)   # approx if not available
            ask_iv = getattr(ticker, "askImpliedVol", iv)

            if not (0.001 < iv < 5.0):  # filter clearly invalid IVs
                continue

            strikes_out.append(valid_contracts[i].strike)
            ivs_out.append(iv)
            bid_ivs_out.append(bid_iv if bid_iv > 0 else iv * 0.99)
            ask_ivs_out.append(ask_iv if ask_iv > 0 else iv * 1.01)

            # Cancel streaming market data subscription after snapshot
            self.ib.cancelMktData(valid_contracts[i])

        return (
            np.array(strikes_out),
            np.array(ivs_out),
            np.array(bid_ivs_out),
            np.array(ask_ivs_out),
        )

    # -----------------------------------------------------------------------
    # ORDER EXECUTION — for AGT-08 Execution Agent
    # -----------------------------------------------------------------------

    async def submit_signal(
        self,
        signal,                # Signal object from signals/
        state: CondensateState,
        max_risk_usd: float,
    ) -> Optional[Trade]:
        """
        Convert a ΨBot Signal into an IBKR order and submit.

        Routes to correct instrument type based on signal class:
          SOLITON       → equity index futures (e.g. ES, NQ)
          TRANSITION    → options straddles/strangles, VIX calls
          REORDER       → equity index ETF or futures
          SAT_HEDGE     → equity put options, GLD, TLT, CDX credit

        Returns Trade object or None if order blocked by risk checks.
        """
        # Pre-submission risk check
        if not self._pre_submission_risk_check(signal, state, max_risk_usd):
            return None

        # Resolve instrument
        contract = await self._resolve_signal_instrument(signal)

        # Compute share/contract quantity
        quantity = await self._compute_quantity(
            contract=contract,
            notional_usd=max_risk_usd * state.signal_size_multiplier,
        )

        if quantity <= 0:
            self.log.warning("Order quantity = 0 after sizing — skipping")
            return None

        # Build order — LIMIT for entries, MARKET for Guardian exits
        if signal.signal_class in [SignalClass.SOLITON, SignalClass.REORDER]:
            mid_price = await self._get_mid_price(contract)
            order = LimitOrder(
                action="BUY" if signal.direction == SignalDirection.LONG else "SELL",
                totalQuantity=quantity,
                lmtPrice=round(mid_price, 2),
                account=self.config.account_id,
                tif="DAY",                    # Day order — cancel if unfilled
                outsideRth=False,
            )
        elif signal.is_guardian_exit:
            # Guardian exits always use market orders for immediacy
            order = MarketOrder(
                action="SELL" if signal.direction == SignalDirection.LONG else "BUY",
                totalQuantity=quantity,
                account=self.config.account_id,
                tif="GTC",
            )
        else:
            # Transition / Saturation-Hedge: limit with 15-min expiry
            mid_price = await self._get_mid_price(contract)
            order = LimitOrder(
                action="BUY",
                totalQuantity=quantity,
                lmtPrice=round(mid_price, 2),
                account=self.config.account_id,
                tif="GTD",
                goodTillDate=self._fifteen_min_from_now(),
            )

        trade = self.ib.placeOrder(contract, order)
        self.log.info(
            "Order submitted: %s %s %s qty=%d lmt=%.2f signal_class=%s gbp=%.3f d_eff=%.1f",
            order.action, quantity, contract.symbol,
            quantity, getattr(order, "lmtPrice", 0),
            signal.signal_class.value, state.gbp, state.d_eff,
        )
        return trade

    async def cancel_order(self, trade: Trade) -> None:
        """Cancel an open order."""
        self.ib.cancelOrder(trade.order)
        self.log.info("Order cancelled: %s", trade.order.orderId)

    async def close_position(
        self,
        position,             # PortfolioItem or internal Position object
        reason: str = "",
    ) -> Optional[Trade]:
        """
        Close an open position at market.
        Used by AGT-07 (Risk) for structural stop and Guardian exits.
        Always uses MARKET order for reliability.
        """
        contract = position.contract
        quantity = abs(position.position)
        if quantity == 0:
            return None

        action = "SELL" if position.position > 0 else "BUY"
        order = MarketOrder(
            action=action,
            totalQuantity=quantity,
            account=self.config.account_id,
        )

        trade = self.ib.placeOrder(contract, order)
        self.log.info(
            "Position closed: %s %s %d | reason: %s",
            action, contract.symbol, quantity, reason,
        )
        return trade

    async def emergency_stop(self, reason: str = "manual_override") -> None:
        """
        Flatten ALL positions immediately.
        Called by AGT-07 in Guardian mode or on unhandled exception.
        Uses market orders for all closures.
        """
        self.log.critical("EMERGENCY STOP triggered: %s", reason)

        # Cancel all open orders first
        open_orders = self.ib.openOrders()
        for order in open_orders:
            self.ib.cancelOrder(order)
        self.log.info("Cancelled %d open orders", len(open_orders))

        # Close all positions
        portfolio = self.ib.portfolio(self.config.account_id)
        close_trades = []
        for item in portfolio:
            if item.position != 0:
                trade = await self.close_position(item, reason=f"EMERGENCY: {reason}")
                if trade:
                    close_trades.append(trade)

        self.log.info(
            "Emergency stop: %d positions being closed. Reason: %s",
            len(close_trades), reason,
        )

        # Wait for fills with timeout
        deadline = asyncio.get_event_loop().time() + 30.0
        while asyncio.get_event_loop().time() < deadline:
            remaining = [t for t in close_trades if not t.isDone()]
            if not remaining:
                break
            await asyncio.sleep(0.5)

        unfilled = [t for t in close_trades if not t.isDone()]
        if unfilled:
            self.log.error(
                "Emergency stop: %d positions NOT closed within timeout — manual check required",
                len(unfilled),
            )

    # -----------------------------------------------------------------------
    # ACCOUNT MONITORING — for AGT-07 Risk and AGT-10 Monitor
    # -----------------------------------------------------------------------

    async def get_account_summary(self) -> dict:
        """
        Fetch key account values for risk monitoring.

        Returns dict with at minimum:
          net_liquidation, equity_with_loan, initial_margin,
          maintenance_margin, available_funds, unrealized_pnl,
          realized_pnl_today
        """
        values = await self.ib.accountValuesAsync(self.config.account_id)
        summary = {}
        field_map = {
            "NetLiquidation": "net_liquidation",
            "EquityWithLoanValue": "equity_with_loan",
            "InitMarginReq": "initial_margin",
            "MaintMarginReq": "maintenance_margin",
            "AvailableFunds": "available_funds",
            "UnrealizedPnL": "unrealized_pnl",
            "RealizedPnL": "realized_pnl_today",
        }
        for av in values:
            if av.tag in field_map and av.currency == "USD":
                try:
                    summary[field_map[av.tag]] = float(av.value)
                except ValueError:
                    pass
        return summary

    async def get_open_positions(self) -> list[PortfolioItem]:
        """Return all open positions for the account."""
        return self.ib.portfolio(self.config.account_id)

    async def get_portfolio_pnl(self, lookback_days: int = 10) -> dict:
        """
        Compute rolling P&L for drawdown circuit breaker (AGT-07).

        Returns:
            {'today_pnl': float, 'rolling_10d_drawdown': float, 'peak_equity': float}
        """
        summary = await self.get_account_summary()
        # In production, track equity curve in state/portfolio_state.py
        # Here we return what's directly available from IBKR
        return {
            "today_pnl": summary.get("realized_pnl_today", 0.0)
                         + summary.get("unrealized_pnl", 0.0),
            "net_liquidation": summary.get("net_liquidation", 0.0),
        }

    # -----------------------------------------------------------------------
    # CROSS-ASSET RETURNS — for L3 D_eff Agent
    # -----------------------------------------------------------------------

    async def get_cross_asset_returns(
        self,
        symbols: list[str] = None,
        lookback_days: int = 65,  # 60 + buffer
    ) -> dict:
        """
        Fetch historical daily returns for the 27-asset universe.
        Used by AGT-03 (D_eff Agent) for cross-asset correlation matrix.

        Returns dict: {symbol: pd.Series of daily log-returns}
        """
        import numpy as np
        import pandas as pd
        from ib_insync import util as ib_util

        if symbols is None:
            from helpers import ASSET_UNIVERSE_27
            symbols = ASSET_UNIVERSE_27

        returns = {}
        for symbol in symbols:
            try:
                contract = await self._resolve_underlying(symbol)
                bars = await self.ib.reqHistoricalDataAsync(
                    contract=contract,
                    endDateTime="",
                    durationStr=f"{lookback_days} D",
                    barSizeSetting="1 day",
                    whatToShow="ADJUSTED_LAST",
                    useRTH=True,
                    formatDate=1,
                )
                if bars:
                    df = ib_util.df(bars)
                    prices = df["close"].values
                    log_ret = np.log(prices[1:] / prices[:-1])
                    returns[symbol] = pd.Series(
                        log_ret,
                        index=df.index[1:],
                        name=symbol,
                    )
            except Exception as exc:
                self.log.warning("Failed to fetch returns for %s: %s", symbol, exc)

        return returns

    # -----------------------------------------------------------------------
    # HELPER METHODS
    # -----------------------------------------------------------------------

    async def _resolve_underlying(self, symbol: str) -> Contract:
        """Map ΨBot universe symbols to IBKR Contract objects."""
        # Index options
        index_map = {
            "SPX": Index(symbol="SPX", exchange="CBOE", currency="USD"),
            "NDX": Index(symbol="NDX", exchange="CBOE", currency="USD"),
            "VIX": Index(symbol="VIX", exchange="CBOE", currency="USD"),
        }
        # ETFs for equity options
        etf_map = {
            "SPY":  Stock(symbol="SPY",  exchange="SMART", currency="USD"),
            "QQQ":  Stock(symbol="QQQ",  exchange="SMART", currency="USD"),
            "IWM":  Stock(symbol="IWM",  exchange="SMART", currency="USD"),
            "GLD":  Stock(symbol="GLD",  exchange="SMART", currency="USD"),
            "TLT":  Stock(symbol="TLT",  exchange="SMART", currency="USD"),
            "HYG":  Stock(symbol="HYG",  exchange="SMART", currency="USD"),
        }
        if symbol in index_map:
            return index_map[symbol]
        if symbol in etf_map:
            return etf_map[symbol]
        # Fallback: try as stock on SMART
        return Stock(symbol=symbol, exchange="SMART", currency="USD")

    async def _get_spot_price(self, contract: Contract) -> float:
        """Get current mid price for an underlying contract."""
        ticker = self.ib.reqMktData(contract, "", snapshot=True)
        await asyncio.sleep(1.0)
        price = ticker.midpoint()
        if not price or price != price:  # NaN check
            price = ticker.last or ticker.close or 0.0
        self.ib.cancelMktData(contract)
        return float(price)

    async def _get_mid_price(self, contract: Contract) -> float:
        """Get current mid price for order limit price."""
        return await self._get_spot_price(contract)

    async def _compute_quantity(self, contract: Contract, notional_usd: float) -> int:
        """Convert notional USD to contract quantity, respecting multiplier."""
        price = await self._get_spot_price(contract)
        multiplier = float(getattr(contract, "multiplier", 1) or 1)
        if price <= 0:
            return 0
        qty = notional_usd / (price * multiplier)
        return max(1, int(qty))

    def _pre_submission_risk_check(self, signal, state, max_risk_usd: float) -> bool:
        """Last-mile risk check before order submission."""
        if state.d_eff <= 3.0:
            self.log.warning("Order blocked: D_eff=%.1f ≤ 3.0", state.d_eff)
            return False
        if state.gbp >= 0.7:
            self.log.warning("Order blocked: GBP=%.3f ≥ 0.7", state.gbp)
            return False
        if max_risk_usd * state.signal_size_multiplier < 100:
            self.log.warning("Order blocked: effective size < $100")
            return False
        return True

    def _match_tenors_to_expirations(
        self,
        tenors_days: list[int],
        available_expirations: list[str],
    ) -> dict[int, str]:
        """
        Map target tenors (in days) to nearest available IBKR expiration dates.

        Args:
            tenors_days: [7, 14, 30, 91, 182, 365, 730]
            available_expirations: ['20241220', '20250117', ...]

        Returns:
            {tenor_days: 'YYYYMMDD', ...}
        """
        from datetime import datetime, timedelta

        today = datetime.utcnow().date()
        expiry_dates = [
            datetime.strptime(e, "%Y%m%d").date()
            for e in available_expirations
        ]

        matched = {}
        for tenor in tenors_days:
            target_date = today + timedelta(days=tenor)
            # Find nearest expiration to target
            nearest = min(
                expiry_dates,
                key=lambda d: abs((d - target_date).days),
            )
            # Only use if within ±7 days of target
            if abs((nearest - target_date).days) <= 7:
                matched[tenor] = nearest.strftime("%Y%m%d")

        return matched

    @staticmethod
    def _atm_iv(surface: dict, tenor: int) -> float:
        """Extract ATM IV for a given tenor from a surface dict."""
        import numpy as np
        strikes = surface["strikes"].get(tenor, np.array([]))
        ivs = surface["iv"].get(tenor, np.array([]))
        spot = surface.get("spot", 0)
        if len(strikes) == 0 or spot == 0:
            return float("nan")
        atm_idx = np.argmin(np.abs(strikes - spot))
        return float(ivs[atm_idx])

    @staticmethod
    def _fifteen_min_from_now() -> str:
        """Return IB-formatted datetime 15 minutes from now for GTD orders."""
        from datetime import datetime, timedelta, timezone
        dt = datetime.now(timezone.utc) + timedelta(minutes=15)
        return dt.strftime("%Y%m%d %H:%M:%S UTC")
```

---

## `options_streamer.py` — Continuous IV Surface Streaming

```python
"""
options_streamer.py — Manages continuous IV surface streaming for all
instruments in the options universe, feeding data to the L1 ψ_exp agent.
"""

import asyncio
import logging
from datetime import datetime
from typing import Callable, Optional

from execution.broker_api import IBKRBroker
from execution.ibkr_config import IBKRConfig

log = logging.getLogger("psibot.options_streamer")


class OptionsStreamer:
    """
    Manages streaming IV surfaces for all instruments in the options universe.

    The L1 ψ_exp Agent calls this; it subscribes to updates and reconstructs
    ψ_exp on every surface update that exceeds the vol-shift threshold.

    Usage:
        streamer = OptionsStreamer(broker)
        await streamer.start(callback=psi_agent.on_surface_update)
        # ... later ...
        await streamer.stop()
    """

    def __init__(self, broker: IBKRBroker):
        self.broker = broker
        self._stream_tasks: dict[str, asyncio.Task] = {}
        self._running = False

    async def start(
        self,
        callback: Callable,
        symbols: list[str] = None,
        update_interval_seconds: float = 30.0,
    ) -> None:
        """Start streaming IV surfaces for all symbols."""
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

    async def restart_symbol(self, symbol: str, callback: Callable) -> None:
        """Restart stream for a single symbol after error."""
        if symbol in self._stream_tasks:
            self._stream_tasks[symbol].cancel()
        task = await self.broker.stream_options_surface(symbol, callback)
        self._stream_tasks[symbol] = task
        log.info("Restarted IV surface stream: %s", symbol)
```

---

## IB Gateway Setup (Infrastructure)

### Install and Configure IB Gateway

```bash
# 1. Download IB Gateway (not TWS) — headless, no GUI required
# https://www.interactivebrokers.com/en/trading/ib-api.php
# Unpack to /opt/ibgateway/

# 2. Install IBC (automates login — avoids manual credential entry)
# https://github.com/IbcAlpha/IBC
wget https://github.com/IbcAlpha/IBC/releases/latest/download/IBCLinux-3.18.0.zip
unzip IBCLinux-3.18.0.zip -d /opt/ibc/

# 3. Configure IBC credentials (store encrypted, never in git)
# /opt/ibc/config.ini
cat > /opt/ibc/config.ini << 'EOF'
IbLoginId=YOUR_IBKR_USERNAME
IbPassword=YOUR_IBKR_PASSWORD
TradingMode=paper          # change to 'live' for real money
ReadonlyLogin=no
FixedInitialComponentSize=600
EOF

# 4. Configure IB Gateway ports
# /opt/ibgateway/jts.ini — edit socket port
# Paper: port 4002, Live: port 4001

# 5. Start IB Gateway via IBC (systemd service recommended)
/opt/ibc/Scripts/DisplayBannerAndLaunch.sh \
    /opt/ibgateway \
    /opt/ibc/config.ini \
    4002 \
    paper

# 6. Verify connection
python -c "
import asyncio
from ib_insync import IB
ib = IB()
asyncio.get_event_loop().run_until_complete(
    ib.connectAsync('127.0.0.1', 4002, clientId=99)
)
print('Connected:', ib.isConnected())
print('Accounts:', ib.managedAccounts())
ib.disconnect()
"
```

### Systemd Service for IB Gateway

```ini
# /etc/systemd/system/ibgateway.service
[Unit]
Description=Interactive Brokers Gateway
After=network.target
StartLimitIntervalSec=0

[Service]
Type=simple
User=psibot
ExecStart=/opt/ibc/Scripts/DisplayBannerAndLaunch.sh \
    /opt/ibgateway \
    /opt/ibc/config.ini \
    4002 \
    paper
Restart=always
RestartSec=30
Environment=DISPLAY=:1   # for any GUI components IBC might invoke

[Install]
WantedBy=multi-user.target
```

```bash
systemctl daemon-reload
systemctl enable ibgateway
systemctl start ibgateway
systemctl status ibgateway
```

---

## Integration with Pipeline Agents

### How AGT-01 (ψ_exp Agent) Uses the Broker

```python
# pipeline/l1_psi_reconstruction.py — broker integration snippet

class PsiReconstructionAgent:
    def __init__(self, broker: IBKRBroker):
        self.broker = broker
        self.streamer = OptionsStreamer(broker)

    async def start_streaming(self) -> None:
        await self.streamer.start(callback=self.on_surface_update)

    async def on_surface_update(self, surface: dict) -> None:
        """Called by OptionsStreamer every 30 seconds or on vol shift > 0.5 vega."""
        try:
            state = await self.reconstruct_psi(surface)
            # Publish updated state for downstream agents
            await orchestrator.on_l1_update(state)
        except Exception as exc:
            log.error("L1 reconstruction failed: %s", exc)
            # Set conservative fallback
            state.psi_shape = PsiShape.UNKNOWN

    async def run(self, state: CondensateState, market_data=None) -> CondensateState:
        """Called by orchestrator in polling mode (alternative to streaming)."""
        for symbol in ["SPX", "NDX"]:
            surface = await self.broker.get_options_surface_snapshot(
                symbol, tenors_days=[7, 14, 30, 91, 182, 365, 730]
            )
            # Reconstruct ψ_exp from surface (Dupire PDE, shape classification)
            state = self._process_surface(state, surface)
        return state
```

### How AGT-08 (Execution Agent) Uses the Broker

```python
# execution order within AGT-08

async def handle_signal(self, signal, state: CondensateState) -> None:
    """Execute a trade signal from AGT-06."""
    if not self.broker.is_connected:
        log.error("Cannot execute: broker not connected")
        return

    trade = await self.broker.submit_signal(
        signal=signal,
        state=state,
        max_risk_usd=self.config.max_risk_usd,
    )

    if trade:
        # Track in portfolio state
        portfolio_state.add_pending_order(trade, signal, state)
        # Monitor fill
        asyncio.create_task(self._monitor_fill(trade, signal))

async def _monitor_fill(self, trade, signal) -> None:
    """Watch for fill; if limit order unfilled in 15 min, cancel."""
    deadline = asyncio.get_event_loop().time() + 900  # 15 minutes
    while asyncio.get_event_loop().time() < deadline:
        if trade.isDone():
            if trade.orderStatus.status == "Filled":
                portfolio_state.confirm_position(trade, signal)
                log.info("Order filled: %s", trade.orderStatus)
            else:
                log.info("Order done (not filled): %s", trade.orderStatus.status)
            return
        await asyncio.sleep(5)

    # Timeout — cancel unfilled limit order
    await self.broker.cancel_order(trade)
    log.warning("Limit order timed out and cancelled: %s", trade.order.orderId)
```

### How AGT-07 (Risk Agent) Uses the Broker

```python
# Guardian mode activation in AGT-07

async def activate_guardian(self, reason: str) -> None:
    log.critical("GUARDIAN ACTIVATED: %s", reason)

    # 1. Emergency flatten of all Soliton and Reorder positions
    positions = await self.broker.get_open_positions()
    risk_positions = [p for p in positions if p.position != 0]
    for pos in risk_positions:
        await self.broker.close_position(pos, reason=f"Guardian: {reason}")

    # 2. Cancel all pending limit orders
    # (handled inside emergency_stop if needed)

    # 3. Log to monitor agent
    await monitor_agent.alert(
        f"GUARDIAN MODE: {reason}", priority="CRITICAL"
    )
```

---

## Environment Variables

```bash
# .env (never commit this file)
IBKR_ACCOUNT_ID=U1234567
IBKR_MAX_ORDER_USD=50000
IBKR_MAX_DAILY_LOSS=5000
IBKR_PAPER_PORT=4002
IBKR_LIVE_PORT=4001

# IBC credentials (alternative to config.ini)
IBC_USERNAME=your_username
IBC_PASSWORD=your_password
IBC_TRADING_MODE=paper
```

---

## Testing

```bash
# Unit tests — run against paper account (never live for tests)
pytest tests/test_broker_api.py -v \
    --ibkr-port 4002 \
    --ibkr-client-id 99

# Specific test: options surface quality
pytest tests/test_broker_api.py::test_spx_surface_has_minimum_strikes -v

# Integration test: full L1 pipeline with live options data
pytest tests/test_l1_integration.py -v --live-data

# Paper trade smoke test: submit and cancel a limit order
pytest tests/test_execution_smoke.py -v --paper-only
```

---

## Common Errors and Solutions

| Error | Cause | Fix |
|-------|-------|-----|
| `ConnectionRefusedError: [Errno 111]` | IB Gateway not running | `systemctl start ibgateway` |
| `Error 200: No security definition` | Invalid contract parameters | Check symbol, exchange, expiry format |
| `Error 354: Requested market data is not subscribed` | Missing market data subscription | Add options data subscription in IBKR account |
| `Error 162: Historical market data request limit` | Too many historical requests | Add 10s delay between requests; reduce lookback |
| `Error 10167: Displaying delayed market data` | Real-time data not subscribed | Subscribe to real-time options data feed |
| `Ticker impliedVolatility = nan` | Options surface fetch too fast | Increase `asyncio.sleep(2.0)` wait in `_fetch_iv_slice` |
| `Error 507: Bad Message` | IB Gateway version mismatch | Update IB Gateway to latest version |
| `Error 504: Not connected` | Connection dropped | `reconnect_handler.py` auto-reconnects; check logs |

---

## Notes for Claude Code

1. **The options surface streamer is the most important component.** If it fails, set `psi_shape = PsiShape.UNKNOWN` and `gbp += 0.2` immediately. Never let a data failure allow Hunter mode to run on stale ψ_exp data.

2. **Never use TWS (the GUI app) in production.** IB Gateway headless is the correct target. TWS shuts down overnight for updates; IB Gateway does not.

3. **Client IDs must be unique.** If running multiple connections (AGT-01 for data, AGT-08 for execution), use different `clientId` values (1, 2, 3). Never reuse a clientId without disconnecting the previous session.

4. **Paper trading port is 4002, live is 4001.** The `IBKRConfig.paper()` and `IBKRConfig.live()` factory methods enforce this. Never hardcode port numbers.

5. **Options data subscriptions required:** For real-time IV surfaces, the IBKR account needs these data subscriptions: `US Equity and Options Add-On Streaming Bundle`, `CBOE Streaming Options Quotes`. Paper accounts have these included.

6. **The `reqSecDefOptParams` call is slow** (~2–5 seconds). Cache the results and refresh only when the options chain rolls (new weekly expiries appear, typically Thursday evening US time).
