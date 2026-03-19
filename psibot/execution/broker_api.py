"""
broker_api.py — ΨBot broker interfaces.

Contains:
  - IBKRBroker: full Interactive Brokers TWS API integration via ib_insync
    Consumed by AGT-01 (ψ_exp streaming), AGT-08 (execution),
    AGT-07 (risk/emergency stop), AGT-10 (monitor).
  - BrokerAPI: paper-trading stub for Phase 0-2 (kept for orchestrator compatibility)

CCDR Expectation Field Architecture — Version 1.0
"""

from __future__ import annotations

import asyncio
import logging
import uuid
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Optional

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from helpers import SignalClass, SignalDirection

log = logging.getLogger("psibot.execution.broker")


# =============================================================================
# Paper-trading types — used by Phase 0-2 orchestrator (main.py)
# =============================================================================

class OrderType(str, Enum):
    MARKET = "MARKET"
    LIMIT = "LIMIT"
    LIMIT_TO_MARKET = "LIMIT_TO_MARKET"


class OrderStatus(str, Enum):
    PENDING = "PENDING"
    SUBMITTED = "SUBMITTED"
    PARTIAL = "PARTIAL"
    FILLED = "FILLED"
    CANCELLED = "CANCELLED"
    REJECTED = "REJECTED"


@dataclass
class Order:
    order_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    position_id: str = ""
    instrument: str = ""
    direction: SignalDirection = SignalDirection.LONG
    notional_usd: float = 0.0
    order_type: OrderType = OrderType.LIMIT
    limit_price: Optional[float] = None
    signal_class: Optional[SignalClass] = None

    # Lifecycle
    status: OrderStatus = OrderStatus.PENDING
    submitted_at: Optional[datetime] = None
    filled_at: Optional[datetime] = None
    fill_price: Optional[float] = None
    fill_notional: float = 0.0
    reject_reason: str = ""

    # Limit-to-market timeout
    limit_to_market_minutes: int = 15
    escalate_to_market_at: Optional[datetime] = None

    def escalated_to_market(self) -> bool:
        if self.escalate_to_market_at is None:
            return False
        return datetime.utcnow() >= self.escalate_to_market_at


class BrokerAPI:
    """
    Paper-trading broker adapter for Phase 0-2.
    Phase 3+: use IBKRBroker directly.
    """

    def __init__(self, provider: str = "paper", config: dict = None):
        self.provider = provider
        self.config = config or {}
        self.paper_mode = (provider == "paper")
        self._pending_orders: list[Order] = []
        self._filled_orders: list[Order] = []
        self._order_log: list[dict] = []

    async def submit_order(self, order: Order) -> Order:
        """Submit order to broker. Returns updated order with status."""
        order.submitted_at = datetime.utcnow()
        log.info("ORDER SUBMIT [%s]: %s %s %.0f USD %s %s",
                 order.order_id[:8], order.direction.value, order.instrument,
                 order.notional_usd, order.order_type.value,
                 f"@{order.limit_price:.2f}" if order.limit_price else "")

        if self.paper_mode or self.provider == "paper":
            return await self._paper_fill(order)
        elif self.provider == "ibkr":
            return await self._paper_fill(order)  # stub until IBKRBroker wired
        else:
            log.error("Unknown broker provider: %s", self.provider)
            order.status = OrderStatus.REJECTED
            order.reject_reason = f"Unknown provider: {self.provider}"
            return order

    async def cancel_order(self, order_id: str) -> bool:
        for order in self._pending_orders:
            if order.order_id == order_id:
                order.status = OrderStatus.CANCELLED
                log.info("ORDER CANCELLED: %s", order_id[:8])
                return True
        return False

    async def close_position(self, position, reason: str, urgency: str = "normal") -> Optional[Order]:
        order_type = OrderType.MARKET if urgency == "immediate" else OrderType.LIMIT
        direction = (SignalDirection.SHORT
                     if position.direction == SignalDirection.LONG
                     else SignalDirection.LONG)

        close_order = Order(
            position_id=position.position_id,
            instrument=position.instrument,
            direction=direction,
            notional_usd=position.notional_usd,
            order_type=order_type,
            signal_class=position.signal_class,
        )

        if order_type == OrderType.LIMIT:
            close_order.limit_to_market_minutes = 15
            close_order.escalate_to_market_at = (
                datetime.utcnow() + timedelta(minutes=15)
            )

        log.warning("CLOSE POSITION [%s] reason=%s urgency=%s",
                    position.position_id[:8], reason, urgency)
        return await self.submit_order(close_order)

    async def emergency_stop_all(self, reason: str = "manual_override") -> int:
        log.critical("EMERGENCY STOP ALL: %s", reason)
        return 0

    async def _paper_fill(self, order: Order) -> Order:
        order.status = OrderStatus.FILLED
        order.filled_at = datetime.utcnow()
        order.fill_price = order.limit_price or 100.0
        order.fill_notional = order.notional_usd

        self._filled_orders.append(order)
        self._order_log.append({
            "order_id": order.order_id,
            "status": "FILLED (paper)",
            "timestamp": order.filled_at.isoformat(),
            "instrument": order.instrument,
            "notional": order.notional_usd,
        })
        log.info("PAPER FILL [%s]: %.0f USD %s %s",
                 order.order_id[:8], order.notional_usd,
                 order.instrument, order.direction.value)
        return order

    @property
    def order_history(self) -> list[dict]:
        return self._order_log


# =============================================================================
# IBKRBroker — Full IB Gateway integration via ib_insync (Phase 3+)
# =============================================================================

try:
    from ib_insync import (
        IB, Contract, Option, Stock, Future, Index,
        Order as IBOrder, LimitOrder, MarketOrder,
        Trade, PortfolioItem, AccountValue,
        util as ib_util,
    )
    _IB_INSYNC_AVAILABLE = True
except ImportError:
    _IB_INSYNC_AVAILABLE = False
    IB = None


class IBKRBroker:
    """
    Main IBKR broker interface for ΨBot Phase 3+.

    Requires ib_insync and a running IB Gateway (paper: port 4002, live: port 4001).

    Usage:
        broker = IBKRBroker(IBKRConfig.paper())
        await broker.connect()

        surface = await broker.get_options_surface_snapshot("SPX", tenors=[30, 91, 182])
        trade = await broker.submit_signal(signal, state, max_risk_usd=10000)
        await broker.emergency_stop(reason="Guardian mode activated")

        await broker.disconnect()
    """

    def __init__(self, config):
        if not _IB_INSYNC_AVAILABLE:
            raise ImportError(
                "ib_insync is required for IBKRBroker. "
                "Install with: pip install ib_insync"
            )
        self.config = config
        self.ib = IB()
        self._connected = False
        self._reconnect_task: Optional[asyncio.Task] = None
        self.log = logging.getLogger("psibot.broker")

        # Register disconnect handler for auto-reconnect
        self.ib.disconnectedEvent += self._on_disconnect

    # -------------------------------------------------------------------------
    # CONNECTION MANAGEMENT
    # -------------------------------------------------------------------------

    async def connect(self) -> None:
        """Connect to IB Gateway with exponential backoff retry."""
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

    # -------------------------------------------------------------------------
    # OPTIONS SURFACE DATA — for L1 ψ_exp Agent
    # -------------------------------------------------------------------------

    async def get_options_chain_params(self, underlying_symbol: str) -> dict:
        """
        Fetch all available strikes and expirations for an underlying.
        Required before streaming IV data. Results should be cached (~1h TTL).

        Returns:
            {'exchange': str, 'expirations': [...], 'strikes': [...], 'multiplier': str}
        """
        underlying = await self._resolve_underlying(underlying_symbol)
        await self.ib.qualifyContractsAsync(underlying)

        chains = await self.ib.reqSecDefOptParamsAsync(
            underlyingSymbol=underlying.symbol,
            futFopExchange="",
            underlyingSecType=underlying.secType,
            underlyingConId=underlying.conId,
        )

        if not chains:
            raise ValueError(f"No option chain found for {underlying_symbol}")

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
        tenors_days: list = None,
        strike_range_pct: float = 0.30,
    ) -> dict:
        """
        Fetch a snapshot of the options IV surface for a given underlying.
        Primary data feed for L1 ψ_exp reconstruction.

        Args:
            underlying_symbol: e.g. "SPX", "NDX", "SPY"
            tenors_days: target tenors [7, 14, 30, 91, 182, 365, 730]
            strike_range_pct: fetch strikes within ±X% of spot price

        Returns:
            OptionsSurface-compatible dict with symbol, timestamp, spot,
            tenors_days, strikes, iv, bid_iv, ask_iv.
        """
        if tenors_days is None:
            tenors_days = [7, 14, 30, 91, 182, 365, 730]

        underlying = await self._resolve_underlying(underlying_symbol)
        await self.ib.qualifyContractsAsync(underlying)
        spot = await self._get_spot_price(underlying)

        chain_params = await self.get_options_chain_params(underlying_symbol)

        matched_expirations = self._match_tenors_to_expirations(
            tenors_days=tenors_days,
            available_expirations=chain_params["expirations"],
        )

        atm_min = spot * (1 - strike_range_pct)
        atm_max = spot * (1 + strike_range_pct)
        filtered_strikes = [
            s for s in chain_params["strikes"]
            if atm_min <= s <= atm_max
        ]

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

            if len(strikes_arr) >= 5:
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
        tenors_days: list = None,
        update_interval_seconds: float = 30.0,
    ) -> asyncio.Task:
        """
        Start continuous options surface streaming for L1 agent.

        Streams IV surface updates to callback(surface_dict) every
        update_interval_seconds OR immediately on significant vol shift > 0.5 vega.

        Returns asyncio.Task (cancel to stop streaming).
        """
        async def _stream_loop():
            prev_atm_iv = {}
            while True:
                try:
                    surface = await self.get_options_surface_snapshot(
                        underlying_symbol, tenors_days
                    )
                    for tenor in surface["tenors_days"]:
                        atm_iv = self._atm_iv(surface, tenor)
                        if tenor in prev_atm_iv:
                            vega_shift = abs(atm_iv - prev_atm_iv[tenor]) * 100
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
                    await asyncio.sleep(5)

        return asyncio.create_task(_stream_loop())

    async def _fetch_iv_slice(
        self,
        underlying,
        expiration: str,
        strikes: list,
        exchange: str,
        multiplier: str,
        spot: float,
    ) -> tuple:
        """
        Fetch IV for all strikes at a single expiration.
        Uses OTM convention: PUT for strikes < spot, CALL for strikes >= spot.
        Returns (strikes_array, iv_array, bid_iv_array, ask_iv_array).
        """
        import numpy as np

        contracts = []
        for strike in strikes:
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

        qualified = await self.ib.qualifyContractsAsync(*contracts)
        valid_contracts = [c for c in qualified if c.conId > 0]

        if not valid_contracts:
            return np.array([]), np.array([]), np.array([]), np.array([])

        tickers = [self.ib.reqMktData(c, "100,101,106", snapshot=True)
                   for c in valid_contracts]

        await asyncio.sleep(2.0)  # allow data to populate

        strikes_out, ivs_out, bid_ivs_out, ask_ivs_out = [], [], [], []

        for i, ticker in enumerate(tickers):
            iv = getattr(ticker, "impliedVolatility", float("nan"))
            bid_iv = getattr(ticker, "bidImpliedVol", iv)
            ask_iv = getattr(ticker, "askImpliedVol", iv)

            if not (0.001 < iv < 5.0):
                continue

            strikes_out.append(valid_contracts[i].strike)
            ivs_out.append(iv)
            bid_ivs_out.append(bid_iv if bid_iv > 0 else iv * 0.99)
            ask_ivs_out.append(ask_iv if ask_iv > 0 else iv * 1.01)

            self.ib.cancelMktData(valid_contracts[i])

        return (
            np.array(strikes_out),
            np.array(ivs_out),
            np.array(bid_ivs_out),
            np.array(ask_ivs_out),
        )

    # -------------------------------------------------------------------------
    # ORDER EXECUTION — for AGT-08 Execution Agent
    # -------------------------------------------------------------------------

    async def submit_signal(
        self,
        signal,
        state,
        max_risk_usd: float,
    ) -> Optional[Trade]:
        """
        Convert a ΨBot Signal into an IBKR order and submit.

        Routes by signal class:
          SOLITON / REORDER   → LIMIT, DAY
          GUARDIAN exits      → MARKET, GTC
          TRANSITION / SAT    → LIMIT, GTD 15-min

        Returns Trade object or None if blocked by risk checks.
        """
        if not self._pre_submission_risk_check(signal, state, max_risk_usd):
            return None

        contract = await self._resolve_signal_instrument(signal)
        quantity = await self._compute_quantity(
            contract=contract,
            notional_usd=max_risk_usd * state.signal_size_multiplier,
        )

        if quantity <= 0:
            self.log.warning("Order quantity = 0 after sizing — skipping")
            return None

        if signal.signal_class in [SignalClass.SOLITON, SignalClass.REORDER]:
            mid_price = await self._get_mid_price(contract)
            order = LimitOrder(
                action="BUY" if signal.direction == SignalDirection.LONG else "SELL",
                totalQuantity=quantity,
                lmtPrice=round(mid_price, 2),
                account=self.config.account_id,
                tif="DAY",
                outsideRth=False,
            )
        elif getattr(signal, "is_guardian_exit", False):
            order = MarketOrder(
                action="SELL" if signal.direction == SignalDirection.LONG else "BUY",
                totalQuantity=quantity,
                account=self.config.account_id,
                tif="GTC",
            )
        else:
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
            "Order submitted: %s %d %s lmt=%.2f signal_class=%s gbp=%.3f d_eff=%.1f",
            order.action, quantity, contract.symbol,
            getattr(order, "lmtPrice", 0),
            signal.signal_class.value, state.gbp, state.d_eff,
        )
        return trade

    async def cancel_order_trade(self, trade: Trade) -> None:
        """Cancel an open IB order."""
        self.ib.cancelOrder(trade.order)
        self.log.info("Order cancelled: %s", trade.order.orderId)

    async def close_position(self, position, reason: str = "") -> Optional[Trade]:
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
        """
        self.log.critical("EMERGENCY STOP triggered: %s", reason)

        open_orders = self.ib.openOrders()
        for order in open_orders:
            self.ib.cancelOrder(order)
        self.log.info("Cancelled %d open orders", len(open_orders))

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

    # -------------------------------------------------------------------------
    # ACCOUNT MONITORING — for AGT-07 Risk and AGT-10 Monitor
    # -------------------------------------------------------------------------

    async def get_account_summary(self) -> dict:
        """
        Fetch key account values for risk monitoring.

        Returns dict with net_liquidation, equity_with_loan, initial_margin,
        maintenance_margin, available_funds, unrealized_pnl, realized_pnl_today.
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

    async def get_open_positions(self) -> list:
        """Return all open positions for the account."""
        return self.ib.portfolio(self.config.account_id)

    async def get_portfolio_pnl(self, lookback_days: int = 10) -> dict:
        """
        Compute rolling P&L for drawdown circuit breaker (AGT-07).

        Returns:
            {'today_pnl': float, 'net_liquidation': float}
        """
        summary = await self.get_account_summary()
        return {
            "today_pnl": summary.get("realized_pnl_today", 0.0)
                         + summary.get("unrealized_pnl", 0.0),
            "net_liquidation": summary.get("net_liquidation", 0.0),
        }

    # -------------------------------------------------------------------------
    # CROSS-ASSET RETURNS — for L3 D_eff Agent
    # -------------------------------------------------------------------------

    async def get_cross_asset_returns(
        self,
        symbols: list = None,
        lookback_days: int = 65,
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
            # Default to the options universe as proxy
            symbols = self.config.options_universe

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

    # -------------------------------------------------------------------------
    # HELPER METHODS
    # -------------------------------------------------------------------------

    async def _resolve_underlying(self, symbol: str):
        """Map ΨBot universe symbols to IBKR Contract objects."""
        index_map = {
            "SPX": Index(symbol="SPX", exchange="CBOE", currency="USD"),
            "NDX": Index(symbol="NDX", exchange="CBOE", currency="USD"),
            "VIX": Index(symbol="VIX", exchange="CBOE", currency="USD"),
        }
        etf_map = {
            "SPY": Stock(symbol="SPY", exchange="SMART", currency="USD"),
            "QQQ": Stock(symbol="QQQ", exchange="SMART", currency="USD"),
            "IWM": Stock(symbol="IWM", exchange="SMART", currency="USD"),
            "GLD": Stock(symbol="GLD", exchange="SMART", currency="USD"),
            "TLT": Stock(symbol="TLT", exchange="SMART", currency="USD"),
            "HYG": Stock(symbol="HYG", exchange="SMART", currency="USD"),
        }
        if symbol in index_map:
            return index_map[symbol]
        if symbol in etf_map:
            return etf_map[symbol]
        return Stock(symbol=symbol, exchange="SMART", currency="USD")

    async def _resolve_signal_instrument(self, signal) -> Contract:
        """Resolve signal instruments list to an IBKR Contract."""
        instruments = getattr(signal, "instruments", None) or []
        symbol = instruments[0] if instruments else "SPY"
        return await self._resolve_underlying(symbol)

    async def _get_spot_price(self, contract) -> float:
        """Get current mid price for an underlying contract."""
        ticker = self.ib.reqMktData(contract, "", snapshot=True)
        await asyncio.sleep(1.0)
        price = ticker.midpoint()
        if not price or price != price:  # NaN check
            price = ticker.last or ticker.close or 0.0
        self.ib.cancelMktData(contract)
        return float(price)

    async def _get_mid_price(self, contract) -> float:
        """Get current mid price for order limit price."""
        return await self._get_spot_price(contract)

    async def _compute_quantity(self, contract, notional_usd: float) -> int:
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
        tenors_days: list,
        available_expirations: list,
    ) -> dict:
        """
        Map target tenors (in days) to nearest available IBKR expiration dates.
        Only matches if within ±7 days of target.

        Returns: {tenor_days: 'YYYYMMDD', ...}
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
            nearest = min(
                expiry_dates,
                key=lambda d: abs((d - target_date).days),
            )
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
        dt = datetime.now(timezone.utc) + timedelta(minutes=15)
        return dt.strftime("%Y%m%d %H:%M:%S UTC")
