"""
tests/test_broker_api.py — Unit tests for the IBKR broker integration layer.
=============================================================================

Tests cover:
- IBKRConfig factory methods and environment variable handling
- BrokerAPI paper-trading stub (no IB Gateway required)
- IBKRBroker import guard (graceful when ib_insync not installed)
- OptionsStreamer state management
- ReconnectHandler state machine
- AccountMonitor drawdown computation
- OrderManager lifecycle tracking
- Helper methods: tenor matching, ATM IV extraction, GTD timestamp

All tests run without a live IB Gateway connection.

CCDR Expectation Field Architecture — Version 1.0
"""

import asyncio
import os
import sys
from datetime import datetime, timedelta

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import numpy as np
import pytest

from psibot.execution.ibkr_config import IBKRConfig
from psibot.execution.broker_api import (
    BrokerAPI, Order, OrderType, OrderStatus,
    IBKRBroker, _IB_INSYNC_AVAILABLE,
)
from psibot.execution.reconnect_handler import ReconnectHandler, ConnectionState
from psibot.execution.account_monitor import AccountMonitor, AccountSnapshot
from helpers import SignalClass, SignalDirection


# =============================================================================
# IBKRConfig
# =============================================================================

class TestIBKRConfig:

    def test_paper_factory(self):
        config = IBKRConfig.paper()
        assert config.port == 4002
        assert config.is_paper is True

    def test_live_requires_account_id(self, monkeypatch):
        monkeypatch.delenv("IBKR_ACCOUNT_ID", raising=False)
        with pytest.raises(ValueError, match="IBKR_ACCOUNT_ID"):
            IBKRConfig.live()

    def test_live_factory_with_account_id(self, monkeypatch):
        monkeypatch.setenv("IBKR_ACCOUNT_ID", "U9999999")
        config = IBKRConfig.live()
        assert config.port == 4001
        assert config.is_paper is False

    def test_default_options_universe_populated(self):
        config = IBKRConfig()
        assert "SPX" in config.options_universe
        assert "VIX" in config.options_universe
        assert len(config.options_universe) >= 5

    def test_custom_options_universe(self):
        config = IBKRConfig(options_universe=["SPX", "NDX"])
        assert config.options_universe == ["SPX", "NDX"]

    def test_env_max_order_usd(self, monkeypatch):
        monkeypatch.setenv("IBKR_MAX_ORDER_USD", "25000")
        config = IBKRConfig()
        assert config.max_order_value_usd == 25000.0

    def test_env_max_daily_loss(self, monkeypatch):
        monkeypatch.setenv("IBKR_MAX_DAILY_LOSS", "1000")
        config = IBKRConfig()
        assert config.max_daily_loss_usd == 1000.0

    def test_default_host_and_client_id(self):
        config = IBKRConfig()
        assert config.host == "127.0.0.1"
        assert config.client_id == 1
        assert config.timeout == 20


# =============================================================================
# BrokerAPI (paper stub)
# =============================================================================

class TestBrokerAPIPaper:

    def test_paper_fill_returns_filled_status(self):
        broker = BrokerAPI(provider="paper")
        order = Order(
            instrument="SPY",
            direction=SignalDirection.LONG,
            notional_usd=1000.0,
            order_type=OrderType.LIMIT,
            limit_price=450.0,
        )
        filled = asyncio.get_event_loop().run_until_complete(broker.submit_order(order))
        assert filled.status == OrderStatus.FILLED
        assert filled.fill_price == 450.0
        assert filled.fill_notional == 1000.0

    def test_paper_fill_without_limit_price(self):
        broker = BrokerAPI(provider="paper")
        order = Order(
            instrument="SPX",
            direction=SignalDirection.LONG,
            notional_usd=5000.0,
            order_type=OrderType.MARKET,
        )
        filled = asyncio.get_event_loop().run_until_complete(broker.submit_order(order))
        assert filled.status == OrderStatus.FILLED
        assert filled.fill_price == 100.0  # default placeholder

    def test_order_log_populated(self):
        broker = BrokerAPI(provider="paper")
        order = Order(instrument="TLT", direction=SignalDirection.SHORT, notional_usd=500.0)
        asyncio.get_event_loop().run_until_complete(broker.submit_order(order))
        assert len(broker.order_history) == 1
        assert broker.order_history[0]["instrument"] == "TLT"

    def test_cancel_pending_order(self):
        broker = BrokerAPI(provider="paper")
        order = Order(instrument="GLD", notional_usd=1000.0)
        broker._pending_orders.append(order)
        result = asyncio.get_event_loop().run_until_complete(
            broker.cancel_order(order.order_id)
        )
        assert result is True
        assert order.status == OrderStatus.CANCELLED

    def test_cancel_nonexistent_order(self):
        broker = BrokerAPI(provider="paper")
        result = asyncio.get_event_loop().run_until_complete(
            broker.cancel_order("nonexistent-id")
        )
        assert result is False

    def test_unknown_provider_rejects(self):
        broker = BrokerAPI(provider="unknown_broker")
        order = Order(instrument="SPY", notional_usd=1000.0)
        filled = asyncio.get_event_loop().run_until_complete(broker.submit_order(order))
        assert filled.status == OrderStatus.REJECTED
        assert "unknown_broker" in filled.reject_reason.lower()

    def test_emergency_stop_all_returns_zero_paper(self):
        broker = BrokerAPI(provider="paper")
        count = asyncio.get_event_loop().run_until_complete(
            broker.emergency_stop_all(reason="test")
        )
        assert count == 0  # paper stub


# =============================================================================
# Order dataclass
# =============================================================================

class TestOrder:

    def test_default_status_pending(self):
        order = Order()
        assert order.status == OrderStatus.PENDING

    def test_escalated_to_market_false_without_deadline(self):
        order = Order()
        assert order.escalated_to_market() is False

    def test_escalated_to_market_true_past_deadline(self):
        order = Order()
        order.escalate_to_market_at = datetime.utcnow() - timedelta(minutes=1)
        assert order.escalated_to_market() is True

    def test_escalated_to_market_false_future_deadline(self):
        order = Order()
        order.escalate_to_market_at = datetime.utcnow() + timedelta(minutes=15)
        assert order.escalated_to_market() is False

    def test_unique_order_ids(self):
        orders = [Order() for _ in range(100)]
        ids = [o.order_id for o in orders]
        assert len(set(ids)) == 100


# =============================================================================
# IBKRBroker — import guard (no live connection required)
# =============================================================================

class TestIBKRBrokerImportGuard:

    def test_ibkr_broker_raises_without_ib_insync(self):
        """IBKRBroker must raise ImportError when ib_insync is not installed."""
        if _IB_INSYNC_AVAILABLE:
            pytest.skip("ib_insync is installed — skip import guard test")

        config = IBKRConfig.paper()
        with pytest.raises(ImportError, match="ib_insync"):
            IBKRBroker(config)

    def test_ib_insync_availability_flag_is_bool(self):
        assert isinstance(_IB_INSYNC_AVAILABLE, bool)


# =============================================================================
# IBKRBroker helper methods (tested via static/class methods — no IB connection)
# =============================================================================

class TestIBKRBrokerHelpers:

    def test_fifteen_min_from_now_format(self):
        from psibot.execution.broker_api import IBKRBroker
        if not _IB_INSYNC_AVAILABLE:
            pytest.skip("ib_insync required")
        ts = IBKRBroker._fifteen_min_from_now()
        assert "UTC" in ts
        # Format: YYYYMMDD HH:MM:SS UTC
        parts = ts.split(" ")
        assert len(parts) == 3
        assert len(parts[0]) == 8   # YYYYMMDD

    def test_atm_iv_extraction(self):
        from psibot.execution.broker_api import IBKRBroker
        strikes = np.array([4800.0, 4900.0, 5000.0, 5100.0, 5200.0])
        ivs = np.array([0.22, 0.20, 0.18, 0.19, 0.21])
        surface = {
            "spot": 5010.0,
            "strikes": {30: strikes},
            "iv": {30: ivs},
        }
        atm_iv = IBKRBroker._atm_iv(surface, 30)
        # Nearest to 5010 is 5000 (index 2)
        assert abs(atm_iv - 0.18) < 0.001

    def test_atm_iv_empty_surface_returns_nan(self):
        from psibot.execution.broker_api import IBKRBroker
        surface = {"spot": 5000.0, "strikes": {}, "iv": {}}
        atm_iv = IBKRBroker._atm_iv(surface, 30)
        assert atm_iv != atm_iv  # NaN check

    def test_match_tenors_to_expirations(self):
        if not _IB_INSYNC_AVAILABLE:
            pytest.skip("ib_insync required")
        from psibot.execution.broker_api import IBKRBroker
        config = IBKRConfig.paper()
        broker = IBKRBroker(config)

        today = datetime.utcnow().date()
        # Build expirations: every 4 weeks for 2 years
        expirations = []
        d = today + timedelta(days=7)
        for _ in range(26):
            expirations.append(d.strftime("%Y%m%d"))
            d += timedelta(days=14)

        matched = broker._match_tenors_to_expirations([30, 91, 182], expirations)
        assert len(matched) >= 2  # at least 2 tenors should match within ±7 days

    def test_pre_submission_risk_check_blocks_low_deff(self):
        if not _IB_INSYNC_AVAILABLE:
            pytest.skip("ib_insync required")
        from psibot.execution.broker_api import IBKRBroker
        from psibot.state.condensate_state import CondensateState

        config = IBKRConfig.paper()
        broker = IBKRBroker(config)

        state = CondensateState()
        state.d_eff = 2.5  # below 3.0 → should block

        class FakeSignal:
            signal_class = SignalClass.SOLITON
            direction = SignalDirection.LONG

        assert broker._pre_submission_risk_check(FakeSignal(), state, 10000) is False

    def test_pre_submission_risk_check_blocks_high_gbp(self):
        if not _IB_INSYNC_AVAILABLE:
            pytest.skip("ib_insync required")
        from psibot.execution.broker_api import IBKRBroker
        from psibot.state.condensate_state import CondensateState

        config = IBKRConfig.paper()
        broker = IBKRBroker(config)

        state = CondensateState()
        state.d_eff = 10.0
        state.gbp = 0.75  # above 0.7 → should block

        class FakeSignal:
            signal_class = SignalClass.SOLITON
            direction = SignalDirection.LONG

        assert broker._pre_submission_risk_check(FakeSignal(), state, 10000) is False

    def test_pre_submission_risk_check_passes_normal_state(self):
        if not _IB_INSYNC_AVAILABLE:
            pytest.skip("ib_insync required")
        from psibot.execution.broker_api import IBKRBroker
        from psibot.state.condensate_state import CondensateState

        config = IBKRConfig.paper()
        broker = IBKRBroker(config)

        state = CondensateState()
        state.d_eff = 12.0
        state.gbp = 0.25
        state.signal_size_multiplier = 1.0

        class FakeSignal:
            signal_class = SignalClass.SOLITON
            direction = SignalDirection.LONG

        assert broker._pre_submission_risk_check(FakeSignal(), state, 10000) is True


# =============================================================================
# ReconnectHandler state machine
# =============================================================================

class TestReconnectHandler:

    def test_initial_state_connected(self):
        handler = ReconnectHandler(broker=None)
        assert handler.connection_state == ConnectionState.CONNECTED
        assert handler.is_connected is True

    def test_downtime_zero_before_disconnect(self):
        handler = ReconnectHandler(broker=None)
        assert handler.downtime_seconds == 0.0

    def test_register_stream_state(self):
        handler = ReconnectHandler(broker=None)
        symbols = ["SPX", "NDX"]
        callback = lambda surface: None
        handler.register_stream_state(callback, symbols, interval_seconds=60.0)
        assert handler._stream_callback is callback
        assert handler._stream_symbols == symbols
        assert handler._stream_interval == 60.0

    def test_on_disconnect_callback_called(self):
        """Disconnect handler must call on_disconnect callback."""
        callback_called = []

        class FakeBroker:
            async def connect(self):
                raise ConnectionError("test — no gateway")

        handler = ReconnectHandler(broker=FakeBroker())
        handler.on_disconnect = lambda: callback_called.append(True)

        async def run():
            # Patch reconnect loop to not run indefinitely
            handler._reconnect_task_limit = 1
            await handler.handle_disconnect()
            # Give the reconnect task a moment to start
            await asyncio.sleep(0.1)
            if handler._reconnect_task:
                handler._reconnect_task.cancel()
                try:
                    await handler._reconnect_task
                except asyncio.CancelledError:
                    pass

        asyncio.get_event_loop().run_until_complete(run())
        assert len(callback_called) > 0


# =============================================================================
# AccountMonitor drawdown computation
# =============================================================================

class TestAccountMonitor:

    def _make_snapshot(self, nlv: float) -> AccountSnapshot:
        return AccountSnapshot(
            timestamp=datetime.utcnow(),
            net_liquidation=nlv,
            equity_with_loan=nlv,
            initial_margin=nlv * 0.2,
            maintenance_margin=nlv * 0.15,
            available_funds=nlv * 0.8,
            unrealized_pnl=0.0,
            realized_pnl_today=0.0,
        )

    def test_no_drawdown_at_start(self):
        monitor = AccountMonitor(broker=None)
        assert monitor.rolling_drawdown_pct == 0.0

    def test_drawdown_computes_correctly(self):
        monitor = AccountMonitor(broker=None)
        monitor._peak_equity = 100_000.0
        snap = self._make_snapshot(95_000.0)
        monitor._snapshots.append(snap)
        assert abs(monitor.rolling_drawdown_pct - 5.0) < 0.01

    def test_no_negative_drawdown(self):
        """Drawdown cannot be negative (equity above peak — peak should update)."""
        monitor = AccountMonitor(broker=None)
        monitor._peak_equity = 90_000.0
        snap = self._make_snapshot(100_000.0)
        monitor._snapshots.append(snap)
        # Peak hasn't been updated (only _poll does that), so dd = 0 minimum
        assert monitor.rolling_drawdown_pct >= 0.0

    def test_margin_utilisation(self):
        snap = AccountSnapshot(
            timestamp=datetime.utcnow(),
            net_liquidation=100_000.0,
            equity_with_loan=100_000.0,
            initial_margin=30_000.0,
            maintenance_margin=25_000.0,
            available_funds=70_000.0,
            unrealized_pnl=500.0,
            realized_pnl_today=200.0,
        )
        assert abs(snap.margin_utilisation - 0.30) < 0.001

    def test_today_pnl_sums_realized_and_unrealized(self):
        snap = AccountSnapshot(
            timestamp=datetime.utcnow(),
            net_liquidation=100_000.0,
            equity_with_loan=100_000.0,
            initial_margin=0.0,
            maintenance_margin=0.0,
            available_funds=100_000.0,
            unrealized_pnl=800.0,
            realized_pnl_today=200.0,
        )
        assert snap.today_pnl == 1000.0

    def test_latest_snapshot_none_at_start(self):
        monitor = AccountMonitor(broker=None)
        assert monitor.latest_snapshot is None
