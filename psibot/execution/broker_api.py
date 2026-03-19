"""
execution/broker_api.py — SK-22: Broker FIX API Integration
============================================================
Converts signal objects into exchange orders.
Manages fills, order lifecycle, and emergency stops.

Order types per signal class (from agents.md):
  - Soliton / Reorder entries: limit orders (patient fills)
  - Guardian exits: market orders (immediacy required)
  - Saturation hedges: limit → market after 15 min if unfilled
  - Transition: limit orders

Phase 0-2: No broker connection (paper trading mode).
Phase 3+: Real broker via FIX protocol or REST API.

CCDR Expectation Field Architecture — Version 1.0
"""

from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Optional

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from helpers import SignalClass, SignalDirection

log = logging.getLogger("psibot.execution.broker")


class OrderType(str, Enum):
    MARKET = "MARKET"
    LIMIT = "LIMIT"
    LIMIT_TO_MARKET = "LIMIT_TO_MARKET"  # limit expiring to market after timeout


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
    Broker integration adapter.
    In Phase 0-2: paper trading simulation.
    In Phase 3+: real broker via configured provider.
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

        if self.paper_mode:
            return await self._paper_fill(order)
        elif self.provider == "ibkr":
            return await self._submit_ibkr(order)
        elif self.provider == "fix":
            return await self._submit_fix(order)
        else:
            log.error("Unknown broker provider: %s", self.provider)
            order.status = OrderStatus.REJECTED
            order.reject_reason = f"Unknown provider: {self.provider}"
            return order

    async def cancel_order(self, order_id: str) -> bool:
        """Cancel a pending order."""
        for order in self._pending_orders:
            if order.order_id == order_id:
                order.status = OrderStatus.CANCELLED
                log.info("ORDER CANCELLED: %s", order_id[:8])
                return True
        return False

    async def close_position(
        self,
        position,
        reason: str,
        urgency: str = "normal",
    ) -> Optional[Order]:
        """
        Close an open position.
        urgency='immediate' → market order
        urgency='normal' → limit order
        """
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
        """
        Flatten ALL open positions immediately (market orders).
        Used by Guardian mode and emergency scripts.
        Returns count of orders submitted.
        """
        log.critical("EMERGENCY STOP ALL: %s", reason)
        # In production: submit market close for every open position
        # Implementation hooks into portfolio state
        return 0  # placeholder; full implementation in Phase 3

    async def _paper_fill(self, order: Order) -> Order:
        """
        Simulate order fill for paper trading (Phase 0-2).
        Assumes fill at limit price or market (immediate fill).
        """
        order.status = OrderStatus.FILLED
        order.filled_at = datetime.utcnow()
        order.fill_price = order.limit_price or 100.0  # placeholder price
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

    async def _submit_ibkr(self, order: Order) -> Order:
        """IBKR TWS API order submission (placeholder)."""
        log.warning("IBKR order submission not yet implemented — paper fill")
        return await self._paper_fill(order)

    async def _submit_fix(self, order: Order) -> Order:
        """FIX protocol order submission (placeholder for Phase 3)."""
        log.warning("FIX order submission not yet implemented — paper fill")
        return await self._paper_fill(order)

    @property
    def order_history(self) -> list[dict]:
        return self._order_log
