"""
order_manager.py — Order lifecycle management for ΨBot IBKR integration.

Tracks open orders, monitors fills, handles limit-to-market escalation,
and maintains order audit log.

CCDR Expectation Field Architecture — Version 1.0
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Callable, Optional

log = logging.getLogger("psibot.order_manager")

# Limit order timeout before cancellation (mirrors GTD 15-min setting)
LIMIT_ORDER_TIMEOUT_SECONDS = 900  # 15 minutes


@dataclass
class ManagedOrder:
    """Internal order tracking record."""
    ibkr_trade: object          # ib_insync Trade object
    signal_class: str
    direction: str
    symbol: str
    quantity: int
    notional_usd: float
    submitted_at: datetime = field(default_factory=datetime.utcnow)
    filled_at: Optional[datetime] = None
    cancel_deadline: Optional[datetime] = None  # for limit orders
    is_guardian_exit: bool = False

    @property
    def order_id(self) -> int:
        return self.ibkr_trade.order.orderId

    @property
    def is_done(self) -> bool:
        return self.ibkr_trade.isDone()

    @property
    def is_filled(self) -> bool:
        return self.ibkr_trade.orderStatus.status == "Filled"

    @property
    def fill_price(self) -> Optional[float]:
        avg = self.ibkr_trade.orderStatus.avgFillPrice
        return float(avg) if avg else None

    @property
    def age_seconds(self) -> float:
        return (datetime.utcnow() - self.submitted_at).total_seconds()

    @property
    def is_timed_out(self) -> bool:
        if self.cancel_deadline is None:
            return False
        return datetime.utcnow() >= self.cancel_deadline


class OrderManager:
    """
    Manages the lifecycle of all IBKR orders for ΨBot.

    Responsibilities:
    - Track all submitted orders
    - Monitor fills and update portfolio state
    - Cancel limit orders that exceed their timeout
    - Emit callbacks on fill / cancellation
    - Maintain audit log for compliance / debugging

    Usage:
        om = OrderManager(broker)
        om.on_fill = lambda order: portfolio.confirm_position(order)
        om.on_cancel = lambda order: log.warning("Unfilled: %s", order.symbol)

        await om.track(managed_order)
        await om.run_lifecycle()   # run in background task
    """

    def __init__(self, broker):
        self.broker = broker
        self._orders: dict[int, ManagedOrder] = {}  # order_id → ManagedOrder
        self._audit_log: list[dict] = []
        self._running = False

        # Callbacks — set by orchestrator / portfolio manager
        self.on_fill: Optional[Callable] = None
        self.on_cancel: Optional[Callable] = None
        self.on_reject: Optional[Callable] = None

    async def track(self, managed_order: ManagedOrder) -> None:
        """Register an order for lifecycle tracking."""
        self._orders[managed_order.order_id] = managed_order
        log.info(
            "Tracking order %d: %s %s %d @ %s",
            managed_order.order_id,
            managed_order.direction,
            managed_order.symbol,
            managed_order.quantity,
            managed_order.signal_class,
        )

    async def run_lifecycle(self, poll_interval_seconds: float = 5.0) -> None:
        """
        Background task — poll all tracked orders and handle state transitions.
        Run this as an asyncio.Task for the duration of the trading session.
        """
        self._running = True
        log.info("Order lifecycle manager started")

        while self._running:
            await self._poll_orders()
            await asyncio.sleep(poll_interval_seconds)

    async def stop(self) -> None:
        """Stop the lifecycle polling loop."""
        self._running = False
        log.info("Order lifecycle manager stopped")

    async def _poll_orders(self) -> None:
        """Check all pending orders; handle fills, cancellations, timeouts."""
        completed = []

        for order_id, order in self._orders.items():
            if order.is_done:
                if order.is_filled:
                    await self._handle_fill(order)
                else:
                    status = order.ibkr_trade.orderStatus.status
                    await self._handle_non_fill(order, status)
                completed.append(order_id)

            elif order.is_timed_out and not order.is_guardian_exit:
                log.warning(
                    "Limit order %d timed out after %.0fs — cancelling: %s %s",
                    order_id, order.age_seconds, order.symbol, order.signal_class,
                )
                await self.broker.cancel_order_trade(order.ibkr_trade)
                self._audit(order, "TIMEOUT_CANCELLED")
                if self.on_cancel:
                    await self._safe_callback(self.on_cancel, order)
                completed.append(order_id)

        for order_id in completed:
            self._orders.pop(order_id, None)

    async def _handle_fill(self, order: ManagedOrder) -> None:
        order.filled_at = datetime.utcnow()
        log.info(
            "Order FILLED: %d %s %s %d @ %.4f",
            order.order_id, order.direction, order.symbol,
            order.quantity, order.fill_price or 0,
        )
        self._audit(order, "FILLED")
        if self.on_fill:
            await self._safe_callback(self.on_fill, order)

    async def _handle_non_fill(self, order: ManagedOrder, status: str) -> None:
        log.info(
            "Order done (not filled): %d %s status=%s",
            order.order_id, order.symbol, status,
        )
        self._audit(order, status)
        if status in ("Cancelled", "Inactive") and self.on_cancel:
            await self._safe_callback(self.on_cancel, order)
        elif status in ("Rejected",) and self.on_reject:
            await self._safe_callback(self.on_reject, order)

    def _audit(self, order: ManagedOrder, event: str) -> None:
        self._audit_log.append({
            "timestamp": datetime.utcnow().isoformat(),
            "order_id": order.order_id,
            "event": event,
            "symbol": order.symbol,
            "direction": order.direction,
            "quantity": order.quantity,
            "signal_class": order.signal_class,
            "fill_price": order.fill_price,
            "age_seconds": round(order.age_seconds, 1),
        })

    async def _safe_callback(self, cb: Callable, order: ManagedOrder) -> None:
        try:
            result = cb(order)
            if asyncio.iscoroutine(result):
                await result
        except Exception as exc:
            log.error("Order callback error for %d: %s", order.order_id, exc)

    def submit_managed_order(
        self,
        ibkr_trade,
        signal_class: str,
        direction: str,
        symbol: str,
        quantity: int,
        notional_usd: float,
        is_guardian_exit: bool = False,
        limit_timeout_seconds: float = LIMIT_ORDER_TIMEOUT_SECONDS,
    ) -> ManagedOrder:
        """
        Factory: create a ManagedOrder from an ib_insync Trade and register it.

        Call this immediately after broker.submit_signal() returns.
        """
        is_limit = hasattr(ibkr_trade.order, "lmtPrice")
        cancel_deadline = None
        if is_limit and not is_guardian_exit:
            cancel_deadline = datetime.utcnow() + timedelta(seconds=limit_timeout_seconds)

        managed = ManagedOrder(
            ibkr_trade=ibkr_trade,
            signal_class=signal_class,
            direction=direction,
            symbol=symbol,
            quantity=quantity,
            notional_usd=notional_usd,
            cancel_deadline=cancel_deadline,
            is_guardian_exit=is_guardian_exit,
        )
        asyncio.get_event_loop().run_until_complete(self.track(managed))
        return managed

    @property
    def open_order_count(self) -> int:
        return len(self._orders)

    @property
    def audit_log(self) -> list[dict]:
        return self._audit_log
