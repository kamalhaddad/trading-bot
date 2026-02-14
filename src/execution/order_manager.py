"""Order management and execution."""

import asyncio
from dataclasses import dataclass, field
from datetime import datetime
from typing import Callable
from uuid import uuid4

from ..broker.base import (
    BaseBroker,
    Order,
    OrderSide,
    OrderType,
    OrderStatus,
    TimeInForce,
)
from ..strategies.base import Signal, SignalType
from ..risk.risk_manager import RiskManager, TradeRisk
from ..utils.logger import get_logger, trade_logger


logger = get_logger(__name__)


@dataclass
class OrderTicket:
    """Internal order tracking."""
    id: str
    signal: Signal
    risk_assessment: TradeRisk
    broker_order_id: str | None = None
    status: str = "pending"
    created_at: datetime = field(default_factory=datetime.now)
    filled_at: datetime | None = None
    fill_price: float | None = None
    filled_qty: float = 0
    error: str | None = None


class OrderManager:
    """Manages order execution and tracking."""

    def __init__(
        self,
        broker: BaseBroker,
        risk_manager: RiskManager,
        slippage_buffer: float = 0.0001,  # 0.01% slippage buffer for limits
        order_timeout: int = 30,  # Seconds before canceling unfilled orders
    ):
        self.broker = broker
        self.risk_manager = risk_manager
        self.slippage_buffer = slippage_buffer
        self.order_timeout = order_timeout

        # Order tracking
        self._pending_orders: dict[str, OrderTicket] = {}
        self._completed_orders: list[OrderTicket] = []

        # Callbacks
        self._fill_callbacks: list[Callable[[OrderTicket], None]] = []
        self._cancel_callbacks: list[Callable[[OrderTicket], None]] = []

    async def start(self) -> None:
        """Start order manager and subscribe to order updates."""
        await self.broker.subscribe_order_updates(self._on_order_update)
        logger.info("Order manager started")

    async def execute_signal(
        self,
        signal: Signal,
        risk_assessment: TradeRisk,
    ) -> OrderTicket:
        """Execute a trading signal.

        Args:
            signal: Trading signal to execute
            risk_assessment: Risk assessment with position sizing

        Returns:
            OrderTicket for tracking
        """
        ticket = OrderTicket(
            id=str(uuid4()),
            signal=signal,
            risk_assessment=risk_assessment,
        )

        if not risk_assessment.approved:
            ticket.status = "rejected"
            ticket.error = risk_assessment.rejection_reason
            logger.warning(
                "Order rejected by risk manager",
                symbol=signal.symbol,
                reason=risk_assessment.rejection_reason,
            )
            return ticket

        try:
            # Determine order side
            side = OrderSide.BUY if signal.signal_type == SignalType.BUY else OrderSide.SELL

            # Use limit order with slippage buffer for better fills
            if signal.signal_type == SignalType.BUY:
                limit_price = signal.price * (1 + self.slippage_buffer)
            else:
                limit_price = signal.price * (1 - self.slippage_buffer)

            # Round to 2 decimal places
            limit_price = round(limit_price, 2)

            # Generate client order ID
            client_order_id = f"{signal.strategy_name}_{ticket.id[:8]}"

            # Submit order
            order = await self.broker.submit_order(
                symbol=signal.symbol,
                side=side,
                qty=risk_assessment.shares,
                order_type=OrderType.LIMIT,
                time_in_force=TimeInForce.DAY,
                limit_price=limit_price,
                client_order_id=client_order_id,
            )

            ticket.broker_order_id = order.id
            ticket.status = "submitted"
            self._pending_orders[order.id] = ticket

            trade_logger.log_order(
                order_id=order.id,
                symbol=signal.symbol,
                side=side.value,
                qty=risk_assessment.shares,
                order_type="limit",
                price=limit_price,
                status="submitted",
            )

            # Start timeout task
            asyncio.create_task(self._order_timeout(order.id))

            logger.info(
                "Order submitted",
                order_id=order.id,
                symbol=signal.symbol,
                side=side.value,
                qty=risk_assessment.shares,
                limit_price=limit_price,
            )

        except Exception as e:
            ticket.status = "error"
            ticket.error = str(e)
            logger.error(
                "Order submission failed",
                symbol=signal.symbol,
                error=str(e),
            )

        return ticket

    async def execute_exit(
        self,
        symbol: str,
        qty: float,
        signal: Signal | None = None,
    ) -> OrderTicket | None:
        """Execute an exit order.

        Args:
            symbol: Symbol to exit
            qty: Quantity to sell
            signal: Optional exit signal

        Returns:
            OrderTicket if order submitted
        """
        if signal is None:
            # Create a basic exit signal
            from ..data.market_data import MarketDataManager
            signal = Signal(
                symbol=symbol,
                signal_type=SignalType.SELL,
                strategy_name="exit",
                confidence=1.0,
                price=0,  # Will use market order
                timestamp=datetime.now(),
                reason="Exit requested",
            )

        ticket = OrderTicket(
            id=str(uuid4()),
            signal=signal,
            risk_assessment=TradeRisk(
                approved=True,
                position_size=0,
                shares=int(qty),
                risk_amount=0,
                risk_pct=0,
            ),
        )

        try:
            # Use market order for exits to ensure fill
            order = await self.broker.submit_order(
                symbol=symbol,
                side=OrderSide.SELL,
                qty=qty,
                order_type=OrderType.MARKET,
                time_in_force=TimeInForce.DAY,
            )

            ticket.broker_order_id = order.id
            ticket.status = "submitted"
            self._pending_orders[order.id] = ticket

            trade_logger.log_order(
                order_id=order.id,
                symbol=symbol,
                side="sell",
                qty=qty,
                order_type="market",
                status="submitted",
            )

            logger.info(
                "Exit order submitted",
                order_id=order.id,
                symbol=symbol,
                qty=qty,
            )

            return ticket

        except Exception as e:
            ticket.status = "error"
            ticket.error = str(e)
            logger.error(
                "Exit order failed",
                symbol=symbol,
                error=str(e),
            )
            return ticket

    async def cancel_order(self, order_id: str) -> bool:
        """Cancel an order.

        Args:
            order_id: Broker order ID

        Returns:
            True if cancelled successfully
        """
        success = await self.broker.cancel_order(order_id)

        if success and order_id in self._pending_orders:
            ticket = self._pending_orders.pop(order_id)
            ticket.status = "cancelled"
            self._completed_orders.append(ticket)

            for callback in self._cancel_callbacks:
                try:
                    callback(ticket)
                except Exception as e:
                    logger.error("Error in cancel callback", error=str(e))

        return success

    async def cancel_all_orders(self) -> int:
        """Cancel all pending orders.

        Returns:
            Number of orders cancelled
        """
        count = await self.broker.cancel_all_orders()

        # Mark all pending as cancelled
        for order_id, ticket in list(self._pending_orders.items()):
            ticket.status = "cancelled"
            self._completed_orders.append(ticket)

        self._pending_orders.clear()

        logger.info("Cancelled all orders", count=count)
        return count

    def on_fill(self, callback: Callable[[OrderTicket], None]) -> None:
        """Register a fill callback."""
        self._fill_callbacks.append(callback)

    def on_cancel(self, callback: Callable[[OrderTicket], None]) -> None:
        """Register a cancel callback."""
        self._cancel_callbacks.append(callback)

    def get_pending_orders(self) -> list[OrderTicket]:
        """Get all pending orders."""
        return list(self._pending_orders.values())

    def get_pending_order(self, symbol: str) -> OrderTicket | None:
        """Get pending order for a symbol."""
        for ticket in self._pending_orders.values():
            if ticket.signal.symbol == symbol:
                return ticket
        return None

    def _on_order_update(self, order: Order) -> None:
        """Handle order status updates from broker."""
        if order.id not in self._pending_orders:
            return

        ticket = self._pending_orders[order.id]

        if order.status == OrderStatus.FILLED:
            ticket.status = "filled"
            ticket.filled_at = datetime.now()
            ticket.fill_price = order.filled_avg_price
            ticket.filled_qty = order.filled_qty

            # Move to completed
            del self._pending_orders[order.id]
            self._completed_orders.append(ticket)

            trade_logger.log_fill(
                order_id=order.id,
                symbol=order.symbol,
                side=order.side.value,
                qty=order.filled_qty,
                fill_price=order.filled_avg_price or 0,
            )

            # Notify callbacks
            for callback in self._fill_callbacks:
                try:
                    callback(ticket)
                except Exception as e:
                    logger.error("Error in fill callback", error=str(e))

        elif order.status == OrderStatus.PARTIALLY_FILLED:
            ticket.filled_qty = order.filled_qty
            logger.info(
                "Order partially filled",
                order_id=order.id,
                filled=order.filled_qty,
                remaining=order.qty - order.filled_qty,
            )

        elif order.status in (OrderStatus.CANCELED, OrderStatus.REJECTED, OrderStatus.EXPIRED):
            ticket.status = order.status.value
            del self._pending_orders[order.id]
            self._completed_orders.append(ticket)

            for callback in self._cancel_callbacks:
                try:
                    callback(ticket)
                except Exception as e:
                    logger.error("Error in cancel callback", error=str(e))

    async def _order_timeout(self, order_id: str) -> None:
        """Cancel order after timeout."""
        await asyncio.sleep(self.order_timeout)

        if order_id in self._pending_orders:
            ticket = self._pending_orders[order_id]

            # Only cancel if not filled
            if ticket.status == "submitted":
                logger.info(
                    "Cancelling order due to timeout",
                    order_id=order_id,
                    symbol=ticket.signal.symbol,
                )
                await self.cancel_order(order_id)
