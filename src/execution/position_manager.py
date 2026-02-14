"""Position tracking and management."""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from ..broker.base import BaseBroker, Position, Order
from ..strategies.base import Signal, SignalType
from ..utils.logger import get_logger, trade_logger


logger = get_logger(__name__)


@dataclass
class TrackedPosition:
    """Internally tracked position with additional metadata."""
    symbol: str
    qty: float
    entry_price: float
    entry_time: datetime
    strategy: str
    current_price: float = 0.0
    unrealized_pnl: float = 0.0
    unrealized_pnl_pct: float = 0.0
    highest_price: float = 0.0  # For trailing stop
    stop_loss: float | None = None
    take_profit: float | None = None
    trailing_stop_pct: float | None = None
    metadata: dict = field(default_factory=dict)


@dataclass
class ClosedPosition:
    """Record of a closed position."""
    symbol: str
    qty: float
    entry_price: float
    exit_price: float
    entry_time: datetime
    exit_time: datetime
    strategy: str
    pnl: float
    pnl_pct: float
    hold_time_seconds: float
    exit_reason: str


class PositionManager:
    """Manages position tracking and stop/target orders."""

    def __init__(
        self,
        broker: BaseBroker,
        trailing_stop_pct: float = 0.003,  # Default 0.3%
    ):
        self.broker = broker
        self.default_trailing_stop_pct = trailing_stop_pct

        # Position tracking
        self._positions: dict[str, TrackedPosition] = {}
        self._closed_positions: list[ClosedPosition] = []

    async def sync_with_broker(self) -> None:
        """Sync positions with broker."""
        broker_positions = await self.broker.get_positions()

        # Update existing positions
        for bp in broker_positions:
            if bp.symbol in self._positions:
                pos = self._positions[bp.symbol]
                pos.qty = bp.qty
                pos.current_price = bp.current_price
                pos.unrealized_pnl = bp.unrealized_pnl
                pos.unrealized_pnl_pct = bp.unrealized_pnl_pct

                # Update highest price for trailing stop
                if bp.current_price > pos.highest_price:
                    pos.highest_price = bp.current_price

        # Check for positions closed by broker
        broker_symbols = {bp.symbol for bp in broker_positions}
        for symbol in list(self._positions.keys()):
            if symbol not in broker_symbols:
                # Position was closed externally
                pos = self._positions.pop(symbol)
                logger.info(
                    "Position closed externally",
                    symbol=symbol,
                    entry_price=pos.entry_price,
                )

    def open_position(
        self,
        signal: Signal,
        fill_price: float,
        qty: float,
        stop_loss: float | None = None,
        take_profit: float | None = None,
        trailing_stop_pct: float | None = None,
    ) -> TrackedPosition:
        """Record a new position.

        Args:
            signal: Signal that triggered the position
            fill_price: Actual fill price
            qty: Quantity filled
            stop_loss: Stop loss price
            take_profit: Take profit price
            trailing_stop_pct: Trailing stop percentage

        Returns:
            TrackedPosition
        """
        position = TrackedPosition(
            symbol=signal.symbol,
            qty=qty,
            entry_price=fill_price,
            entry_time=datetime.now(),
            strategy=signal.strategy_name,
            current_price=fill_price,
            highest_price=fill_price,
            stop_loss=stop_loss or signal.suggested_stop_loss,
            take_profit=take_profit or signal.suggested_take_profit,
            trailing_stop_pct=trailing_stop_pct or self.default_trailing_stop_pct,
            metadata=signal.metadata,
        )

        self._positions[signal.symbol] = position

        trade_logger.log_position_opened(
            symbol=signal.symbol,
            qty=qty,
            entry_price=fill_price,
            strategy=signal.strategy_name,
        )

        logger.info(
            "Position opened",
            symbol=signal.symbol,
            qty=qty,
            entry_price=fill_price,
            strategy=signal.strategy_name,
            stop_loss=position.stop_loss,
            take_profit=position.take_profit,
        )

        return position

    def close_position(
        self,
        symbol: str,
        exit_price: float,
        exit_reason: str,
    ) -> ClosedPosition | None:
        """Record position closure.

        Args:
            symbol: Symbol being closed
            exit_price: Exit price
            exit_reason: Reason for exit

        Returns:
            ClosedPosition record
        """
        if symbol not in self._positions:
            return None

        pos = self._positions.pop(symbol)

        pnl = (exit_price - pos.entry_price) * pos.qty
        pnl_pct = (exit_price - pos.entry_price) / pos.entry_price
        hold_time = (datetime.now() - pos.entry_time).total_seconds()

        closed = ClosedPosition(
            symbol=symbol,
            qty=pos.qty,
            entry_price=pos.entry_price,
            exit_price=exit_price,
            entry_time=pos.entry_time,
            exit_time=datetime.now(),
            strategy=pos.strategy,
            pnl=pnl,
            pnl_pct=pnl_pct,
            hold_time_seconds=hold_time,
            exit_reason=exit_reason,
        )

        self._closed_positions.append(closed)

        trade_logger.log_position_closed(
            symbol=symbol,
            qty=pos.qty,
            entry_price=pos.entry_price,
            exit_price=exit_price,
            pnl=pnl,
            pnl_pct=pnl_pct,
            hold_time_seconds=hold_time,
            exit_reason=exit_reason,
        )

        logger.info(
            "Position closed",
            symbol=symbol,
            entry_price=pos.entry_price,
            exit_price=exit_price,
            pnl=pnl,
            pnl_pct=pnl_pct,
            hold_time=hold_time,
            exit_reason=exit_reason,
        )

        return closed

    def update_price(self, symbol: str, price: float) -> None:
        """Update current price for a position.

        Args:
            symbol: Symbol
            price: Current price
        """
        if symbol not in self._positions:
            return

        pos = self._positions[symbol]
        pos.current_price = price
        pos.unrealized_pnl = (price - pos.entry_price) * pos.qty
        pos.unrealized_pnl_pct = (price - pos.entry_price) / pos.entry_price

        # Update highest price for trailing stop
        if price > pos.highest_price:
            pos.highest_price = price

    def check_stops(self, symbol: str, price: float) -> dict[str, Any] | None:
        """Check if any stops are triggered.

        Args:
            symbol: Symbol to check
            price: Current price

        Returns:
            Dict with stop info if triggered, None otherwise
        """
        if symbol not in self._positions:
            return None

        pos = self._positions[symbol]
        self.update_price(symbol, price)

        # Check hard stop loss
        if pos.stop_loss and price <= pos.stop_loss:
            return {
                "type": "stop_loss",
                "price": price,
                "stop_price": pos.stop_loss,
                "pnl_pct": pos.unrealized_pnl_pct,
            }

        # Check take profit
        if pos.take_profit and price >= pos.take_profit:
            return {
                "type": "take_profit",
                "price": price,
                "target_price": pos.take_profit,
                "pnl_pct": pos.unrealized_pnl_pct,
            }

        # Check trailing stop
        if pos.trailing_stop_pct and pos.highest_price > pos.entry_price:
            trailing_stop_price = pos.highest_price * (1 - pos.trailing_stop_pct)

            # Only trigger if we're in profit and price drops
            if price <= trailing_stop_price and pos.unrealized_pnl_pct > 0:
                return {
                    "type": "trailing_stop",
                    "price": price,
                    "stop_price": trailing_stop_price,
                    "highest_price": pos.highest_price,
                    "pnl_pct": pos.unrealized_pnl_pct,
                }

        return None

    def get_position(self, symbol: str) -> TrackedPosition | None:
        """Get tracked position for a symbol."""
        return self._positions.get(symbol)

    def get_all_positions(self) -> list[TrackedPosition]:
        """Get all tracked positions."""
        return list(self._positions.values())

    def get_position_symbols(self) -> list[str]:
        """Get symbols with open positions."""
        return list(self._positions.keys())

    def has_position(self, symbol: str) -> bool:
        """Check if we have a position in a symbol."""
        return symbol in self._positions

    def get_closed_positions(self) -> list[ClosedPosition]:
        """Get all closed positions."""
        return self._closed_positions.copy()

    def get_daily_stats(self) -> dict[str, Any]:
        """Get daily position statistics."""
        if not self._closed_positions:
            return {
                "total_trades": 0,
                "winning_trades": 0,
                "losing_trades": 0,
                "win_rate": 0.0,
                "total_pnl": 0.0,
                "avg_pnl": 0.0,
                "avg_hold_time": 0.0,
            }

        winning = [p for p in self._closed_positions if p.pnl > 0]
        losing = [p for p in self._closed_positions if p.pnl <= 0]

        total_pnl = sum(p.pnl for p in self._closed_positions)
        avg_hold = sum(p.hold_time_seconds for p in self._closed_positions) / len(self._closed_positions)

        return {
            "total_trades": len(self._closed_positions),
            "winning_trades": len(winning),
            "losing_trades": len(losing),
            "win_rate": len(winning) / len(self._closed_positions) if self._closed_positions else 0,
            "total_pnl": total_pnl,
            "avg_pnl": total_pnl / len(self._closed_positions),
            "avg_hold_time": avg_hold,
            "best_trade": max(p.pnl for p in self._closed_positions) if self._closed_positions else 0,
            "worst_trade": min(p.pnl for p in self._closed_positions) if self._closed_positions else 0,
        }

    def reset_daily(self) -> None:
        """Reset daily tracking (keep positions, clear closed history)."""
        self._closed_positions.clear()
