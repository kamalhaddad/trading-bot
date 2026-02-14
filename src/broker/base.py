"""Abstract broker interface."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable


class OrderSide(str, Enum):
    """Order side."""
    BUY = "buy"
    SELL = "sell"


class OrderType(str, Enum):
    """Order type."""
    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"
    STOP_LIMIT = "stop_limit"
    TRAILING_STOP = "trailing_stop"


class OrderStatus(str, Enum):
    """Order status."""
    NEW = "new"
    PENDING = "pending_new"
    ACCEPTED = "accepted"
    FILLED = "filled"
    PARTIALLY_FILLED = "partially_filled"
    CANCELED = "canceled"
    REJECTED = "rejected"
    EXPIRED = "expired"


class TimeInForce(str, Enum):
    """Time in force."""
    DAY = "day"
    GTC = "gtc"
    IOC = "ioc"
    FOK = "fok"


@dataclass
class Order:
    """Order representation."""
    id: str
    symbol: str
    side: OrderSide
    qty: float
    order_type: OrderType
    status: OrderStatus
    time_in_force: TimeInForce = TimeInForce.DAY
    limit_price: float | None = None
    stop_price: float | None = None
    trail_percent: float | None = None
    filled_qty: float = 0.0
    filled_avg_price: float | None = None
    created_at: datetime | None = None
    filled_at: datetime | None = None
    client_order_id: str | None = None


@dataclass
class Position:
    """Position representation."""
    symbol: str
    qty: float
    avg_entry_price: float
    current_price: float
    market_value: float
    unrealized_pnl: float
    unrealized_pnl_pct: float
    side: str  # "long" or "short"


@dataclass
class AccountInfo:
    """Account information."""
    equity: float
    cash: float
    buying_power: float
    portfolio_value: float
    day_trade_count: int = 0
    pattern_day_trader: bool = False
    trading_blocked: bool = False
    account_blocked: bool = False


@dataclass
class Bar:
    """OHLCV bar data."""
    symbol: str
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: int
    vwap: float | None = None


@dataclass
class Quote:
    """Quote data."""
    symbol: str
    timestamp: datetime
    bid_price: float
    bid_size: int
    ask_price: float
    ask_size: int


@dataclass
class Trade:
    """Trade data."""
    symbol: str
    timestamp: datetime
    price: float
    size: int


class BaseBroker(ABC):
    """Abstract base class for broker implementations."""

    @abstractmethod
    async def connect(self) -> None:
        """Connect to the broker."""
        pass

    @abstractmethod
    async def disconnect(self) -> None:
        """Disconnect from the broker."""
        pass

    @abstractmethod
    async def get_account(self) -> AccountInfo:
        """Get account information."""
        pass

    @abstractmethod
    async def get_positions(self) -> list[Position]:
        """Get all open positions."""
        pass

    @abstractmethod
    async def get_position(self, symbol: str) -> Position | None:
        """Get position for a specific symbol."""
        pass

    @abstractmethod
    async def submit_order(
        self,
        symbol: str,
        side: OrderSide,
        qty: float,
        order_type: OrderType = OrderType.MARKET,
        time_in_force: TimeInForce = TimeInForce.DAY,
        limit_price: float | None = None,
        stop_price: float | None = None,
        trail_percent: float | None = None,
        client_order_id: str | None = None,
    ) -> Order:
        """Submit an order."""
        pass

    @abstractmethod
    async def cancel_order(self, order_id: str) -> bool:
        """Cancel an order."""
        pass

    @abstractmethod
    async def cancel_all_orders(self) -> int:
        """Cancel all open orders. Returns count of cancelled orders."""
        pass

    @abstractmethod
    async def get_order(self, order_id: str) -> Order | None:
        """Get order by ID."""
        pass

    @abstractmethod
    async def get_open_orders(self, symbol: str | None = None) -> list[Order]:
        """Get all open orders, optionally filtered by symbol."""
        pass

    @abstractmethod
    async def close_position(self, symbol: str) -> Order | None:
        """Close a position."""
        pass

    @abstractmethod
    async def close_all_positions(self) -> list[Order]:
        """Close all positions."""
        pass

    @abstractmethod
    async def get_bars(
        self,
        symbol: str,
        timeframe: str,
        start: datetime | None = None,
        end: datetime | None = None,
        limit: int = 100,
    ) -> list[Bar]:
        """Get historical bar data."""
        pass

    @abstractmethod
    async def get_latest_bar(self, symbol: str) -> Bar | None:
        """Get the latest bar for a symbol."""
        pass

    @abstractmethod
    async def get_latest_quote(self, symbol: str) -> Quote | None:
        """Get the latest quote for a symbol."""
        pass

    @abstractmethod
    async def subscribe_bars(
        self,
        symbols: list[str],
        callback: Callable[[Bar], None],
    ) -> None:
        """Subscribe to real-time bar updates."""
        pass

    @abstractmethod
    async def subscribe_quotes(
        self,
        symbols: list[str],
        callback: Callable[[Quote], None],
    ) -> None:
        """Subscribe to real-time quote updates."""
        pass

    @abstractmethod
    async def subscribe_trades(
        self,
        symbols: list[str],
        callback: Callable[[Trade], None],
    ) -> None:
        """Subscribe to real-time trade updates."""
        pass

    @abstractmethod
    async def subscribe_order_updates(
        self,
        callback: Callable[[Order], None],
    ) -> None:
        """Subscribe to order status updates."""
        pass

    @abstractmethod
    async def unsubscribe_all(self) -> None:
        """Unsubscribe from all data streams."""
        pass

    @abstractmethod
    def is_market_open(self) -> bool:
        """Check if the market is currently open."""
        pass

    @abstractmethod
    async def get_clock(self) -> dict[str, Any]:
        """Get market clock information."""
        pass
