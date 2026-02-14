"""Alpaca broker implementation."""

import asyncio
from datetime import datetime, timedelta
from typing import Any, Callable
from zoneinfo import ZoneInfo

from alpaca.data.live import StockDataStream
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest, StockLatestBarRequest, StockLatestQuoteRequest
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import (
    MarketOrderRequest,
    LimitOrderRequest,
    StopOrderRequest,
    StopLimitOrderRequest,
    TrailingStopOrderRequest,
    GetOrdersRequest,
    ClosePositionRequest,
)
from alpaca.trading.enums import (
    OrderSide as AlpacaOrderSide,
    OrderType as AlpacaOrderType,
    TimeInForce as AlpacaTimeInForce,
    OrderStatus as AlpacaOrderStatus,
    QueryOrderStatus,
)
from alpaca.trading.stream import TradingStream

from .base import (
    BaseBroker,
    Order,
    OrderSide,
    OrderType,
    OrderStatus,
    TimeInForce,
    Position,
    AccountInfo,
    Bar,
    Quote,
    Trade,
)
from ..utils.logger import get_logger


logger = get_logger(__name__)


class AlpacaClient(BaseBroker):
    """Alpaca broker client implementation."""

    def __init__(
        self,
        api_key: str,
        api_secret: str,
        paper: bool = True,
    ):
        self.api_key = api_key
        self.api_secret = api_secret
        self.paper = paper

        # Clients
        self._trading_client: TradingClient | None = None
        self._data_client: StockHistoricalDataClient | None = None
        self._data_stream: StockDataStream | None = None
        self._trading_stream: TradingStream | None = None

        # Callbacks
        self._bar_callbacks: list[Callable[[Bar], None]] = []
        self._quote_callbacks: list[Callable[[Quote], None]] = []
        self._trade_callbacks: list[Callable[[Trade], None]] = []
        self._order_callbacks: list[Callable[[Order], None]] = []

        # State
        self._connected = False
        self._subscribed_symbols: set[str] = set()
        self._market_open = False
        self._stream_task: asyncio.Task | None = None
        self._trading_stream_task: asyncio.Task | None = None

    async def connect(self) -> None:
        """Connect to Alpaca."""
        logger.info("Connecting to Alpaca", paper=self.paper)

        # Initialize trading client
        self._trading_client = TradingClient(
            api_key=self.api_key,
            secret_key=self.api_secret,
            paper=self.paper,
        )

        # Initialize data client
        self._data_client = StockHistoricalDataClient(
            api_key=self.api_key,
            secret_key=self.api_secret,
        )

        # Initialize data stream
        self._data_stream = StockDataStream(
            api_key=self.api_key,
            secret_key=self.api_secret,
        )

        # Initialize trading stream for order updates
        self._trading_stream = TradingStream(
            api_key=self.api_key,
            secret_key=self.api_secret,
            paper=self.paper,
        )

        # Test connection
        account = await self.get_account()
        logger.info(
            "Connected to Alpaca",
            equity=account.equity,
            buying_power=account.buying_power,
        )

        self._connected = True

    async def disconnect(self) -> None:
        """Disconnect from Alpaca."""
        logger.info("Disconnecting from Alpaca")

        # Stop streams
        if self._stream_task:
            self._stream_task.cancel()
            try:
                await self._stream_task
            except asyncio.CancelledError:
                pass

        if self._trading_stream_task:
            self._trading_stream_task.cancel()
            try:
                await self._trading_stream_task
            except asyncio.CancelledError:
                pass

        if self._data_stream:
            await self._data_stream.close()

        if self._trading_stream:
            await self._trading_stream.close()

        self._connected = False
        logger.info("Disconnected from Alpaca")

    async def get_account(self) -> AccountInfo:
        """Get account information."""
        if not self._trading_client:
            raise RuntimeError("Not connected to Alpaca")

        account = self._trading_client.get_account()

        return AccountInfo(
            equity=float(account.equity),
            cash=float(account.cash),
            buying_power=float(account.buying_power),
            portfolio_value=float(account.portfolio_value),
            day_trade_count=account.daytrade_count,
            pattern_day_trader=account.pattern_day_trader,
            trading_blocked=account.trading_blocked,
            account_blocked=account.account_blocked,
        )

    async def get_positions(self) -> list[Position]:
        """Get all open positions."""
        if not self._trading_client:
            raise RuntimeError("Not connected to Alpaca")

        positions = self._trading_client.get_all_positions()

        return [
            Position(
                symbol=p.symbol,
                qty=float(p.qty),
                avg_entry_price=float(p.avg_entry_price),
                current_price=float(p.current_price),
                market_value=float(p.market_value),
                unrealized_pnl=float(p.unrealized_pl),
                unrealized_pnl_pct=float(p.unrealized_plpc) * 100,
                side="long" if float(p.qty) > 0 else "short",
            )
            for p in positions
        ]

    async def get_position(self, symbol: str) -> Position | None:
        """Get position for a specific symbol."""
        if not self._trading_client:
            raise RuntimeError("Not connected to Alpaca")

        try:
            p = self._trading_client.get_open_position(symbol)
            return Position(
                symbol=p.symbol,
                qty=float(p.qty),
                avg_entry_price=float(p.avg_entry_price),
                current_price=float(p.current_price),
                market_value=float(p.market_value),
                unrealized_pnl=float(p.unrealized_pl),
                unrealized_pnl_pct=float(p.unrealized_plpc) * 100,
                side="long" if float(p.qty) > 0 else "short",
            )
        except Exception:
            return None

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
        if not self._trading_client:
            raise RuntimeError("Not connected to Alpaca")

        alpaca_side = AlpacaOrderSide.BUY if side == OrderSide.BUY else AlpacaOrderSide.SELL
        alpaca_tif = self._convert_tif(time_in_force)

        order_request: Any

        if order_type == OrderType.MARKET:
            order_request = MarketOrderRequest(
                symbol=symbol,
                qty=qty,
                side=alpaca_side,
                time_in_force=alpaca_tif,
                client_order_id=client_order_id,
            )
        elif order_type == OrderType.LIMIT:
            if limit_price is None:
                raise ValueError("Limit price required for limit orders")
            order_request = LimitOrderRequest(
                symbol=symbol,
                qty=qty,
                side=alpaca_side,
                time_in_force=alpaca_tif,
                limit_price=limit_price,
                client_order_id=client_order_id,
            )
        elif order_type == OrderType.STOP:
            if stop_price is None:
                raise ValueError("Stop price required for stop orders")
            order_request = StopOrderRequest(
                symbol=symbol,
                qty=qty,
                side=alpaca_side,
                time_in_force=alpaca_tif,
                stop_price=stop_price,
                client_order_id=client_order_id,
            )
        elif order_type == OrderType.STOP_LIMIT:
            if stop_price is None or limit_price is None:
                raise ValueError("Stop and limit prices required for stop-limit orders")
            order_request = StopLimitOrderRequest(
                symbol=symbol,
                qty=qty,
                side=alpaca_side,
                time_in_force=alpaca_tif,
                stop_price=stop_price,
                limit_price=limit_price,
                client_order_id=client_order_id,
            )
        elif order_type == OrderType.TRAILING_STOP:
            if trail_percent is None:
                raise ValueError("Trail percent required for trailing stop orders")
            order_request = TrailingStopOrderRequest(
                symbol=symbol,
                qty=qty,
                side=alpaca_side,
                time_in_force=alpaca_tif,
                trail_percent=trail_percent * 100,  # Convert to percentage
                client_order_id=client_order_id,
            )
        else:
            raise ValueError(f"Unsupported order type: {order_type}")

        logger.info(
            "Submitting order",
            symbol=symbol,
            side=side.value,
            qty=qty,
            order_type=order_type.value,
        )

        order = self._trading_client.submit_order(order_request)
        return self._convert_order(order)

    async def cancel_order(self, order_id: str) -> bool:
        """Cancel an order."""
        if not self._trading_client:
            raise RuntimeError("Not connected to Alpaca")

        try:
            self._trading_client.cancel_order_by_id(order_id)
            logger.info("Order cancelled", order_id=order_id)
            return True
        except Exception as e:
            logger.warning("Failed to cancel order", order_id=order_id, error=str(e))
            return False

    async def cancel_all_orders(self) -> int:
        """Cancel all open orders."""
        if not self._trading_client:
            raise RuntimeError("Not connected to Alpaca")

        cancelled = self._trading_client.cancel_orders()
        count = len(cancelled)
        logger.info("Cancelled all orders", count=count)
        return count

    async def get_order(self, order_id: str) -> Order | None:
        """Get order by ID."""
        if not self._trading_client:
            raise RuntimeError("Not connected to Alpaca")

        try:
            order = self._trading_client.get_order_by_id(order_id)
            return self._convert_order(order)
        except Exception:
            return None

    async def get_open_orders(self, symbol: str | None = None) -> list[Order]:
        """Get all open orders."""
        if not self._trading_client:
            raise RuntimeError("Not connected to Alpaca")

        request = GetOrdersRequest(status=QueryOrderStatus.OPEN)
        if symbol:
            request.symbols = [symbol]

        orders = self._trading_client.get_orders(request)
        return [self._convert_order(o) for o in orders]

    async def close_position(self, symbol: str) -> Order | None:
        """Close a position."""
        if not self._trading_client:
            raise RuntimeError("Not connected to Alpaca")

        try:
            order = self._trading_client.close_position(symbol)
            logger.info("Position closed", symbol=symbol)
            return self._convert_order(order)
        except Exception as e:
            logger.warning("Failed to close position", symbol=symbol, error=str(e))
            return None

    async def close_all_positions(self) -> list[Order]:
        """Close all positions."""
        if not self._trading_client:
            raise RuntimeError("Not connected to Alpaca")

        results = self._trading_client.close_all_positions(cancel_orders=True)
        orders = []
        for result in results:
            if hasattr(result, "body") and result.body:
                orders.append(self._convert_order(result.body))
        logger.info("Closed all positions", count=len(orders))
        return orders

    async def get_bars(
        self,
        symbol: str,
        timeframe: str,
        start: datetime | None = None,
        end: datetime | None = None,
        limit: int = 100,
    ) -> list[Bar]:
        """Get historical bar data."""
        if not self._data_client:
            raise RuntimeError("Not connected to Alpaca")

        tf = self._parse_timeframe(timeframe)

        if start is None:
            start = datetime.now(ZoneInfo("America/New_York")) - timedelta(days=5)
        if end is None:
            end = datetime.now(ZoneInfo("America/New_York"))

        request = StockBarsRequest(
            symbol_or_symbols=symbol,
            timeframe=tf,
            start=start,
            end=end,
            limit=limit,
        )

        bars_data = self._data_client.get_stock_bars(request)

        bars = []
        if symbol in bars_data:
            for b in bars_data[symbol]:
                bars.append(Bar(
                    symbol=symbol,
                    timestamp=b.timestamp,
                    open=b.open,
                    high=b.high,
                    low=b.low,
                    close=b.close,
                    volume=b.volume,
                    vwap=b.vwap if hasattr(b, "vwap") else None,
                ))

        return bars

    async def get_latest_bar(self, symbol: str) -> Bar | None:
        """Get the latest bar for a symbol."""
        if not self._data_client:
            raise RuntimeError("Not connected to Alpaca")

        try:
            request = StockLatestBarRequest(symbol_or_symbols=symbol)
            result = self._data_client.get_stock_latest_bar(request)

            if symbol in result:
                b = result[symbol]
                return Bar(
                    symbol=symbol,
                    timestamp=b.timestamp,
                    open=b.open,
                    high=b.high,
                    low=b.low,
                    close=b.close,
                    volume=b.volume,
                    vwap=b.vwap if hasattr(b, "vwap") else None,
                )
        except Exception as e:
            logger.warning("Failed to get latest bar", symbol=symbol, error=str(e))

        return None

    async def get_latest_quote(self, symbol: str) -> Quote | None:
        """Get the latest quote for a symbol."""
        if not self._data_client:
            raise RuntimeError("Not connected to Alpaca")

        try:
            request = StockLatestQuoteRequest(symbol_or_symbols=symbol)
            result = self._data_client.get_stock_latest_quote(request)

            if symbol in result:
                q = result[symbol]
                return Quote(
                    symbol=symbol,
                    timestamp=q.timestamp,
                    bid_price=q.bid_price,
                    bid_size=q.bid_size,
                    ask_price=q.ask_price,
                    ask_size=q.ask_size,
                )
        except Exception as e:
            logger.warning("Failed to get latest quote", symbol=symbol, error=str(e))

        return None

    async def subscribe_bars(
        self,
        symbols: list[str],
        callback: Callable[[Bar], None],
    ) -> None:
        """Subscribe to real-time bar updates."""
        if not self._data_stream:
            raise RuntimeError("Not connected to Alpaca")

        self._bar_callbacks.append(callback)

        async def bar_handler(bar):
            b = Bar(
                symbol=bar.symbol,
                timestamp=bar.timestamp,
                open=bar.open,
                high=bar.high,
                low=bar.low,
                close=bar.close,
                volume=bar.volume,
                vwap=bar.vwap if hasattr(bar, "vwap") else None,
            )
            for cb in self._bar_callbacks:
                try:
                    cb(b)
                except Exception as e:
                    logger.error("Error in bar callback", error=str(e))

        self._data_stream.subscribe_bars(bar_handler, *symbols)
        self._subscribed_symbols.update(symbols)

        # Start stream if not already running
        await self._ensure_stream_running()

    async def subscribe_quotes(
        self,
        symbols: list[str],
        callback: Callable[[Quote], None],
    ) -> None:
        """Subscribe to real-time quote updates."""
        if not self._data_stream:
            raise RuntimeError("Not connected to Alpaca")

        self._quote_callbacks.append(callback)

        async def quote_handler(quote):
            q = Quote(
                symbol=quote.symbol,
                timestamp=quote.timestamp,
                bid_price=quote.bid_price,
                bid_size=quote.bid_size,
                ask_price=quote.ask_price,
                ask_size=quote.ask_size,
            )
            for cb in self._quote_callbacks:
                try:
                    cb(q)
                except Exception as e:
                    logger.error("Error in quote callback", error=str(e))

        self._data_stream.subscribe_quotes(quote_handler, *symbols)
        self._subscribed_symbols.update(symbols)

        await self._ensure_stream_running()

    async def subscribe_trades(
        self,
        symbols: list[str],
        callback: Callable[[Trade], None],
    ) -> None:
        """Subscribe to real-time trade updates."""
        if not self._data_stream:
            raise RuntimeError("Not connected to Alpaca")

        self._trade_callbacks.append(callback)

        async def trade_handler(trade):
            t = Trade(
                symbol=trade.symbol,
                timestamp=trade.timestamp,
                price=trade.price,
                size=trade.size,
            )
            for cb in self._trade_callbacks:
                try:
                    cb(t)
                except Exception as e:
                    logger.error("Error in trade callback", error=str(e))

        self._data_stream.subscribe_trades(trade_handler, *symbols)
        self._subscribed_symbols.update(symbols)

        await self._ensure_stream_running()

    async def subscribe_order_updates(
        self,
        callback: Callable[[Order], None],
    ) -> None:
        """Subscribe to order status updates."""
        if not self._trading_stream:
            raise RuntimeError("Not connected to Alpaca")

        self._order_callbacks.append(callback)

        async def trade_update_handler(data):
            if hasattr(data, "order"):
                order = self._convert_order(data.order)
                for cb in self._order_callbacks:
                    try:
                        cb(order)
                    except Exception as e:
                        logger.error("Error in order callback", error=str(e))

        self._trading_stream.subscribe_trade_updates(trade_update_handler)

        # Start trading stream
        if self._trading_stream_task is None:
            async def run_trading_stream():
                try:
                    await self._trading_stream._run_forever()
                except asyncio.CancelledError:
                    pass
                except Exception as e:
                    logger.error("Trading stream error", error=str(e))

            self._trading_stream_task = asyncio.create_task(run_trading_stream())

    async def unsubscribe_all(self) -> None:
        """Unsubscribe from all data streams."""
        self._bar_callbacks.clear()
        self._quote_callbacks.clear()
        self._trade_callbacks.clear()
        self._order_callbacks.clear()
        self._subscribed_symbols.clear()

        if self._stream_task:
            self._stream_task.cancel()
            self._stream_task = None

    def is_market_open(self) -> bool:
        """Check if the market is currently open."""
        if not self._trading_client:
            return False

        try:
            clock = self._trading_client.get_clock()
            return clock.is_open
        except Exception:
            return False

    async def get_clock(self) -> dict[str, Any]:
        """Get market clock information."""
        if not self._trading_client:
            raise RuntimeError("Not connected to Alpaca")

        clock = self._trading_client.get_clock()
        return {
            "is_open": clock.is_open,
            "next_open": clock.next_open,
            "next_close": clock.next_close,
            "timestamp": clock.timestamp,
        }

    async def _ensure_stream_running(self) -> None:
        """Ensure the data stream is running."""
        if self._stream_task is None and self._data_stream:
            async def run_stream():
                try:
                    await self._data_stream._run_forever()
                except asyncio.CancelledError:
                    pass
                except Exception as e:
                    logger.error("Data stream error", error=str(e))

            self._stream_task = asyncio.create_task(run_stream())

    def _convert_order(self, order: Any) -> Order:
        """Convert Alpaca order to internal Order."""
        return Order(
            id=str(order.id),
            symbol=order.symbol,
            side=OrderSide.BUY if order.side == AlpacaOrderSide.BUY else OrderSide.SELL,
            qty=float(order.qty),
            order_type=self._convert_order_type(order.order_type),
            status=self._convert_order_status(order.status),
            time_in_force=self._convert_tif_from_alpaca(order.time_in_force),
            limit_price=float(order.limit_price) if order.limit_price else None,
            stop_price=float(order.stop_price) if order.stop_price else None,
            trail_percent=float(order.trail_percent) / 100 if order.trail_percent else None,
            filled_qty=float(order.filled_qty) if order.filled_qty else 0.0,
            filled_avg_price=float(order.filled_avg_price) if order.filled_avg_price else None,
            created_at=order.created_at,
            filled_at=order.filled_at,
            client_order_id=order.client_order_id,
        )

    def _convert_order_type(self, order_type: AlpacaOrderType) -> OrderType:
        """Convert Alpaca order type."""
        mapping = {
            AlpacaOrderType.MARKET: OrderType.MARKET,
            AlpacaOrderType.LIMIT: OrderType.LIMIT,
            AlpacaOrderType.STOP: OrderType.STOP,
            AlpacaOrderType.STOP_LIMIT: OrderType.STOP_LIMIT,
            AlpacaOrderType.TRAILING_STOP: OrderType.TRAILING_STOP,
        }
        return mapping.get(order_type, OrderType.MARKET)

    def _convert_order_status(self, status: AlpacaOrderStatus) -> OrderStatus:
        """Convert Alpaca order status."""
        mapping = {
            AlpacaOrderStatus.NEW: OrderStatus.NEW,
            AlpacaOrderStatus.PENDING_NEW: OrderStatus.PENDING,
            AlpacaOrderStatus.ACCEPTED: OrderStatus.ACCEPTED,
            AlpacaOrderStatus.FILLED: OrderStatus.FILLED,
            AlpacaOrderStatus.PARTIALLY_FILLED: OrderStatus.PARTIALLY_FILLED,
            AlpacaOrderStatus.CANCELED: OrderStatus.CANCELED,
            AlpacaOrderStatus.REJECTED: OrderStatus.REJECTED,
            AlpacaOrderStatus.EXPIRED: OrderStatus.EXPIRED,
        }
        return mapping.get(status, OrderStatus.NEW)

    def _convert_tif(self, tif: TimeInForce) -> AlpacaTimeInForce:
        """Convert TimeInForce to Alpaca format."""
        mapping = {
            TimeInForce.DAY: AlpacaTimeInForce.DAY,
            TimeInForce.GTC: AlpacaTimeInForce.GTC,
            TimeInForce.IOC: AlpacaTimeInForce.IOC,
            TimeInForce.FOK: AlpacaTimeInForce.FOK,
        }
        return mapping.get(tif, AlpacaTimeInForce.DAY)

    def _convert_tif_from_alpaca(self, tif: AlpacaTimeInForce) -> TimeInForce:
        """Convert Alpaca TimeInForce to internal format."""
        mapping = {
            AlpacaTimeInForce.DAY: TimeInForce.DAY,
            AlpacaTimeInForce.GTC: TimeInForce.GTC,
            AlpacaTimeInForce.IOC: TimeInForce.IOC,
            AlpacaTimeInForce.FOK: TimeInForce.FOK,
        }
        return mapping.get(tif, TimeInForce.DAY)

    def _parse_timeframe(self, timeframe: str) -> TimeFrame:
        """Parse timeframe string to Alpaca TimeFrame."""
        tf_lower = timeframe.lower()
        if tf_lower in ("1min", "1m"):
            return TimeFrame(1, TimeFrameUnit.Minute)
        elif tf_lower in ("5min", "5m"):
            return TimeFrame(5, TimeFrameUnit.Minute)
        elif tf_lower in ("15min", "15m"):
            return TimeFrame(15, TimeFrameUnit.Minute)
        elif tf_lower in ("1hour", "1h"):
            return TimeFrame(1, TimeFrameUnit.Hour)
        elif tf_lower in ("1day", "1d"):
            return TimeFrame(1, TimeFrameUnit.Day)
        else:
            return TimeFrame(1, TimeFrameUnit.Minute)
