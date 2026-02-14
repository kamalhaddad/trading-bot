"""Real-time market data management."""

import asyncio
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Callable
from zoneinfo import ZoneInfo

import pandas as pd

from ..broker.base import BaseBroker, Bar, Quote, Trade
from ..utils.logger import get_logger


logger = get_logger(__name__)


@dataclass
class BarBuffer:
    """Buffer for storing recent bars."""
    symbol: str
    max_bars: int = 500
    bars: list[Bar] = field(default_factory=list)
    _df: pd.DataFrame | None = field(default=None, repr=False)
    _df_dirty: bool = True

    def add(self, bar: Bar) -> None:
        """Add a bar to the buffer."""
        self.bars.append(bar)
        if len(self.bars) > self.max_bars:
            self.bars = self.bars[-self.max_bars:]
        self._df_dirty = True

    def get_dataframe(self) -> pd.DataFrame:
        """Get bars as a DataFrame."""
        if self._df_dirty or self._df is None:
            if not self.bars:
                self._df = pd.DataFrame(columns=[
                    "timestamp", "open", "high", "low", "close", "volume", "vwap"
                ])
            else:
                self._df = pd.DataFrame([
                    {
                        "timestamp": b.timestamp,
                        "open": b.open,
                        "high": b.high,
                        "low": b.low,
                        "close": b.close,
                        "volume": b.volume,
                        "vwap": b.vwap,
                    }
                    for b in self.bars
                ])
                self._df.set_index("timestamp", inplace=True)
            self._df_dirty = False
        return self._df

    @property
    def latest(self) -> Bar | None:
        """Get the latest bar."""
        return self.bars[-1] if self.bars else None

    @property
    def count(self) -> int:
        """Get number of bars."""
        return len(self.bars)


@dataclass
class QuoteData:
    """Latest quote data for a symbol."""
    symbol: str
    bid_price: float = 0.0
    bid_size: int = 0
    ask_price: float = 0.0
    ask_size: int = 0
    mid_price: float = 0.0
    spread: float = 0.0
    spread_pct: float = 0.0
    last_update: datetime | None = None

    def update(self, quote: Quote) -> None:
        """Update with new quote data."""
        self.bid_price = quote.bid_price
        self.bid_size = quote.bid_size
        self.ask_price = quote.ask_price
        self.ask_size = quote.ask_size
        self.mid_price = (quote.bid_price + quote.ask_price) / 2
        self.spread = quote.ask_price - quote.bid_price
        self.spread_pct = self.spread / self.mid_price if self.mid_price > 0 else 0
        self.last_update = quote.timestamp


@dataclass
class TradeData:
    """Aggregated trade data for a symbol."""
    symbol: str
    last_price: float = 0.0
    last_size: int = 0
    cumulative_volume: int = 0
    vwap: float = 0.0
    high: float = 0.0
    low: float = float("inf")
    trade_count: int = 0
    last_update: datetime | None = None
    _volume_price_sum: float = 0.0

    def update(self, trade: Trade) -> None:
        """Update with new trade data."""
        self.last_price = trade.price
        self.last_size = trade.size
        self.cumulative_volume += trade.size
        self._volume_price_sum += trade.price * trade.size
        self.vwap = self._volume_price_sum / self.cumulative_volume if self.cumulative_volume > 0 else 0
        self.high = max(self.high, trade.price)
        if self.low == float("inf"):
            self.low = trade.price
        else:
            self.low = min(self.low, trade.price)
        self.trade_count += 1
        self.last_update = trade.timestamp

    def reset_daily(self) -> None:
        """Reset daily aggregates."""
        self.cumulative_volume = 0
        self._volume_price_sum = 0.0
        self.vwap = 0.0
        self.high = 0.0
        self.low = float("inf")
        self.trade_count = 0


class MarketDataManager:
    """Manages real-time market data."""

    def __init__(
        self,
        broker: BaseBroker,
        max_bars: int = 500,
    ):
        self.broker = broker
        self.max_bars = max_bars

        # Data storage
        self._bar_buffers: dict[str, BarBuffer] = {}
        self._quotes: dict[str, QuoteData] = {}
        self._trades: dict[str, TradeData] = {}

        # Callbacks
        self._bar_callbacks: list[Callable[[Bar], None]] = []
        self._quote_callbacks: list[Callable[[Quote], None]] = []
        self._trade_callbacks: list[Callable[[Trade], None]] = []

        # State
        self._subscribed_symbols: set[str] = set()
        self._running = False

    async def start(self, symbols: list[str]) -> None:
        """Start market data manager."""
        logger.info("Starting market data manager", symbols=symbols)

        # Initialize data structures
        for symbol in symbols:
            self._bar_buffers[symbol] = BarBuffer(symbol=symbol, max_bars=self.max_bars)
            self._quotes[symbol] = QuoteData(symbol=symbol)
            self._trades[symbol] = TradeData(symbol=symbol)

        self._subscribed_symbols = set(symbols)

        # Load historical bars for context
        await self._load_historical_bars(symbols)

        # Subscribe to real-time data
        await self.broker.subscribe_bars(symbols, self._on_bar)
        await self.broker.subscribe_quotes(symbols, self._on_quote)
        await self.broker.subscribe_trades(symbols, self._on_trade)

        self._running = True
        logger.info("Market data manager started")

    async def stop(self) -> None:
        """Stop market data manager."""
        logger.info("Stopping market data manager")
        await self.broker.unsubscribe_all()
        self._running = False

    async def add_symbols(self, symbols: list[str]) -> None:
        """Add symbols to watch."""
        new_symbols = [s for s in symbols if s not in self._subscribed_symbols]
        if not new_symbols:
            return

        for symbol in new_symbols:
            self._bar_buffers[symbol] = BarBuffer(symbol=symbol, max_bars=self.max_bars)
            self._quotes[symbol] = QuoteData(symbol=symbol)
            self._trades[symbol] = TradeData(symbol=symbol)

        await self._load_historical_bars(new_symbols)
        await self.broker.subscribe_bars(new_symbols, self._on_bar)
        await self.broker.subscribe_quotes(new_symbols, self._on_quote)
        await self.broker.subscribe_trades(new_symbols, self._on_trade)

        self._subscribed_symbols.update(new_symbols)

    def on_bar(self, callback: Callable[[Bar], None]) -> None:
        """Register a bar callback."""
        self._bar_callbacks.append(callback)

    def on_quote(self, callback: Callable[[Quote], None]) -> None:
        """Register a quote callback."""
        self._quote_callbacks.append(callback)

    def on_trade(self, callback: Callable[[Trade], None]) -> None:
        """Register a trade callback."""
        self._trade_callbacks.append(callback)

    def get_bars(self, symbol: str) -> pd.DataFrame:
        """Get historical bars as DataFrame."""
        if symbol in self._bar_buffers:
            return self._bar_buffers[symbol].get_dataframe()
        return pd.DataFrame()

    def get_latest_bar(self, symbol: str) -> Bar | None:
        """Get latest bar for a symbol."""
        if symbol in self._bar_buffers:
            return self._bar_buffers[symbol].latest
        return None

    def get_quote(self, symbol: str) -> QuoteData | None:
        """Get current quote data."""
        return self._quotes.get(symbol)

    def get_trade_data(self, symbol: str) -> TradeData | None:
        """Get current trade data."""
        return self._trades.get(symbol)

    def get_current_price(self, symbol: str) -> float | None:
        """Get current price (mid-quote or last trade)."""
        quote = self._quotes.get(symbol)
        if quote and quote.mid_price > 0:
            return quote.mid_price

        trade = self._trades.get(symbol)
        if trade and trade.last_price > 0:
            return trade.last_price

        bar = self.get_latest_bar(symbol)
        if bar:
            return bar.close

        return None

    def get_vwap(self, symbol: str) -> float | None:
        """Get current VWAP."""
        trade = self._trades.get(symbol)
        if trade and trade.vwap > 0:
            return trade.vwap
        return None

    def get_spread_pct(self, symbol: str) -> float | None:
        """Get current bid-ask spread as percentage."""
        quote = self._quotes.get(symbol)
        if quote:
            return quote.spread_pct
        return None

    def get_volume(self, symbol: str) -> int:
        """Get cumulative volume today."""
        trade = self._trades.get(symbol)
        return trade.cumulative_volume if trade else 0

    def reset_daily(self) -> None:
        """Reset daily aggregates (call at market open)."""
        for trade_data in self._trades.values():
            trade_data.reset_daily()

    async def _load_historical_bars(self, symbols: list[str]) -> None:
        """Load historical bars for initialization."""
        logger.info("Loading historical bars", count=len(symbols))

        for symbol in symbols:
            try:
                bars = await self.broker.get_bars(
                    symbol=symbol,
                    timeframe="1min",
                    limit=self.max_bars,
                )
                for bar in bars:
                    self._bar_buffers[symbol].add(bar)
                logger.debug(
                    "Loaded historical bars",
                    symbol=symbol,
                    count=len(bars),
                )
            except Exception as e:
                logger.warning(
                    "Failed to load historical bars",
                    symbol=symbol,
                    error=str(e),
                )

    def _on_bar(self, bar: Bar) -> None:
        """Handle incoming bar."""
        if bar.symbol in self._bar_buffers:
            self._bar_buffers[bar.symbol].add(bar)

        for callback in self._bar_callbacks:
            try:
                callback(bar)
            except Exception as e:
                logger.error("Error in bar callback", error=str(e))

    def _on_quote(self, quote: Quote) -> None:
        """Handle incoming quote."""
        if quote.symbol in self._quotes:
            self._quotes[quote.symbol].update(quote)

        for callback in self._quote_callbacks:
            try:
                callback(quote)
            except Exception as e:
                logger.error("Error in quote callback", error=str(e))

    def _on_trade(self, trade: Trade) -> None:
        """Handle incoming trade."""
        if trade.symbol in self._trades:
            self._trades[trade.symbol].update(trade)

        for callback in self._trade_callbacks:
            try:
                callback(trade)
            except Exception as e:
                logger.error("Error in trade callback", error=str(e))
