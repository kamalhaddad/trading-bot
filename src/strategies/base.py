"""Base strategy interface."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any

import pandas as pd

from ..data.indicators import IndicatorValues


class SignalType(str, Enum):
    """Type of trading signal."""
    BUY = "buy"
    SELL = "sell"
    HOLD = "hold"


@dataclass
class Signal:
    """Trading signal from a strategy."""
    symbol: str
    signal_type: SignalType
    strategy_name: str
    confidence: float  # 0.0 to 1.0
    price: float
    timestamp: datetime
    reason: str
    metadata: dict = field(default_factory=dict)

    # Position management hints
    suggested_stop_loss: float | None = None
    suggested_take_profit: float | None = None
    suggested_qty_pct: float | None = None  # Suggested position size as % of max

    def __post_init__(self):
        """Validate signal data."""
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError(f"Confidence must be between 0 and 1, got {self.confidence}")


@dataclass
class StrategyState:
    """State tracked by a strategy for a symbol."""
    symbol: str
    in_position: bool = False
    entry_price: float | None = None
    entry_time: datetime | None = None
    position_side: str | None = None  # "long" or "short"
    signals_generated: int = 0
    last_signal_time: datetime | None = None
    custom_data: dict = field(default_factory=dict)


class BaseStrategy(ABC):
    """Abstract base class for trading strategies."""

    def __init__(self, name: str, params: dict[str, Any] | None = None):
        """Initialize strategy.

        Args:
            name: Strategy name
            params: Strategy parameters
        """
        self.name = name
        self.params = params or {}
        self._states: dict[str, StrategyState] = {}
        self._enabled = True

    @property
    def enabled(self) -> bool:
        """Check if strategy is enabled."""
        return self._enabled

    @enabled.setter
    def enabled(self, value: bool) -> None:
        """Enable or disable strategy."""
        self._enabled = value

    def get_state(self, symbol: str) -> StrategyState:
        """Get state for a symbol, creating if needed."""
        if symbol not in self._states:
            self._states[symbol] = StrategyState(symbol=symbol)
        return self._states[symbol]

    def set_in_position(
        self,
        symbol: str,
        in_position: bool,
        entry_price: float | None = None,
        position_side: str | None = None,
    ) -> None:
        """Update position state for a symbol."""
        state = self.get_state(symbol)
        state.in_position = in_position
        if in_position:
            state.entry_price = entry_price
            state.entry_time = datetime.now()
            state.position_side = position_side
        else:
            state.entry_price = None
            state.entry_time = None
            state.position_side = None

    @abstractmethod
    def analyze(
        self,
        symbol: str,
        bars: pd.DataFrame,
        indicators: IndicatorValues,
        current_price: float,
    ) -> Signal | None:
        """Analyze market data and generate a signal.

        Args:
            symbol: Symbol to analyze
            bars: Historical OHLCV data
            indicators: Pre-calculated indicators
            current_price: Current market price

        Returns:
            Signal if action should be taken, None otherwise
        """
        pass

    @abstractmethod
    def should_exit(
        self,
        symbol: str,
        bars: pd.DataFrame,
        indicators: IndicatorValues,
        current_price: float,
        entry_price: float,
        position_side: str,
    ) -> Signal | None:
        """Check if an existing position should be exited.

        Args:
            symbol: Symbol to check
            bars: Historical OHLCV data
            indicators: Pre-calculated indicators
            current_price: Current market price
            entry_price: Position entry price
            position_side: "long" or "short"

        Returns:
            Sell signal if should exit, None otherwise
        """
        pass

    def get_stop_loss(self, entry_price: float, side: str = "long") -> float:
        """Calculate stop loss price.

        Args:
            entry_price: Entry price
            side: "long" or "short"

        Returns:
            Stop loss price
        """
        stop_pct = self.params.get("stop_loss_pct", 0.01)
        if side == "long":
            return entry_price * (1 - stop_pct)
        return entry_price * (1 + stop_pct)

    def get_take_profit(self, entry_price: float, side: str = "long") -> float:
        """Calculate take profit price.

        Args:
            entry_price: Entry price
            side: "long" or "short"

        Returns:
            Take profit price
        """
        tp_pct = self.params.get("take_profit_pct", 0.005)
        if side == "long":
            return entry_price * (1 + tp_pct)
        return entry_price * (1 - tp_pct)

    def get_trailing_stop_pct(self) -> float:
        """Get trailing stop percentage."""
        return self.params.get("trailing_stop_pct", 0.003)

    def reset(self) -> None:
        """Reset strategy state."""
        self._states.clear()

    def reset_symbol(self, symbol: str) -> None:
        """Reset state for a specific symbol."""
        if symbol in self._states:
            del self._states[symbol]

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name={self.name!r}, enabled={self.enabled})"
