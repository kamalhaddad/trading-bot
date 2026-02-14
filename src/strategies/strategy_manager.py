"""Strategy manager for multi-strategy orchestration."""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

import pandas as pd

from .base import BaseStrategy, Signal, SignalType
from .momentum import MomentumStrategy
from .mean_reversion import MeanReversionStrategy
from .vwap import VWAPStrategy
from ..data.indicators import TechnicalIndicators, IndicatorValues
from ..data.market_data import MarketDataManager
from ..config import Config
from ..utils.logger import get_logger


logger = get_logger(__name__)


@dataclass
class ScoredSignal:
    """Signal with strategy weight applied."""
    signal: Signal
    weight: float
    weighted_score: float


@dataclass
class SymbolAnalysis:
    """Complete analysis for a symbol."""
    symbol: str
    current_price: float
    indicators: IndicatorValues
    signals: list[ScoredSignal] = field(default_factory=list)
    best_signal: Signal | None = None
    consensus_score: float = 0.0  # -1 to 1
    timestamp: datetime = field(default_factory=datetime.now)


class StrategyManager:
    """Manages multiple strategies and coordinates signal generation."""

    def __init__(
        self,
        config: Config,
        market_data: MarketDataManager,
    ):
        self.config = config
        self.market_data = market_data
        self.indicators = TechnicalIndicators()

        # Initialize strategies
        self.strategies: dict[str, BaseStrategy] = {}
        self._weights: dict[str, float] = {}

        self._init_strategies()

        # Track positions per strategy
        self._active_positions: dict[str, dict[str, str]] = {}  # symbol -> strategy

    def _init_strategies(self) -> None:
        """Initialize strategies based on configuration."""
        # Momentum strategy
        if self.config.is_strategy_enabled("momentum"):
            params = self.config.get_strategy_params("momentum")
            self.strategies["momentum"] = MomentumStrategy(params)
            self._weights["momentum"] = self.config.get_strategy_weight("momentum")
            logger.info("Initialized momentum strategy", weight=self._weights["momentum"])

        # Mean reversion strategy
        if self.config.is_strategy_enabled("mean_reversion"):
            params = self.config.get_strategy_params("mean_reversion")
            self.strategies["mean_reversion"] = MeanReversionStrategy(params)
            self._weights["mean_reversion"] = self.config.get_strategy_weight("mean_reversion")
            logger.info("Initialized mean_reversion strategy", weight=self._weights["mean_reversion"])

        # VWAP strategy
        if self.config.is_strategy_enabled("vwap"):
            params = self.config.get_strategy_params("vwap")
            self.strategies["vwap"] = VWAPStrategy(params)
            self._weights["vwap"] = self.config.get_strategy_weight("vwap")
            logger.info("Initialized VWAP strategy", weight=self._weights["vwap"])

    def analyze_symbol(self, symbol: str) -> SymbolAnalysis | None:
        """Analyze a symbol across all strategies.

        Args:
            symbol: Symbol to analyze

        Returns:
            SymbolAnalysis with signals from all strategies
        """
        # Get market data
        bars = self.market_data.get_bars(symbol)
        if bars.empty or len(bars) < 20:
            return None

        current_price = self.market_data.get_current_price(symbol)
        if not current_price:
            return None

        # Calculate indicators
        indicators = self.indicators.calculate(bars, symbol)

        # Get signals from each strategy
        scored_signals: list[ScoredSignal] = []

        for name, strategy in self.strategies.items():
            if not strategy.enabled:
                continue

            weight = self._weights.get(name, 0.33)

            # Check for entry signal
            signal = strategy.analyze(symbol, bars, indicators, current_price)

            if signal:
                weighted_score = signal.confidence * weight
                scored_signals.append(ScoredSignal(
                    signal=signal,
                    weight=weight,
                    weighted_score=weighted_score,
                ))

        # Calculate consensus score
        consensus = self._calculate_consensus(scored_signals, indicators)

        # Select best signal
        best_signal = self._select_best_signal(scored_signals)

        return SymbolAnalysis(
            symbol=symbol,
            current_price=current_price,
            indicators=indicators,
            signals=scored_signals,
            best_signal=best_signal,
            consensus_score=consensus,
        )

    def check_exits(
        self,
        symbol: str,
        entry_price: float,
        position_side: str,
        entry_strategy: str | None = None,
    ) -> Signal | None:
        """Check if any strategy signals an exit.

        Args:
            symbol: Symbol to check
            entry_price: Position entry price
            position_side: "long" or "short"
            entry_strategy: Strategy that opened the position

        Returns:
            Exit signal if any strategy triggers exit
        """
        bars = self.market_data.get_bars(symbol)
        if bars.empty:
            return None

        current_price = self.market_data.get_current_price(symbol)
        if not current_price:
            return None

        indicators = self.indicators.calculate(bars, symbol)

        # If we know which strategy opened the position, check that one first
        if entry_strategy and entry_strategy in self.strategies:
            strategy = self.strategies[entry_strategy]
            exit_signal = strategy.should_exit(
                symbol, bars, indicators, current_price, entry_price, position_side
            )
            if exit_signal:
                return exit_signal

        # Check all strategies for exit
        for name, strategy in self.strategies.items():
            if not strategy.enabled:
                continue

            exit_signal = strategy.should_exit(
                symbol, bars, indicators, current_price, entry_price, position_side
            )
            if exit_signal:
                return exit_signal

        return None

    def set_position_state(
        self,
        symbol: str,
        in_position: bool,
        entry_price: float | None = None,
        position_side: str | None = None,
        strategy_name: str | None = None,
    ) -> None:
        """Update position state across all strategies.

        Args:
            symbol: Symbol
            in_position: Whether in position
            entry_price: Entry price if in position
            position_side: "long" or "short"
            strategy_name: Strategy that opened the position
        """
        for strategy in self.strategies.values():
            strategy.set_in_position(symbol, in_position, entry_price, position_side)

        if in_position and strategy_name:
            if symbol not in self._active_positions:
                self._active_positions[symbol] = {}
            self._active_positions[symbol] = strategy_name
        elif not in_position and symbol in self._active_positions:
            del self._active_positions[symbol]

    def get_entry_strategy(self, symbol: str) -> str | None:
        """Get the strategy that opened a position."""
        return self._active_positions.get(symbol)

    def get_all_signals(self, symbols: list[str]) -> list[SymbolAnalysis]:
        """Analyze all symbols and return sorted by opportunity score.

        Args:
            symbols: List of symbols to analyze

        Returns:
            List of SymbolAnalysis sorted by best opportunities
        """
        analyses = []

        for symbol in symbols:
            analysis = self.analyze_symbol(symbol)
            if analysis and analysis.best_signal:
                analyses.append(analysis)

        # Sort by weighted score of best signal
        analyses.sort(
            key=lambda x: x.best_signal.confidence if x.best_signal else 0,
            reverse=True,
        )

        return analyses

    def get_strategy(self, name: str) -> BaseStrategy | None:
        """Get a strategy by name."""
        return self.strategies.get(name)

    def enable_strategy(self, name: str) -> None:
        """Enable a strategy."""
        if name in self.strategies:
            self.strategies[name].enabled = True
            logger.info("Enabled strategy", name=name)

    def disable_strategy(self, name: str) -> None:
        """Disable a strategy."""
        if name in self.strategies:
            self.strategies[name].enabled = False
            logger.info("Disabled strategy", name=name)

    def reset_all(self) -> None:
        """Reset all strategy states."""
        for strategy in self.strategies.values():
            strategy.reset()
        self._active_positions.clear()

    def _calculate_consensus(
        self,
        scored_signals: list[ScoredSignal],
        indicators: IndicatorValues,
    ) -> float:
        """Calculate consensus score from signals and indicators.

        Returns a score from -1 (strong sell) to 1 (strong buy).
        """
        if not scored_signals:
            # No signals, use indicator momentum
            from ..data.indicators import calculate_momentum_score
            return calculate_momentum_score(indicators) * 0.5

        total_weight = 0.0
        weighted_direction = 0.0

        for scored in scored_signals:
            direction = 1.0 if scored.signal.signal_type == SignalType.BUY else -1.0
            weighted_direction += direction * scored.weighted_score
            total_weight += scored.weight

        if total_weight > 0:
            return weighted_direction / total_weight

        return 0.0

    def _select_best_signal(
        self,
        scored_signals: list[ScoredSignal],
    ) -> Signal | None:
        """Select the best signal from scored signals."""
        if not scored_signals:
            return None

        # Sort by weighted score
        sorted_signals = sorted(
            scored_signals,
            key=lambda x: x.weighted_score,
            reverse=True,
        )

        return sorted_signals[0].signal

    def _check_conflicting_signals(
        self,
        scored_signals: list[ScoredSignal],
    ) -> bool:
        """Check if signals are conflicting (some buy, some sell)."""
        signal_types = {s.signal.signal_type for s in scored_signals}
        return SignalType.BUY in signal_types and SignalType.SELL in signal_types
