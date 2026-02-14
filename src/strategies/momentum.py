"""Momentum/breakout trading strategy."""

from datetime import datetime
from typing import Any

import pandas as pd

from .base import BaseStrategy, Signal, SignalType
from ..data.indicators import IndicatorValues, calculate_breakout_levels


class MomentumStrategy(BaseStrategy):
    """Momentum breakout strategy.

    Entry conditions:
    - Price breaks above previous N-bar high
    - Volume surge (>1.5x average)
    - ATR > minimum threshold (volatile enough)
    - RSI not overbought (<70)

    Exit conditions:
    - Trailing stop hit (0.3%)
    - Take profit target (0.5%)
    - RSI overbought (>80)
    - Volume dies (< 0.8x average)
    """

    def __init__(self, params: dict[str, Any] | None = None):
        default_params = {
            "lookback_periods": 5,  # Bars for breakout detection
            "volume_multiplier": 1.5,  # Volume surge threshold
            "min_atr_pct": 0.005,  # Minimum ATR as % of price
            "trailing_stop_pct": 0.003,  # 0.3% trailing stop
            "take_profit_pct": 0.005,  # 0.5% profit target
            "stop_loss_pct": 0.01,  # 1% hard stop
            "max_rsi_entry": 70,  # Don't enter if RSI above this
            "exit_rsi": 80,  # Exit if RSI exceeds this
            "min_volume_ratio": 0.8,  # Exit if volume drops below
        }
        merged_params = {**default_params, **(params or {})}
        super().__init__(name="momentum", params=merged_params)

    def analyze(
        self,
        symbol: str,
        bars: pd.DataFrame,
        indicators: IndicatorValues,
        current_price: float,
    ) -> Signal | None:
        """Check for momentum breakout entry."""
        if not self.enabled:
            return None

        state = self.get_state(symbol)

        # Don't generate entry signals if already in position
        if state.in_position:
            return None

        # Need enough data
        lookback = self.params["lookback_periods"]
        if len(bars) < lookback + 5:
            return None

        # Check ATR threshold
        if indicators.atr_pct is None or indicators.atr_pct < self.params["min_atr_pct"]:
            return None

        # Check volume surge
        if not indicators.volume_surge:
            return None

        if indicators.volume_ratio and indicators.volume_ratio < self.params["volume_multiplier"]:
            return None

        # Check RSI not overbought
        if indicators.rsi_14 and indicators.rsi_14 > self.params["max_rsi_entry"]:
            return None

        # Check for breakout
        levels = calculate_breakout_levels(bars, lookback)
        breakout_high = levels.get("breakout_high")

        if breakout_high is None:
            return None

        # Price must break above recent high
        if current_price <= breakout_high:
            return None

        # Additional confirmation: price should be above VWAP
        if not indicators.above_vwap:
            return None

        # Generate buy signal
        confidence = self._calculate_confidence(indicators, current_price, breakout_high)

        stop_loss = self.get_stop_loss(current_price, "long")
        take_profit = self.get_take_profit(current_price, "long")

        state.signals_generated += 1
        state.last_signal_time = datetime.now()

        return Signal(
            symbol=symbol,
            signal_type=SignalType.BUY,
            strategy_name=self.name,
            confidence=confidence,
            price=current_price,
            timestamp=datetime.now(),
            reason=f"Breakout above {breakout_high:.2f} with {indicators.volume_ratio:.1f}x volume",
            metadata={
                "breakout_level": breakout_high,
                "volume_ratio": indicators.volume_ratio,
                "rsi": indicators.rsi_14,
                "atr_pct": indicators.atr_pct,
            },
            suggested_stop_loss=stop_loss,
            suggested_take_profit=take_profit,
            suggested_qty_pct=confidence,  # Size based on confidence
        )

    def should_exit(
        self,
        symbol: str,
        bars: pd.DataFrame,
        indicators: IndicatorValues,
        current_price: float,
        entry_price: float,
        position_side: str,
    ) -> Signal | None:
        """Check for exit conditions."""
        if not self.enabled:
            return None

        if position_side != "long":
            return None

        state = self.get_state(symbol)
        pnl_pct = (current_price - entry_price) / entry_price

        # Check take profit
        if pnl_pct >= self.params["take_profit_pct"]:
            return self._create_exit_signal(
                symbol, current_price, entry_price,
                reason=f"Take profit at {pnl_pct*100:.2f}%",
                exit_type="take_profit"
            )

        # Check hard stop loss
        if pnl_pct <= -self.params["stop_loss_pct"]:
            return self._create_exit_signal(
                symbol, current_price, entry_price,
                reason=f"Stop loss at {pnl_pct*100:.2f}%",
                exit_type="stop_loss"
            )

        # Check RSI overbought exit
        if indicators.rsi_14 and indicators.rsi_14 > self.params["exit_rsi"]:
            return self._create_exit_signal(
                symbol, current_price, entry_price,
                reason=f"RSI overbought at {indicators.rsi_14:.1f}",
                exit_type="rsi_exit"
            )

        # Check volume dying
        if indicators.volume_ratio and indicators.volume_ratio < self.params["min_volume_ratio"]:
            # Only exit if also in profit
            if pnl_pct > 0:
                return self._create_exit_signal(
                    symbol, current_price, entry_price,
                    reason=f"Volume dying ({indicators.volume_ratio:.2f}x avg)",
                    exit_type="volume_exit"
                )

        # Check if price dropped below VWAP
        if not indicators.above_vwap and pnl_pct < 0:
            return self._create_exit_signal(
                symbol, current_price, entry_price,
                reason="Price below VWAP with loss",
                exit_type="vwap_exit"
            )

        return None

    def _calculate_confidence(
        self,
        indicators: IndicatorValues,
        current_price: float,
        breakout_level: float,
    ) -> float:
        """Calculate signal confidence based on factors."""
        confidence = 0.5  # Base confidence

        # Volume strength
        if indicators.volume_ratio:
            if indicators.volume_ratio > 2.0:
                confidence += 0.15
            elif indicators.volume_ratio > 1.5:
                confidence += 0.10

        # Breakout strength (how far above breakout level)
        breakout_strength = (current_price - breakout_level) / breakout_level
        if breakout_strength > 0.005:  # More than 0.5% above
            confidence += 0.10

        # RSI position (prefer mid-range)
        if indicators.rsi_14:
            if 50 <= indicators.rsi_14 <= 65:
                confidence += 0.10
            elif 40 <= indicators.rsi_14 < 50:
                confidence += 0.05

        # Trend alignment
        if indicators.above_sma_20:
            confidence += 0.10

        # ADX (trend strength)
        if indicators.adx and indicators.adx > 25:
            confidence += 0.05

        return min(confidence, 1.0)

    def _create_exit_signal(
        self,
        symbol: str,
        current_price: float,
        entry_price: float,
        reason: str,
        exit_type: str,
    ) -> Signal:
        """Create an exit signal."""
        pnl_pct = (current_price - entry_price) / entry_price

        return Signal(
            symbol=symbol,
            signal_type=SignalType.SELL,
            strategy_name=self.name,
            confidence=1.0,
            price=current_price,
            timestamp=datetime.now(),
            reason=reason,
            metadata={
                "exit_type": exit_type,
                "entry_price": entry_price,
                "pnl_pct": pnl_pct,
            },
        )
