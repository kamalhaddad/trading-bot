"""Mean reversion trading strategy."""

from datetime import datetime
from typing import Any

import pandas as pd

from .base import BaseStrategy, Signal, SignalType
from ..data.indicators import IndicatorValues


class MeanReversionStrategy(BaseStrategy):
    """Mean reversion strategy using RSI and Bollinger Bands.

    Entry conditions (long):
    - RSI(14) < 25 (oversold)
    - Price at or below lower Bollinger Band
    - Still in overall uptrend (above 20 SMA on daily or above VWAP)

    Exit conditions:
    - RSI crosses above 50
    - Price touches middle Bollinger Band
    - Stop loss at 1%
    """

    def __init__(self, params: dict[str, Any] | None = None):
        default_params = {
            "rsi_period": 14,
            "rsi_oversold": 25,  # Entry threshold
            "rsi_overbought": 75,  # For potential short entries
            "rsi_exit": 50,  # Exit when RSI crosses this
            "bb_period": 20,
            "bb_std": 2.0,
            "stop_loss_pct": 0.01,  # 1% stop loss
            "take_profit_pct": 0.008,  # 0.8% take profit
            "require_uptrend": True,  # Require price above SMA 20
        }
        merged_params = {**default_params, **(params or {})}
        super().__init__(name="mean_reversion", params=merged_params)

    def analyze(
        self,
        symbol: str,
        bars: pd.DataFrame,
        indicators: IndicatorValues,
        current_price: float,
    ) -> Signal | None:
        """Check for mean reversion entry."""
        if not self.enabled:
            return None

        state = self.get_state(symbol)

        # Don't generate entry signals if already in position
        if state.in_position:
            return None

        # Need RSI and Bollinger Bands
        if indicators.rsi_14 is None:
            return None
        if indicators.bb_lower is None or indicators.bb_middle is None:
            return None

        # Check oversold condition
        if indicators.rsi_14 > self.params["rsi_oversold"]:
            return None

        # Check price at lower Bollinger Band
        if not indicators.at_bb_lower:
            # Additional check: price must be within 0.5% of lower band
            if indicators.bb_lower:
                distance_to_bb = (current_price - indicators.bb_lower) / indicators.bb_lower
                if distance_to_bb > 0.005:  # More than 0.5% above lower band
                    return None

        # Check uptrend requirement
        if self.params["require_uptrend"]:
            # Use VWAP or SMA for trend filter
            if not indicators.above_vwap and not indicators.above_sma_20:
                # Both below, but we can still trade if RSI is very low
                if indicators.rsi_14 > 20:
                    return None

        # Generate buy signal
        confidence = self._calculate_confidence(indicators)

        stop_loss = self.get_stop_loss(current_price, "long")
        take_profit = indicators.bb_middle  # Target middle band

        state.signals_generated += 1
        state.last_signal_time = datetime.now()

        return Signal(
            symbol=symbol,
            signal_type=SignalType.BUY,
            strategy_name=self.name,
            confidence=confidence,
            price=current_price,
            timestamp=datetime.now(),
            reason=f"RSI={indicators.rsi_14:.1f} at lower BB, expecting reversion",
            metadata={
                "rsi": indicators.rsi_14,
                "bb_lower": indicators.bb_lower,
                "bb_middle": indicators.bb_middle,
                "bb_upper": indicators.bb_upper,
            },
            suggested_stop_loss=stop_loss,
            suggested_take_profit=take_profit,
            suggested_qty_pct=confidence,
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

        pnl_pct = (current_price - entry_price) / entry_price

        # Check stop loss
        if pnl_pct <= -self.params["stop_loss_pct"]:
            return self._create_exit_signal(
                symbol, current_price, entry_price,
                reason=f"Stop loss at {pnl_pct*100:.2f}%",
                exit_type="stop_loss"
            )

        # Check take profit
        if pnl_pct >= self.params["take_profit_pct"]:
            return self._create_exit_signal(
                symbol, current_price, entry_price,
                reason=f"Take profit at {pnl_pct*100:.2f}%",
                exit_type="take_profit"
            )

        # Check RSI exit (reversion complete)
        if indicators.rsi_14 and indicators.rsi_14 > self.params["rsi_exit"]:
            return self._create_exit_signal(
                symbol, current_price, entry_price,
                reason=f"RSI crossed above {self.params['rsi_exit']}: {indicators.rsi_14:.1f}",
                exit_type="rsi_exit"
            )

        # Check middle Bollinger Band target
        if indicators.bb_middle:
            if current_price >= indicators.bb_middle:
                return self._create_exit_signal(
                    symbol, current_price, entry_price,
                    reason=f"Reached middle BB at {indicators.bb_middle:.2f}",
                    exit_type="bb_target"
                )

        # Check if RSI is getting overbought (close position to lock gains)
        if indicators.rsi_14 and indicators.rsi_14 > 65 and pnl_pct > 0:
            return self._create_exit_signal(
                symbol, current_price, entry_price,
                reason=f"RSI reaching overbought ({indicators.rsi_14:.1f}) with profit",
                exit_type="rsi_profit_exit"
            )

        return None

    def _calculate_confidence(self, indicators: IndicatorValues) -> float:
        """Calculate signal confidence."""
        confidence = 0.5

        # RSI depth (lower RSI = higher confidence)
        if indicators.rsi_14:
            if indicators.rsi_14 < 15:
                confidence += 0.25
            elif indicators.rsi_14 < 20:
                confidence += 0.15
            elif indicators.rsi_14 < 25:
                confidence += 0.10

        # Bollinger Band position
        if indicators.bb_pct is not None:
            if indicators.bb_pct < 0:  # Below lower band
                confidence += 0.15
            elif indicators.bb_pct < 0.1:  # Near lower band
                confidence += 0.10

        # VWAP position (still above = better)
        if indicators.above_vwap:
            confidence += 0.10

        # Volume (prefer normal to high volume on oversold)
        if indicators.volume_ratio:
            if 0.8 <= indicators.volume_ratio <= 1.5:
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
