"""VWAP bounce trading strategy."""

from datetime import datetime, time
from typing import Any
from zoneinfo import ZoneInfo

import pandas as pd

from .base import BaseStrategy, Signal, SignalType
from ..data.indicators import IndicatorValues


class VWAPStrategy(BaseStrategy):
    """VWAP bounce strategy.

    Entry conditions:
    - Price touches VWAP from above in an uptrend
    - Price bounces off VWAP with volume confirmation
    - Only trade in first 2 hours of session (highest liquidity)

    Exit conditions:
    - Fixed take profit (0.4%)
    - Stop loss below VWAP (0.2%)
    - Price closes below VWAP
    """

    def __init__(self, params: dict[str, Any] | None = None):
        default_params = {
            "bounce_threshold": 0.001,  # 0.1% from VWAP to consider touch
            "volume_confirm": 1.2,  # Volume confirmation multiplier
            "take_profit_pct": 0.004,  # 0.4% profit target
            "stop_loss_pct": 0.002,  # 0.2% stop loss
            "active_hours_only": True,  # Only first 2 hours
            "active_hours_start": "09:30",
            "active_hours_end": "11:30",
            "timezone": "America/New_York",
            "require_uptrend": True,  # Price must be in uptrend
            "min_touches": 1,  # Minimum VWAP touches before entry
        }
        merged_params = {**default_params, **(params or {})}
        super().__init__(name="vwap", params=merged_params)

        # Track VWAP touches per symbol
        self._vwap_touches: dict[str, int] = {}
        self._last_touch_time: dict[str, datetime] = {}

    def analyze(
        self,
        symbol: str,
        bars: pd.DataFrame,
        indicators: IndicatorValues,
        current_price: float,
    ) -> Signal | None:
        """Check for VWAP bounce entry."""
        if not self.enabled:
            return None

        state = self.get_state(symbol)

        # Don't generate entry signals if already in position
        if state.in_position:
            return None

        # Check active hours
        if self.params["active_hours_only"] and not self._is_active_hours():
            return None

        # Need VWAP
        if indicators.vwap is None:
            return None

        # Calculate distance from VWAP
        vwap_distance = (current_price - indicators.vwap) / indicators.vwap

        # Check if price is touching VWAP (within threshold)
        if abs(vwap_distance) > self.params["bounce_threshold"]:
            # Not touching VWAP
            self._check_and_record_touch(symbol, current_price, indicators.vwap)
            return None

        # Price is at VWAP - check for bounce setup

        # Must be above VWAP (bouncing up)
        if vwap_distance < 0:
            # Below VWAP, record touch but don't enter
            self._record_touch(symbol)
            return None

        # Check uptrend requirement
        if self.params["require_uptrend"]:
            if not indicators.above_sma_20:
                return None

        # Check volume confirmation
        if indicators.volume_ratio and indicators.volume_ratio < self.params["volume_confirm"]:
            return None

        # Check we've seen enough touches (indicates VWAP as support)
        touches = self._vwap_touches.get(symbol, 0)
        if touches < self.params["min_touches"]:
            self._record_touch(symbol)
            return None

        # Check previous bar touched VWAP (bounce confirmation)
        if len(bars) < 2:
            return None

        prev_low = bars["low"].iloc[-2]
        prev_touched_vwap = abs(prev_low - indicators.vwap) / indicators.vwap < 0.002

        if not prev_touched_vwap:
            return None

        # Generate buy signal
        confidence = self._calculate_confidence(indicators, touches)

        stop_loss = indicators.vwap * (1 - self.params["stop_loss_pct"])
        take_profit = current_price * (1 + self.params["take_profit_pct"])

        state.signals_generated += 1
        state.last_signal_time = datetime.now()

        # Reset touch counter after entry
        self._vwap_touches[symbol] = 0

        return Signal(
            symbol=symbol,
            signal_type=SignalType.BUY,
            strategy_name=self.name,
            confidence=confidence,
            price=current_price,
            timestamp=datetime.now(),
            reason=f"VWAP bounce at {indicators.vwap:.2f}, {touches} touches",
            metadata={
                "vwap": indicators.vwap,
                "vwap_distance_pct": vwap_distance * 100,
                "touches": touches,
                "volume_ratio": indicators.volume_ratio,
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

        # Check take profit
        if pnl_pct >= self.params["take_profit_pct"]:
            return self._create_exit_signal(
                symbol, current_price, entry_price,
                reason=f"Take profit at {pnl_pct*100:.2f}%",
                exit_type="take_profit"
            )

        # Check stop loss (price below VWAP by threshold)
        if indicators.vwap:
            vwap_distance = (current_price - indicators.vwap) / indicators.vwap
            if vwap_distance < -self.params["stop_loss_pct"]:
                return self._create_exit_signal(
                    symbol, current_price, entry_price,
                    reason=f"Stop loss below VWAP ({vwap_distance*100:.2f}%)",
                    exit_type="stop_loss"
                )

        # Check if price closes below VWAP
        if not indicators.above_vwap:
            # Give some tolerance for noise
            if indicators.vwap and current_price < indicators.vwap * 0.998:
                return self._create_exit_signal(
                    symbol, current_price, entry_price,
                    reason="Price closed below VWAP",
                    exit_type="vwap_break"
                )

        # Check end of active hours (exit before low liquidity)
        if self.params["active_hours_only"]:
            now = datetime.now(ZoneInfo(self.params["timezone"]))
            end_time = datetime.strptime(self.params["active_hours_end"], "%H:%M").time()

            # If approaching end of active hours and in profit, exit
            current_time = now.time()
            if current_time >= end_time and pnl_pct > 0:
                return self._create_exit_signal(
                    symbol, current_price, entry_price,
                    reason="End of active hours with profit",
                    exit_type="time_exit"
                )

        return None

    def _is_active_hours(self) -> bool:
        """Check if within active trading hours."""
        now = datetime.now(ZoneInfo(self.params["timezone"]))
        current_time = now.time()

        start_time = datetime.strptime(self.params["active_hours_start"], "%H:%M").time()
        end_time = datetime.strptime(self.params["active_hours_end"], "%H:%M").time()

        return start_time <= current_time <= end_time

    def _check_and_record_touch(
        self,
        symbol: str,
        current_price: float,
        vwap: float,
    ) -> None:
        """Check if price recently touched VWAP and record it."""
        # Check if we're coming from a touch
        last_touch = self._last_touch_time.get(symbol)
        if last_touch:
            # If last touch was recent (within 5 minutes), don't count again
            if (datetime.now() - last_touch).total_seconds() < 300:
                return

    def _record_touch(self, symbol: str) -> None:
        """Record a VWAP touch."""
        self._vwap_touches[symbol] = self._vwap_touches.get(symbol, 0) + 1
        self._last_touch_time[symbol] = datetime.now()

    def _calculate_confidence(
        self,
        indicators: IndicatorValues,
        touches: int,
    ) -> float:
        """Calculate signal confidence."""
        confidence = 0.5

        # More touches = more confidence (VWAP acting as support)
        if touches >= 3:
            confidence += 0.20
        elif touches >= 2:
            confidence += 0.10

        # Volume confirmation
        if indicators.volume_ratio:
            if indicators.volume_ratio > 1.5:
                confidence += 0.15
            elif indicators.volume_ratio > 1.2:
                confidence += 0.10

        # Trend strength
        if indicators.above_sma_20:
            confidence += 0.10

        # RSI in favorable range
        if indicators.rsi_14:
            if 40 <= indicators.rsi_14 <= 60:
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

    def reset(self) -> None:
        """Reset strategy state."""
        super().reset()
        self._vwap_touches.clear()
        self._last_touch_time.clear()

    def reset_symbol(self, symbol: str) -> None:
        """Reset state for a specific symbol."""
        super().reset_symbol(symbol)
        if symbol in self._vwap_touches:
            del self._vwap_touches[symbol]
        if symbol in self._last_touch_time:
            del self._last_touch_time[symbol]
