"""Technical indicators for trading strategies."""

from dataclasses import dataclass

import numpy as np
import pandas as pd
import ta as ta_lib
from ta.momentum import RSIIndicator
from ta.volatility import BollingerBands, AverageTrueRange
from ta.trend import ADXIndicator, MACD


@dataclass
class IndicatorValues:
    """Container for indicator values."""
    symbol: str
    timestamp: pd.Timestamp | None = None

    # Price info
    close: float | None = None
    high: float | None = None
    low: float | None = None
    volume: int | None = None

    # Moving averages
    sma_20: float | None = None
    sma_50: float | None = None
    ema_9: float | None = None
    ema_21: float | None = None

    # Momentum
    rsi_14: float | None = None
    rsi_7: float | None = None

    # Bollinger Bands
    bb_upper: float | None = None
    bb_middle: float | None = None
    bb_lower: float | None = None
    bb_width: float | None = None
    bb_pct: float | None = None  # %B

    # VWAP
    vwap: float | None = None
    vwap_upper: float | None = None
    vwap_lower: float | None = None

    # Volatility
    atr_14: float | None = None
    atr_pct: float | None = None  # ATR as % of price

    # Volume
    volume_sma_20: float | None = None
    volume_ratio: float | None = None  # Current vs average

    # Trend
    adx: float | None = None
    plus_di: float | None = None
    minus_di: float | None = None

    # MACD
    macd: float | None = None
    macd_signal: float | None = None
    macd_hist: float | None = None

    # Derived signals
    above_vwap: bool = False
    above_sma_20: bool = False
    rsi_oversold: bool = False
    rsi_overbought: bool = False
    at_bb_lower: bool = False
    at_bb_upper: bool = False
    volume_surge: bool = False
    strong_trend: bool = False


class TechnicalIndicators:
    """Calculate technical indicators from price data."""

    def __init__(
        self,
        rsi_period: int = 14,
        bb_period: int = 20,
        bb_std: float = 2.0,
        atr_period: int = 14,
        sma_periods: list[int] | None = None,
        ema_periods: list[int] | None = None,
    ):
        self.rsi_period = rsi_period
        self.bb_period = bb_period
        self.bb_std = bb_std
        self.atr_period = atr_period
        self.sma_periods = sma_periods or [20, 50]
        self.ema_periods = ema_periods or [9, 21]

    def calculate(self, df: pd.DataFrame, symbol: str) -> IndicatorValues:
        """Calculate all indicators from OHLCV DataFrame.

        Args:
            df: DataFrame with columns [open, high, low, close, volume]
            symbol: Symbol for identification

        Returns:
            IndicatorValues with all calculated indicators
        """
        if df.empty or len(df) < 20:
            return IndicatorValues(symbol=symbol)

        values = IndicatorValues(symbol=symbol)

        # Latest values
        values.timestamp = df.index[-1] if hasattr(df.index[-1], 'timestamp') else df.index[-1]
        values.close = df["close"].iloc[-1]
        values.high = df["high"].iloc[-1]
        values.low = df["low"].iloc[-1]
        values.volume = int(df["volume"].iloc[-1])

        # Calculate indicators
        self._calc_moving_averages(df, values)
        self._calc_rsi(df, values)
        self._calc_bollinger_bands(df, values)
        self._calc_vwap(df, values)
        self._calc_atr(df, values)
        self._calc_volume_indicators(df, values)
        self._calc_adx(df, values)
        self._calc_macd(df, values)

        # Derive signals
        self._derive_signals(values)

        return values

    def _calc_moving_averages(self, df: pd.DataFrame, values: IndicatorValues) -> None:
        """Calculate moving averages."""
        # SMAs
        if len(df) >= 20:
            values.sma_20 = df["close"].rolling(20).mean().iloc[-1]
        if len(df) >= 50:
            values.sma_50 = df["close"].rolling(50).mean().iloc[-1]

        # EMAs
        if len(df) >= 9:
            values.ema_9 = df["close"].ewm(span=9, adjust=False).mean().iloc[-1]
        if len(df) >= 21:
            values.ema_21 = df["close"].ewm(span=21, adjust=False).mean().iloc[-1]

    def _calc_rsi(self, df: pd.DataFrame, values: IndicatorValues) -> None:
        """Calculate RSI."""
        if len(df) < self.rsi_period + 1:
            return

        # RSI 14
        rsi_indicator = RSIIndicator(df["close"], window=self.rsi_period)
        rsi = rsi_indicator.rsi()
        if rsi is not None and len(rsi) > 0:
            values.rsi_14 = rsi.iloc[-1]

        # RSI 7 (faster)
        if len(df) >= 8:
            rsi_7_indicator = RSIIndicator(df["close"], window=7)
            rsi_7 = rsi_7_indicator.rsi()
            if rsi_7 is not None and len(rsi_7) > 0:
                values.rsi_7 = rsi_7.iloc[-1]

    def _calc_bollinger_bands(self, df: pd.DataFrame, values: IndicatorValues) -> None:
        """Calculate Bollinger Bands."""
        if len(df) < self.bb_period:
            return

        bb = BollingerBands(df["close"], window=self.bb_period, window_dev=self.bb_std)
        values.bb_lower = bb.bollinger_lband().iloc[-1]
        values.bb_middle = bb.bollinger_mavg().iloc[-1]
        values.bb_upper = bb.bollinger_hband().iloc[-1]
        values.bb_width = bb.bollinger_wband().iloc[-1]
        values.bb_pct = bb.bollinger_pband().iloc[-1]

    def _calc_vwap(self, df: pd.DataFrame, values: IndicatorValues) -> None:
        """Calculate VWAP and bands."""
        if len(df) < 2:
            return

        # Check if vwap column exists (from broker)
        if "vwap" in df.columns and df["vwap"].iloc[-1] is not None:
            values.vwap = df["vwap"].iloc[-1]
        else:
            # Calculate VWAP manually
            typical_price = (df["high"] + df["low"] + df["close"]) / 3
            cumulative_tp_vol = (typical_price * df["volume"]).cumsum()
            cumulative_vol = df["volume"].cumsum()
            vwap = cumulative_tp_vol / cumulative_vol
            values.vwap = vwap.iloc[-1]

        # VWAP bands (using standard deviation of price from VWAP)
        if values.vwap and values.close:
            # Simple bands at 1% from VWAP
            values.vwap_upper = values.vwap * 1.01
            values.vwap_lower = values.vwap * 0.99

    def _calc_atr(self, df: pd.DataFrame, values: IndicatorValues) -> None:
        """Calculate ATR."""
        if len(df) < self.atr_period + 1:
            return

        atr_indicator = AverageTrueRange(df["high"], df["low"], df["close"], window=self.atr_period)
        atr = atr_indicator.average_true_range()
        if atr is not None and len(atr) > 0:
            values.atr_14 = atr.iloc[-1]
            if values.close and values.close > 0:
                values.atr_pct = values.atr_14 / values.close

    def _calc_volume_indicators(self, df: pd.DataFrame, values: IndicatorValues) -> None:
        """Calculate volume indicators."""
        if len(df) < 20:
            return

        vol_sma = df["volume"].rolling(20).mean()
        values.volume_sma_20 = vol_sma.iloc[-1]

        if values.volume_sma_20 and values.volume_sma_20 > 0:
            values.volume_ratio = values.volume / values.volume_sma_20

    def _calc_adx(self, df: pd.DataFrame, values: IndicatorValues) -> None:
        """Calculate ADX for trend strength."""
        if len(df) < 15:
            return

        adx_indicator = ADXIndicator(df["high"], df["low"], df["close"], window=14)
        values.adx = adx_indicator.adx().iloc[-1]
        values.plus_di = adx_indicator.adx_pos().iloc[-1]
        values.minus_di = adx_indicator.adx_neg().iloc[-1]

    def _calc_macd(self, df: pd.DataFrame, values: IndicatorValues) -> None:
        """Calculate MACD."""
        if len(df) < 26:
            return

        macd_indicator = MACD(df["close"])
        values.macd = macd_indicator.macd().iloc[-1]
        values.macd_signal = macd_indicator.macd_signal().iloc[-1]
        values.macd_hist = macd_indicator.macd_diff().iloc[-1]

    def _derive_signals(self, values: IndicatorValues) -> None:
        """Derive trading signals from indicators."""
        if values.close is None:
            return

        # VWAP position
        if values.vwap:
            values.above_vwap = values.close > values.vwap

        # SMA position
        if values.sma_20:
            values.above_sma_20 = values.close > values.sma_20

        # RSI signals
        if values.rsi_14:
            values.rsi_oversold = values.rsi_14 < 30
            values.rsi_overbought = values.rsi_14 > 70

        # Bollinger Band positions
        if values.bb_lower and values.bb_upper:
            bb_range = values.bb_upper - values.bb_lower
            if bb_range > 0:
                values.at_bb_lower = (values.close - values.bb_lower) < bb_range * 0.1
                values.at_bb_upper = (values.bb_upper - values.close) < bb_range * 0.1

        # Volume surge
        if values.volume_ratio:
            values.volume_surge = values.volume_ratio > 1.5

        # Trend strength
        if values.adx:
            values.strong_trend = values.adx > 25


def calculate_breakout_levels(df: pd.DataFrame, lookback: int = 5) -> dict:
    """Calculate breakout levels from recent price action.

    Args:
        df: DataFrame with OHLCV data
        lookback: Number of periods to look back

    Returns:
        Dict with breakout_high and breakout_low levels
    """
    if df.empty or len(df) < lookback:
        return {"breakout_high": None, "breakout_low": None}

    recent = df.tail(lookback)
    return {
        "breakout_high": recent["high"].max(),
        "breakout_low": recent["low"].min(),
    }


def calculate_support_resistance(
    df: pd.DataFrame,
    lookback: int = 50,
    num_levels: int = 3,
) -> dict:
    """Calculate support and resistance levels.

    Args:
        df: DataFrame with OHLCV data
        lookback: Number of periods to analyze
        num_levels: Number of levels to return

    Returns:
        Dict with support and resistance levels
    """
    if df.empty or len(df) < lookback:
        return {"support": [], "resistance": []}

    recent = df.tail(lookback)
    current_price = df["close"].iloc[-1]

    # Find pivot points (local highs and lows)
    highs = recent["high"].values
    lows = recent["low"].values

    # Simple approach: use recent peaks and troughs
    resistance_levels = []
    support_levels = []

    # Find local maxima for resistance
    for i in range(2, len(highs) - 2):
        if highs[i] > highs[i-1] and highs[i] > highs[i+1]:
            if highs[i] > current_price:
                resistance_levels.append(highs[i])

    # Find local minima for support
    for i in range(2, len(lows) - 2):
        if lows[i] < lows[i-1] and lows[i] < lows[i+1]:
            if lows[i] < current_price:
                support_levels.append(lows[i])

    # Sort and return top levels
    resistance_levels = sorted(set(resistance_levels))[:num_levels]
    support_levels = sorted(set(support_levels), reverse=True)[:num_levels]

    return {
        "support": support_levels,
        "resistance": resistance_levels,
    }


def calculate_momentum_score(indicators: IndicatorValues) -> float:
    """Calculate a momentum score from -1 to 1.

    Args:
        indicators: IndicatorValues object

    Returns:
        Momentum score from -1 (bearish) to 1 (bullish)
    """
    score = 0.0
    factors = 0

    # RSI contribution
    if indicators.rsi_14:
        rsi_score = (indicators.rsi_14 - 50) / 50  # -1 to 1
        score += rsi_score * 0.3
        factors += 0.3

    # MACD contribution
    if indicators.macd_hist is not None:
        # Normalize MACD histogram
        macd_score = np.clip(indicators.macd_hist / 0.5, -1, 1)
        score += macd_score * 0.25
        factors += 0.25

    # Price vs VWAP
    if indicators.vwap and indicators.close:
        vwap_pct = (indicators.close - indicators.vwap) / indicators.vwap
        vwap_score = np.clip(vwap_pct * 20, -1, 1)  # 5% deviation = max score
        score += vwap_score * 0.2
        factors += 0.2

    # ADX/DI contribution
    if indicators.adx and indicators.plus_di and indicators.minus_di:
        if indicators.adx > 20:  # Only if trending
            di_diff = indicators.plus_di - indicators.minus_di
            di_score = np.clip(di_diff / 30, -1, 1)
            score += di_score * 0.25
            factors += 0.25

    if factors > 0:
        score = score / factors

    return np.clip(score, -1, 1)
