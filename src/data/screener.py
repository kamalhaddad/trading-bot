"""Stock screener for finding trading opportunities."""

from dataclasses import dataclass
from datetime import datetime

from .market_data import MarketDataManager
from .indicators import TechnicalIndicators, IndicatorValues
from ..utils.logger import get_logger


logger = get_logger(__name__)


@dataclass
class ScreenerCriteria:
    """Criteria for stock screening."""
    min_price: float = 10.0
    max_price: float = 200.0
    min_volume: int = 100000  # Minimum daily volume
    min_atr_pct: float = 0.005  # Minimum volatility (0.5%)
    max_spread_pct: float = 0.002  # Maximum spread (0.2%)
    require_uptrend: bool = False  # Price above SMA 20
    require_volume_surge: bool = False  # Volume > 1.5x average


@dataclass
class ScreenerResult:
    """Result from stock screening."""
    symbol: str
    price: float
    volume: int
    atr_pct: float | None
    spread_pct: float | None
    above_vwap: bool
    above_sma: bool
    rsi: float | None
    volume_ratio: float | None
    momentum_score: float
    passed: bool
    reasons: list[str]


class StockScreener:
    """Screens stocks based on technical criteria."""

    def __init__(
        self,
        market_data: MarketDataManager,
        indicators: TechnicalIndicators,
        criteria: ScreenerCriteria | None = None,
    ):
        self.market_data = market_data
        self.indicators = indicators
        self.criteria = criteria or ScreenerCriteria()

    def screen(self, symbols: list[str]) -> list[ScreenerResult]:
        """Screen symbols against criteria.

        Args:
            symbols: List of symbols to screen

        Returns:
            List of ScreenerResult objects, sorted by momentum score
        """
        results = []

        for symbol in symbols:
            result = self._screen_symbol(symbol)
            if result:
                results.append(result)

        # Sort by momentum score descending
        results.sort(key=lambda x: x.momentum_score, reverse=True)

        return results

    def screen_for_momentum(self, symbols: list[str]) -> list[ScreenerResult]:
        """Screen for momentum setups.

        Looks for:
        - Price above VWAP
        - Price above SMA 20
        - Volume surge (>1.5x average)
        - RSI between 40-70 (not oversold/overbought)
        """
        criteria = ScreenerCriteria(
            min_price=self.criteria.min_price,
            max_price=self.criteria.max_price,
            min_volume=self.criteria.min_volume,
            min_atr_pct=self.criteria.min_atr_pct,
            max_spread_pct=self.criteria.max_spread_pct,
            require_uptrend=True,
            require_volume_surge=True,
        )
        original_criteria = self.criteria
        self.criteria = criteria
        results = self.screen(symbols)
        self.criteria = original_criteria

        # Additional momentum filter
        return [
            r for r in results
            if r.above_vwap and r.rsi and 40 <= r.rsi <= 70
        ]

    def screen_for_mean_reversion(self, symbols: list[str]) -> list[ScreenerResult]:
        """Screen for mean reversion setups.

        Looks for:
        - RSI < 30 (oversold)
        - Price near lower Bollinger Band
        - Still in uptrend (above SMA 20 on daily)
        """
        results = []

        for symbol in symbols:
            df = self.market_data.get_bars(symbol)
            if df.empty or len(df) < 20:
                continue

            ind = self.indicators.calculate(df, symbol)

            # Check mean reversion criteria
            if ind.rsi_14 is None or ind.rsi_14 > 30:
                continue

            if not ind.at_bb_lower:
                continue

            # Basic price/spread check
            price = self.market_data.get_current_price(symbol)
            if not price:
                continue

            if price < self.criteria.min_price or price > self.criteria.max_price:
                continue

            spread_pct = self.market_data.get_spread_pct(symbol)
            if spread_pct and spread_pct > self.criteria.max_spread_pct:
                continue

            results.append(ScreenerResult(
                symbol=symbol,
                price=price,
                volume=self.market_data.get_volume(symbol),
                atr_pct=ind.atr_pct,
                spread_pct=spread_pct,
                above_vwap=ind.above_vwap,
                above_sma=ind.above_sma_20,
                rsi=ind.rsi_14,
                volume_ratio=ind.volume_ratio,
                momentum_score=-1,  # Contrarian
                passed=True,
                reasons=["RSI oversold", "At lower BB"],
            ))

        # Sort by lowest RSI (most oversold)
        results.sort(key=lambda x: x.rsi or 100)
        return results

    def screen_for_vwap_bounce(self, symbols: list[str]) -> list[ScreenerResult]:
        """Screen for VWAP bounce setups.

        Looks for:
        - Price at or just above VWAP
        - Previous candle touched VWAP
        - Still in uptrend
        - Volume confirmation
        """
        results = []

        for symbol in symbols:
            df = self.market_data.get_bars(symbol)
            if df.empty or len(df) < 20:
                continue

            ind = self.indicators.calculate(df, symbol)

            if not ind.vwap or not ind.close:
                continue

            # Check if price is near VWAP (within 0.3%)
            vwap_distance = abs(ind.close - ind.vwap) / ind.vwap
            if vwap_distance > 0.003:
                continue

            # Should be above VWAP (bouncing up)
            if not ind.above_vwap:
                continue

            # Check uptrend
            if not ind.above_sma_20:
                continue

            # Basic checks
            price = self.market_data.get_current_price(symbol)
            if not price:
                continue

            if price < self.criteria.min_price or price > self.criteria.max_price:
                continue

            spread_pct = self.market_data.get_spread_pct(symbol)
            if spread_pct and spread_pct > self.criteria.max_spread_pct:
                continue

            results.append(ScreenerResult(
                symbol=symbol,
                price=price,
                volume=self.market_data.get_volume(symbol),
                atr_pct=ind.atr_pct,
                spread_pct=spread_pct,
                above_vwap=True,
                above_sma=True,
                rsi=ind.rsi_14,
                volume_ratio=ind.volume_ratio,
                momentum_score=0.5,
                passed=True,
                reasons=["At VWAP", "Uptrend"],
            ))

        return results

    def _screen_symbol(self, symbol: str) -> ScreenerResult | None:
        """Screen a single symbol."""
        df = self.market_data.get_bars(symbol)
        if df.empty or len(df) < 20:
            return None

        price = self.market_data.get_current_price(symbol)
        if not price:
            return None

        # Calculate indicators
        ind = self.indicators.calculate(df, symbol)

        # Check criteria
        reasons = []
        passed = True

        # Price range
        if price < self.criteria.min_price:
            reasons.append(f"Price too low: ${price:.2f}")
            passed = False
        elif price > self.criteria.max_price:
            reasons.append(f"Price too high: ${price:.2f}")
            passed = False

        # Volume
        volume = self.market_data.get_volume(symbol)
        if volume < self.criteria.min_volume:
            reasons.append(f"Volume too low: {volume:,}")
            passed = False

        # Volatility
        if ind.atr_pct and ind.atr_pct < self.criteria.min_atr_pct:
            reasons.append(f"ATR too low: {ind.atr_pct:.4f}")
            passed = False

        # Spread
        spread_pct = self.market_data.get_spread_pct(symbol)
        if spread_pct and spread_pct > self.criteria.max_spread_pct:
            reasons.append(f"Spread too wide: {spread_pct:.4f}")
            passed = False

        # Uptrend requirement
        if self.criteria.require_uptrend and not ind.above_sma_20:
            reasons.append("Not in uptrend")
            passed = False

        # Volume surge requirement
        if self.criteria.require_volume_surge and not ind.volume_surge:
            reasons.append("No volume surge")
            passed = False

        # Calculate momentum score
        from .indicators import calculate_momentum_score
        momentum = calculate_momentum_score(ind)

        return ScreenerResult(
            symbol=symbol,
            price=price,
            volume=volume,
            atr_pct=ind.atr_pct,
            spread_pct=spread_pct,
            above_vwap=ind.above_vwap,
            above_sma=ind.above_sma_20,
            rsi=ind.rsi_14,
            volume_ratio=ind.volume_ratio,
            momentum_score=momentum,
            passed=passed,
            reasons=reasons if not passed else ["All criteria met"],
        )

    def get_top_movers(
        self,
        symbols: list[str],
        count: int = 10,
    ) -> list[ScreenerResult]:
        """Get top momentum movers.

        Args:
            symbols: Symbols to check
            count: Number of top movers to return

        Returns:
            Top movers sorted by momentum score
        """
        results = self.screen(symbols)
        passed = [r for r in results if r.passed]
        return passed[:count]

    def get_oversold(
        self,
        symbols: list[str],
        rsi_threshold: float = 30,
    ) -> list[ScreenerResult]:
        """Get oversold stocks.

        Args:
            symbols: Symbols to check
            rsi_threshold: RSI threshold for oversold

        Returns:
            Oversold stocks sorted by RSI (lowest first)
        """
        results = []

        for symbol in symbols:
            df = self.market_data.get_bars(symbol)
            if df.empty:
                continue

            ind = self.indicators.calculate(df, symbol)
            if ind.rsi_14 and ind.rsi_14 < rsi_threshold:
                price = self.market_data.get_current_price(symbol)
                if price:
                    results.append(ScreenerResult(
                        symbol=symbol,
                        price=price,
                        volume=self.market_data.get_volume(symbol),
                        atr_pct=ind.atr_pct,
                        spread_pct=self.market_data.get_spread_pct(symbol),
                        above_vwap=ind.above_vwap,
                        above_sma=ind.above_sma_20,
                        rsi=ind.rsi_14,
                        volume_ratio=ind.volume_ratio,
                        momentum_score=-1,
                        passed=True,
                        reasons=["Oversold"],
                    ))

        results.sort(key=lambda x: x.rsi or 100)
        return results
