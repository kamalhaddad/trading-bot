"""Historical data loading for backtesting."""

from datetime import datetime, timedelta

import pandas as pd

from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit


class HistoricalDataLoader:
    """Loads historical data from Alpaca for backtesting."""

    def __init__(self, api_key: str, api_secret: str):
        self.client = StockHistoricalDataClient(
            api_key=api_key,
            secret_key=api_secret,
        )

    def load(
        self,
        symbol: str,
        start_date: str,
        end_date: str,
        timeframe: str = "1Min",
    ) -> pd.DataFrame | None:
        """Load historical bar data.

        Args:
            symbol: Stock symbol
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            timeframe: Bar timeframe (1Min, 5Min, 1Hour, 1Day)

        Returns:
            DataFrame with OHLCV data
        """
        tf = self._parse_timeframe(timeframe)

        request = StockBarsRequest(
            symbol_or_symbols=symbol,
            timeframe=tf,
            start=datetime.strptime(start_date, "%Y-%m-%d"),
            end=datetime.strptime(end_date, "%Y-%m-%d"),
        )

        try:
            bars = self.client.get_stock_bars(request)

            # Access the data from BarSet
            try:
                bar_list = bars[symbol]
            except (KeyError, TypeError):
                return None

            if not bar_list:
                return None

            data = []
            for bar in bar_list:
                data.append({
                    "timestamp": bar.timestamp,
                    "open": bar.open,
                    "high": bar.high,
                    "low": bar.low,
                    "close": bar.close,
                    "volume": bar.volume,
                    "vwap": bar.vwap if hasattr(bar, "vwap") else None,
                })

            df = pd.DataFrame(data)
            df.set_index("timestamp", inplace=True)
            return df

        except Exception as e:
            print(f"Error loading data for {symbol}: {e}")
            return None

    def load_multiple(
        self,
        symbols: list[str],
        start_date: str,
        end_date: str,
        timeframe: str = "1Min",
    ) -> dict[str, pd.DataFrame]:
        """Load historical data for multiple symbols.

        Args:
            symbols: List of stock symbols
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            timeframe: Bar timeframe

        Returns:
            Dict mapping symbol to DataFrame
        """
        result = {}
        for symbol in symbols:
            df = self.load(symbol, start_date, end_date, timeframe)
            if df is not None and not df.empty:
                result[symbol] = df
        return result

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
