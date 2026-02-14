"""Backtesting engine for strategy validation."""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

import pandas as pd
import numpy as np

from src.config import Config
from src.data.indicators import TechnicalIndicators, IndicatorValues
from src.strategies.momentum import MomentumStrategy
from src.strategies.mean_reversion import MeanReversionStrategy
from src.strategies.vwap import VWAPStrategy
from src.strategies.base import Signal, SignalType


@dataclass
class BacktestTrade:
    """Record of a backtest trade."""
    symbol: str
    entry_time: datetime
    entry_price: float
    exit_time: datetime | None = None
    exit_price: float | None = None
    qty: float = 0
    pnl: float = 0
    pnl_pct: float = 0
    strategy: str = ""
    exit_reason: str = ""


@dataclass
class BacktestResults:
    """Results from a backtest run."""
    total_return: float = 0.0
    total_return_pct: float = 0.0
    sharpe_ratio: float = 0.0
    max_drawdown: float = 0.0
    max_drawdown_pct: float = 0.0
    win_rate: float = 0.0
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    profit_factor: float = 0.0
    avg_trade_pnl: float = 0.0
    avg_win: float = 0.0
    avg_loss: float = 0.0
    trades: list[BacktestTrade] = field(default_factory=list)
    equity_curve: list[float] = field(default_factory=list)


class BacktestEngine:
    """Engine for backtesting trading strategies."""

    def __init__(
        self,
        config: Config,
        start_date: str,
        end_date: str,
        initial_capital: float = 30000,
    ):
        self.config = config
        self.start_date = pd.Timestamp(start_date).to_pydatetime()
        self.end_date = pd.Timestamp(end_date).to_pydatetime() + pd.Timedelta(days=1)  # Include end date
        self.initial_capital = initial_capital

        # Data storage
        self._data: dict[str, pd.DataFrame] = {}

        # Initialize strategies
        self.strategies = {}
        self._init_strategies()

        # Technical indicators
        self.indicators = TechnicalIndicators()

        # State
        self._capital = initial_capital
        self._positions: dict[str, BacktestTrade] = {}
        self._trades: list[BacktestTrade] = []
        self._equity_curve: list[float] = []

    def _init_strategies(self) -> None:
        """Initialize strategies from config."""
        if self.config.is_strategy_enabled("momentum"):
            params = self.config.get_strategy_params("momentum")
            self.strategies["momentum"] = MomentumStrategy(params)

        if self.config.is_strategy_enabled("mean_reversion"):
            params = self.config.get_strategy_params("mean_reversion")
            self.strategies["mean_reversion"] = MeanReversionStrategy(params)

        if self.config.is_strategy_enabled("vwap"):
            params = self.config.get_strategy_params("vwap")
            self.strategies["vwap"] = VWAPStrategy(params)

    def add_data(self, symbol: str, data: pd.DataFrame) -> None:
        """Add historical data for a symbol.

        Args:
            symbol: Symbol
            data: DataFrame with OHLCV columns
        """
        # Ensure proper columns
        required = ["open", "high", "low", "close", "volume"]
        for col in required:
            if col not in data.columns:
                raise ValueError(f"Missing column: {col}")

        self._data[symbol] = data.copy()

    def run(self) -> dict[str, Any]:
        """Run the backtest.

        Returns:
            Dict with backtest results
        """
        if not self._data:
            raise ValueError("No data loaded")

        # Get all unique timestamps across all symbols
        all_timestamps = set()
        for df in self._data.values():
            all_timestamps.update(df.index.tolist())

        timestamps = sorted(all_timestamps)
        # Filter to date range (handle tz-aware timestamps)
        filtered = []
        for t in timestamps:
            t_naive = t.replace(tzinfo=None) if hasattr(t, 'tzinfo') and t.tzinfo else t
            if self.start_date <= t_naive <= self.end_date:
                filtered.append(t)
        timestamps = filtered

        print(f"Simulating {len(timestamps)} time steps...")

        # Simulate each time step
        for i, timestamp in enumerate(timestamps):
            self._simulate_step(timestamp, i)

        # Close any remaining positions
        self._close_all_positions(timestamps[-1] if timestamps else datetime.now())

        # Calculate results
        results = self._calculate_results()

        return results

    def _simulate_step(self, timestamp: pd.Timestamp, step: int) -> None:
        """Simulate one time step."""
        # Check exits for existing positions
        for symbol in list(self._positions.keys()):
            if symbol not in self._data:
                continue

            df = self._data[symbol]
            if timestamp not in df.index:
                continue

            bar = df.loc[timestamp]
            position = self._positions[symbol]

            # Calculate indicators
            hist = df.loc[:timestamp].tail(100)
            indicators = self.indicators.calculate(hist, symbol)

            # Check strategy exits
            for name, strategy in self.strategies.items():
                if name != position.strategy:
                    continue

                exit_signal = strategy.should_exit(
                    symbol=symbol,
                    bars=hist,
                    indicators=indicators,
                    current_price=bar["close"],
                    entry_price=position.entry_price,
                    position_side="long",
                )

                if exit_signal:
                    self._close_position(symbol, bar["close"], timestamp, exit_signal.reason)
                    break

            # Check stop loss
            if symbol in self._positions:
                position = self._positions[symbol]
                pnl_pct = (bar["close"] - position.entry_price) / position.entry_price

                if pnl_pct <= -self.config.risk.stop_loss_pct:
                    self._close_position(symbol, bar["close"], timestamp, "stop_loss")

        # Look for new entries
        if len(self._positions) >= self.config.trading.max_positions:
            return

        for symbol, df in self._data.items():
            if symbol in self._positions:
                continue

            if timestamp not in df.index:
                continue

            bar = df.loc[timestamp]
            hist = df.loc[:timestamp].tail(100)

            if len(hist) < 30:
                continue

            indicators = self.indicators.calculate(hist, symbol)

            # Check each strategy for entry
            for name, strategy in self.strategies.items():
                signal = strategy.analyze(
                    symbol=symbol,
                    bars=hist,
                    indicators=indicators,
                    current_price=bar["close"],
                )

                if signal and signal.signal_type == SignalType.BUY:
                    self._open_position(signal, bar["close"], timestamp)
                    break

        # Record equity
        equity = self._calculate_equity(timestamp)
        self._equity_curve.append(equity)

    def _open_position(
        self,
        signal: Signal,
        price: float,
        timestamp: pd.Timestamp,
    ) -> None:
        """Open a new position."""
        # Calculate position size
        position_value = self._capital * self.config.trading.max_position_pct * signal.confidence
        qty = int(position_value / price)

        if qty <= 0:
            return

        trade = BacktestTrade(
            symbol=signal.symbol,
            entry_time=timestamp,
            entry_price=price,
            qty=qty,
            strategy=signal.strategy_name,
        )

        self._positions[signal.symbol] = trade
        self._capital -= qty * price

    def _close_position(
        self,
        symbol: str,
        price: float,
        timestamp: pd.Timestamp,
        reason: str,
    ) -> None:
        """Close a position."""
        if symbol not in self._positions:
            return

        trade = self._positions.pop(symbol)
        trade.exit_time = timestamp
        trade.exit_price = price
        trade.exit_reason = reason
        trade.pnl = (price - trade.entry_price) * trade.qty
        trade.pnl_pct = (price - trade.entry_price) / trade.entry_price

        self._trades.append(trade)
        self._capital += trade.qty * price

    def _close_all_positions(self, timestamp: pd.Timestamp) -> None:
        """Close all open positions."""
        for symbol in list(self._positions.keys()):
            if symbol in self._data:
                df = self._data[symbol]
                if not df.empty:
                    last_price = df["close"].iloc[-1]
                    self._close_position(symbol, last_price, timestamp, "end_of_backtest")

    def _calculate_equity(self, timestamp: pd.Timestamp) -> float:
        """Calculate current equity."""
        equity = self._capital

        for symbol, position in self._positions.items():
            if symbol in self._data:
                df = self._data[symbol]
                if timestamp in df.index:
                    current_price = df.loc[timestamp, "close"]
                    equity += position.qty * current_price

        return equity

    def _calculate_results(self) -> dict[str, Any]:
        """Calculate backtest results."""
        if not self._trades:
            return {
                "total_return": 0,
                "total_return_pct": 0,
                "sharpe_ratio": 0,
                "max_drawdown": 0,
                "max_drawdown_pct": 0,
                "win_rate": 0,
                "total_trades": 0,
                "winning_trades": 0,
                "losing_trades": 0,
                "profit_factor": 0,
                "avg_trade_pnl": 0,
                "avg_win": 0,
                "avg_loss": 0,
                "trades": [],
                "equity_curve": self._equity_curve,
            }

        # Calculate returns
        final_equity = self._equity_curve[-1] if self._equity_curve else self.initial_capital
        total_return = final_equity - self.initial_capital
        total_return_pct = total_return / self.initial_capital * 100

        # Win/loss stats
        wins = [t for t in self._trades if t.pnl > 0]
        losses = [t for t in self._trades if t.pnl <= 0]

        win_rate = len(wins) / len(self._trades) if self._trades else 0

        avg_win = np.mean([t.pnl for t in wins]) if wins else 0
        avg_loss = np.mean([abs(t.pnl) for t in losses]) if losses else 0

        gross_profit = sum(t.pnl for t in wins)
        gross_loss = abs(sum(t.pnl for t in losses))
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf")

        # Calculate Sharpe ratio (simplified)
        if len(self._equity_curve) > 1:
            returns = pd.Series(self._equity_curve).pct_change().dropna()
            if len(returns) > 0 and returns.std() > 0:
                sharpe = returns.mean() / returns.std() * np.sqrt(252)  # Annualized
            else:
                sharpe = 0
        else:
            sharpe = 0

        # Calculate max drawdown
        equity_series = pd.Series(self._equity_curve)
        rolling_max = equity_series.expanding().max()
        drawdown = (equity_series - rolling_max) / rolling_max
        max_drawdown_pct = abs(drawdown.min()) * 100 if len(drawdown) > 0 else 0
        max_drawdown = abs((equity_series - rolling_max).min()) if len(equity_series) > 0 else 0

        return {
            "total_return": total_return,
            "total_return_pct": total_return_pct,
            "sharpe_ratio": sharpe,
            "max_drawdown": max_drawdown,
            "max_drawdown_pct": max_drawdown_pct,
            "win_rate": win_rate,
            "total_trades": len(self._trades),
            "winning_trades": len(wins),
            "losing_trades": len(losses),
            "profit_factor": profit_factor,
            "avg_trade_pnl": np.mean([t.pnl for t in self._trades]),
            "avg_win": avg_win,
            "avg_loss": avg_loss,
            "trades": self._trades,
            "equity_curve": self._equity_curve,
        }
