#!/usr/bin/env python3
"""Run backtesting on historical data."""

import argparse
import sys
from datetime import datetime, timedelta
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from backtest.engine import BacktestEngine
from backtest.data_loader import HistoricalDataLoader
from src.config import Config


def main():
    parser = argparse.ArgumentParser(description="Backtest trading strategies")
    parser.add_argument(
        "--config",
        "-c",
        default="config/settings.yaml",
        help="Path to configuration file",
    )
    parser.add_argument(
        "--start",
        "-s",
        default=(datetime.now() - timedelta(days=90)).strftime("%Y-%m-%d"),
        help="Start date (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--end",
        "-e",
        default=datetime.now().strftime("%Y-%m-%d"),
        help="End date (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--symbols",
        nargs="+",
        default=None,
        help="Symbols to backtest (default: use config)",
    )
    parser.add_argument(
        "--capital",
        type=float,
        default=30000,
        help="Starting capital",
    )

    args = parser.parse_args()

    print(f"Loading configuration from {args.config}")
    config = Config.load(args.config)

    symbols = args.symbols or config.symbols[:10]  # Limit for backtesting

    print(f"Backtesting {len(symbols)} symbols from {args.start} to {args.end}")
    print(f"Starting capital: ${args.capital:,.2f}")
    print()

    # Initialize backtest engine
    engine = BacktestEngine(
        config=config,
        start_date=args.start,
        end_date=args.end,
        initial_capital=args.capital,
    )

    # Load historical data
    loader = HistoricalDataLoader(
        api_key=config.broker.api_key,
        api_secret=config.broker.api_secret,
    )

    print("Loading historical data...")
    for symbol in symbols:
        try:
            data = loader.load(symbol, args.start, args.end)
            if data is not None and not data.empty:
                engine.add_data(symbol, data)
                print(f"  Loaded {len(data)} bars for {symbol}")
        except Exception as e:
            print(f"  Failed to load {symbol}: {e}")

    print()
    print("Running backtest...")
    results = engine.run()

    print()
    print("=" * 60)
    print("BACKTEST RESULTS")
    print("=" * 60)
    print(f"Total Return: {results['total_return_pct']:.2f}%")
    print(f"Sharpe Ratio: {results['sharpe_ratio']:.2f}")
    print(f"Max Drawdown: {results['max_drawdown_pct']:.2f}%")
    print(f"Win Rate: {results['win_rate']*100:.1f}%")
    print(f"Total Trades: {results['total_trades']}")
    print(f"Profit Factor: {results['profit_factor']:.2f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
