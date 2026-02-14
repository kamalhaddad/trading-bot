# Stock Scalping Trading Bot

A multi-strategy stock scalping bot designed for aggressive intraday trading with comprehensive risk management.

## Features

- **Multi-Strategy Engine**: Momentum breakout, mean reversion (RSI/Bollinger), and VWAP bounce strategies
- **Real-Time Data**: WebSocket-based market data via Alpaca
- **Risk Management**: Position sizing, daily loss limits, stop losses, trailing stops
- **Portfolio Controls**: Sector exposure limits, correlation checks
- **Paper Trading**: Full paper trading support for testing
- **Backtesting**: Historical strategy validation

## Quick Start

### 1. Prerequisites

- Python 3.11+
- Alpaca account (paper or live)

### 2. Installation

```bash
# Clone and enter directory
cd trading

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 3. Configuration

```bash
# Copy example environment file
cp .env.example .env

# Edit .env with your Alpaca API credentials
# Get keys from https://app.alpaca.markets/
```

### 4. Run Paper Trading

```bash
python scripts/run_paper.py
```

## Project Structure

```
trading/
├── src/
│   ├── main.py              # Entry point
│   ├── config.py            # Configuration management
│   ├── broker/              # Alpaca integration
│   ├── data/                # Market data & indicators
│   ├── strategies/          # Trading strategies
│   ├── execution/           # Order management
│   ├── risk/                # Risk management
│   └── utils/               # Logging, notifications
├── backtest/                # Backtesting engine
├── config/
│   ├── settings.yaml        # Main configuration
│   └── symbols.yaml         # Watchlist
├── scripts/
│   ├── run_paper.py         # Paper trading script
│   └── run_backtest.py      # Backtesting script
└── docker/                  # Docker deployment
```

## Strategies

### 1. Momentum Breakout
- **Entry**: Price breaks above 5-bar high with 1.5x volume surge
- **Exit**: 0.5% profit target or 0.3% trailing stop
- **Filter**: ATR > 0.5%, price $10-$200, above VWAP

### 2. Mean Reversion (RSI)
- **Entry**: RSI(14) < 25 at lower Bollinger Band
- **Exit**: RSI crosses 50 or price reaches middle band
- **Filter**: Still in uptrend (above SMA 20)

### 3. VWAP Bounce
- **Entry**: Price bounces off VWAP with volume confirmation
- **Exit**: 0.4% profit or 0.2% stop below VWAP
- **Filter**: First 2 hours of session only

## Risk Management

| Parameter | Default | Description |
|-----------|---------|-------------|
| Max Position | 5% | Maximum per-trade allocation |
| Max Positions | 5 | Maximum concurrent positions |
| Daily Loss Limit | 2% | Stop trading threshold |
| Stop Loss | 1% | Per-trade stop loss |
| Trailing Stop | 0.3% | Lock in profits |

## Configuration

Edit `config/settings.yaml`:

```yaml
broker:
  paper_trading: true  # ALWAYS start with paper trading

trading:
  capital: 30000
  max_position_pct: 0.05
  max_positions: 5

strategies:
  momentum:
    enabled: true
    weight: 0.4
  mean_reversion:
    enabled: true
    weight: 0.3
  vwap:
    enabled: true
    weight: 0.3
```

## Backtesting

```bash
# Run 90-day backtest
python scripts/run_backtest.py --start 2024-01-01 --end 2024-03-31

# Backtest specific symbols
python scripts/run_backtest.py --symbols AAPL MSFT NVDA
```

## Docker Deployment

```bash
cd docker
docker-compose up -d
```

## AWS Deployment

1. Launch EC2 instance (t3.medium recommended)
2. Install Docker
3. Clone repository
4. Configure `.env` with API credentials
5. Run with docker-compose

## Monitoring

- Logs: `logs/trading.log`
- Slack notifications (optional): Configure webhook in `.env`

## Safety Guidelines

1. **ALWAYS** start with paper trading
2. Test for at least 2 weeks before live trading
3. Start live trading with 10% of intended capital
4. Monitor daily, especially first few weeks
5. Never risk more than you can afford to lose

## License

MIT

## Disclaimer

This software is for educational purposes only. Trading involves significant risk of loss. Past performance does not guarantee future results. Always do your own research and consult a financial advisor before trading.
