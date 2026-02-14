"""Configuration management for the trading bot."""

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv


@dataclass
class BrokerConfig:
    """Broker configuration."""
    name: str = "alpaca"
    paper_trading: bool = True
    api_key: str = ""
    api_secret: str = ""
    base_url: str = "https://paper-api.alpaca.markets"
    data_url: str = "https://data.alpaca.markets"


@dataclass
class TradingConfig:
    """Trading parameters."""
    capital: float = 30000.0
    max_position_pct: float = 0.05
    max_positions: int = 5
    daily_loss_limit_pct: float = 0.02
    min_price: float = 10.0
    max_price: float = 200.0


@dataclass
class StrategyConfig:
    """Individual strategy configuration."""
    enabled: bool = True
    weight: float = 0.33
    params: dict = field(default_factory=dict)


@dataclass
class RiskConfig:
    """Risk management configuration."""
    stop_loss_pct: float = 0.01
    trailing_stop_pct: float = 0.003
    take_profit_pct: float = 0.005
    max_daily_trades: int = 50
    cooldown_after_loss: int = 60
    max_correlation: float = 0.7


@dataclass
class MarketHoursConfig:
    """Market hours configuration."""
    timezone: str = "America/New_York"
    start: str = "09:30"
    end: str = "16:00"
    pre_market: bool = False
    after_hours: bool = False


@dataclass
class LoggingConfig:
    """Logging configuration."""
    level: str = "INFO"
    file: str = "logs/trading.log"
    max_size_mb: int = 100
    backup_count: int = 5


@dataclass
class Config:
    """Main configuration container."""
    broker: BrokerConfig = field(default_factory=BrokerConfig)
    trading: TradingConfig = field(default_factory=TradingConfig)
    strategies: dict[str, StrategyConfig] = field(default_factory=dict)
    risk: RiskConfig = field(default_factory=RiskConfig)
    market_hours: MarketHoursConfig = field(default_factory=MarketHoursConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    symbols: list[str] = field(default_factory=list)

    @classmethod
    def load(
        cls,
        settings_path: str | Path = "config/settings.yaml",
        symbols_path: str | Path = "config/symbols.yaml",
        env_file: str | Path | None = ".env",
    ) -> "Config":
        """Load configuration from YAML files and environment variables."""
        # Load environment variables
        if env_file and Path(env_file).exists():
            load_dotenv(env_file)

        # Load settings
        settings = cls._load_yaml(settings_path)
        symbols_data = cls._load_yaml(symbols_path)

        # Build configuration
        config = cls()

        # Broker config
        broker_data = settings.get("broker", {})
        config.broker = BrokerConfig(
            name=broker_data.get("name", "alpaca"),
            paper_trading=broker_data.get("paper_trading", True),
            api_key=os.getenv(broker_data.get("api_key_env", "ALPACA_API_KEY"), ""),
            api_secret=os.getenv(broker_data.get("api_secret_env", "ALPACA_API_SECRET"), ""),
            base_url=broker_data.get("base_url", "https://paper-api.alpaca.markets"),
            data_url=broker_data.get("data_url", "https://data.alpaca.markets"),
        )

        # Trading config
        trading_data = settings.get("trading", {})
        config.trading = TradingConfig(
            capital=trading_data.get("capital", 30000.0),
            max_position_pct=trading_data.get("max_position_pct", 0.05),
            max_positions=trading_data.get("max_positions", 5),
            daily_loss_limit_pct=trading_data.get("daily_loss_limit_pct", 0.02),
            min_price=trading_data.get("min_price", 10.0),
            max_price=trading_data.get("max_price", 200.0),
        )

        # Strategy configs
        strategies_data = settings.get("strategies", {})
        for name, strategy_data in strategies_data.items():
            params = {k: v for k, v in strategy_data.items() if k not in ("enabled", "weight")}
            config.strategies[name] = StrategyConfig(
                enabled=strategy_data.get("enabled", True),
                weight=strategy_data.get("weight", 0.33),
                params=params,
            )

        # Risk config
        risk_data = settings.get("risk", {})
        config.risk = RiskConfig(
            stop_loss_pct=risk_data.get("stop_loss_pct", 0.01),
            trailing_stop_pct=risk_data.get("trailing_stop_pct", 0.003),
            take_profit_pct=risk_data.get("take_profit_pct", 0.005),
            max_daily_trades=risk_data.get("max_daily_trades", 50),
            cooldown_after_loss=risk_data.get("cooldown_after_loss", 60),
            max_correlation=risk_data.get("max_correlation", 0.7),
        )

        # Market hours config
        hours_data = settings.get("market_hours", {})
        config.market_hours = MarketHoursConfig(
            timezone=hours_data.get("timezone", "America/New_York"),
            start=hours_data.get("start", "09:30"),
            end=hours_data.get("end", "16:00"),
            pre_market=hours_data.get("pre_market", False),
            after_hours=hours_data.get("after_hours", False),
        )

        # Logging config
        log_data = settings.get("logging", {})
        config.logging = LoggingConfig(
            level=log_data.get("level", "INFO"),
            file=log_data.get("file", "logs/trading.log"),
            max_size_mb=log_data.get("max_size_mb", 100),
            backup_count=log_data.get("backup_count", 5),
        )

        # Symbols - load all groups except metadata sections
        config.symbols = []
        if symbols_data:
            exclude_keys = {"screening", "gap_scanner", "sector_etfs"}
            for key, value in symbols_data.items():
                if key not in exclude_keys and isinstance(value, list):
                    config.symbols.extend(value)
            # Remove duplicates while preserving order
            seen = set()
            config.symbols = [s for s in config.symbols if not (s in seen or seen.add(s))]

        return config

    @staticmethod
    def _load_yaml(path: str | Path) -> dict[str, Any]:
        """Load a YAML file."""
        path = Path(path)
        if not path.exists():
            return {}
        with open(path) as f:
            return yaml.safe_load(f) or {}

    def get_strategy_params(self, strategy_name: str) -> dict:
        """Get parameters for a specific strategy."""
        if strategy_name in self.strategies:
            return self.strategies[strategy_name].params
        return {}

    def is_strategy_enabled(self, strategy_name: str) -> bool:
        """Check if a strategy is enabled."""
        if strategy_name in self.strategies:
            return self.strategies[strategy_name].enabled
        return False

    def get_strategy_weight(self, strategy_name: str) -> float:
        """Get weight for a strategy."""
        if strategy_name in self.strategies:
            return self.strategies[strategy_name].weight
        return 0.0
