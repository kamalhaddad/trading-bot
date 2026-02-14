"""Logging setup for the trading bot."""

import logging
import sys
from datetime import datetime
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Any

import structlog


# Global logger storage
_loggers: dict[str, Any] = {}


def setup_logger(
    level: str = "INFO",
    log_file: str | None = "logs/trading.log",
    max_size_mb: int = 100,
    backup_count: int = 5,
) -> None:
    """Set up structured logging for the trading bot.

    Args:
        level: Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Path to log file, or None for console only
        max_size_mb: Maximum size of log file before rotation
        backup_count: Number of backup files to keep
    """
    # Create logs directory if needed
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)

    # Configure standard logging
    log_level = getattr(logging, level.upper(), logging.INFO)

    handlers: list[logging.Handler] = []

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(log_level)
    handlers.append(console_handler)

    # File handler with rotation
    if log_file:
        file_handler = RotatingFileHandler(
            log_file,
            maxBytes=max_size_mb * 1024 * 1024,
            backupCount=backup_count,
        )
        file_handler.setLevel(log_level)
        handlers.append(file_handler)

    # Configure structlog
    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.UnicodeDecoder(),
            structlog.dev.ConsoleRenderer() if sys.stdout.isatty() else structlog.processors.JSONRenderer(),
        ],
        wrapper_class=structlog.make_filtering_bound_logger(log_level),
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(),
        cache_logger_on_first_use=True,
    )

    # Also configure standard logging for third-party libraries
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=handlers,
    )

    # Reduce noise from third-party loggers
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("websockets").setLevel(logging.WARNING)
    logging.getLogger("asyncio").setLevel(logging.WARNING)


def get_logger(name: str) -> structlog.BoundLogger:
    """Get a logger instance for a module.

    Args:
        name: Logger name (usually __name__)

    Returns:
        A structured logger instance
    """
    if name not in _loggers:
        _loggers[name] = structlog.get_logger(name)
    return _loggers[name]


class TradeLogger:
    """Specialized logger for trade events."""

    def __init__(self):
        self.logger = get_logger("trades")

    def log_signal(
        self,
        symbol: str,
        strategy: str,
        signal_type: str,
        price: float,
        confidence: float,
        reason: str,
    ) -> None:
        """Log a trading signal."""
        self.logger.info(
            "signal_generated",
            symbol=symbol,
            strategy=strategy,
            signal_type=signal_type,
            price=price,
            confidence=confidence,
            reason=reason,
        )

    def log_order(
        self,
        order_id: str,
        symbol: str,
        side: str,
        qty: float,
        order_type: str,
        price: float | None = None,
        status: str = "submitted",
    ) -> None:
        """Log an order event."""
        self.logger.info(
            "order_event",
            order_id=order_id,
            symbol=symbol,
            side=side,
            qty=qty,
            order_type=order_type,
            price=price,
            status=status,
        )

    def log_fill(
        self,
        order_id: str,
        symbol: str,
        side: str,
        qty: float,
        fill_price: float,
        commission: float = 0.0,
    ) -> None:
        """Log an order fill."""
        self.logger.info(
            "order_filled",
            order_id=order_id,
            symbol=symbol,
            side=side,
            qty=qty,
            fill_price=fill_price,
            commission=commission,
        )

    def log_position_opened(
        self,
        symbol: str,
        qty: float,
        entry_price: float,
        strategy: str,
    ) -> None:
        """Log position opening."""
        self.logger.info(
            "position_opened",
            symbol=symbol,
            qty=qty,
            entry_price=entry_price,
            strategy=strategy,
        )

    def log_position_closed(
        self,
        symbol: str,
        qty: float,
        entry_price: float,
        exit_price: float,
        pnl: float,
        pnl_pct: float,
        hold_time_seconds: float,
        exit_reason: str,
    ) -> None:
        """Log position closing."""
        self.logger.info(
            "position_closed",
            symbol=symbol,
            qty=qty,
            entry_price=entry_price,
            exit_price=exit_price,
            pnl=pnl,
            pnl_pct=pnl_pct,
            hold_time_seconds=hold_time_seconds,
            exit_reason=exit_reason,
        )

    def log_daily_summary(
        self,
        date: str,
        total_trades: int,
        winning_trades: int,
        total_pnl: float,
        max_drawdown: float,
    ) -> None:
        """Log daily trading summary."""
        win_rate = winning_trades / total_trades if total_trades > 0 else 0
        self.logger.info(
            "daily_summary",
            date=date,
            total_trades=total_trades,
            winning_trades=winning_trades,
            win_rate=win_rate,
            total_pnl=total_pnl,
            max_drawdown=max_drawdown,
        )

    def log_error(
        self,
        error_type: str,
        message: str,
        symbol: str | None = None,
        **extra,
    ) -> None:
        """Log an error."""
        self.logger.error(
            "trading_error",
            error_type=error_type,
            message=message,
            symbol=symbol,
            **extra,
        )


# Singleton trade logger
trade_logger = TradeLogger()
