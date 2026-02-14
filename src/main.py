"""Main entry point for the trading bot."""

import asyncio
import signal
import sys
from datetime import datetime, time
from zoneinfo import ZoneInfo

from .config import Config
from .broker.alpaca_client import AlpacaClient
from .data.market_data import MarketDataManager
from .data.indicators import TechnicalIndicators
from .data.screener import StockScreener, ScreenerCriteria
from .strategies.strategy_manager import StrategyManager
from .risk.risk_manager import RiskManager
from .risk.portfolio import PortfolioManager
from .execution.order_manager import OrderManager
from .execution.position_manager import PositionManager
from .utils.logger import setup_logger, get_logger
from .utils.notifications import NotificationManager


logger = get_logger(__name__)


class TradingBot:
    """Main trading bot orchestrator."""

    def __init__(self, config: Config):
        self.config = config
        self._running = False
        self._shutdown_event = asyncio.Event()

        # Components (initialized in start())
        self.broker: AlpacaClient | None = None
        self.market_data: MarketDataManager | None = None
        self.strategy_manager: StrategyManager | None = None
        self.risk_manager: RiskManager | None = None
        self.portfolio_manager: PortfolioManager | None = None
        self.order_manager: OrderManager | None = None
        self.position_manager: PositionManager | None = None
        self.screener: StockScreener | None = None
        self.notifications: NotificationManager | None = None

    async def start(self) -> None:
        """Start the trading bot."""
        logger.info("Starting trading bot...")

        # Initialize broker
        self.broker = AlpacaClient(
            api_key=self.config.broker.api_key,
            api_secret=self.config.broker.api_secret,
            paper=self.config.broker.paper_trading,
        )
        await self.broker.connect()

        # Initialize market data
        self.market_data = MarketDataManager(self.broker)
        await self.market_data.start(self.config.symbols)

        # Initialize strategy manager
        self.strategy_manager = StrategyManager(self.config, self.market_data)

        # Initialize risk management
        self.risk_manager = RiskManager(self.config, self.broker)
        await self.risk_manager.initialize()

        self.portfolio_manager = PortfolioManager(
            max_sector_exposure=0.4,
            max_correlation=self.config.risk.max_correlation,
            max_position_pct=self.config.trading.max_position_pct,
        )

        # Initialize execution
        self.order_manager = OrderManager(
            self.broker,
            self.risk_manager,
        )
        await self.order_manager.start()

        self.position_manager = PositionManager(
            self.broker,
            trailing_stop_pct=self.config.risk.trailing_stop_pct,
        )
        await self.position_manager.sync_with_broker()

        # Initialize screener
        self.screener = StockScreener(
            self.market_data,
            TechnicalIndicators(),
            ScreenerCriteria(
                min_price=self.config.trading.min_price,
                max_price=self.config.trading.max_price,
            ),
        )

        # Initialize notifications
        self.notifications = NotificationManager(enabled=False)  # Enable via config
        await self.notifications.start()

        # Set up order fill callback
        self.order_manager.on_fill(self._on_order_fill)

        self._running = True

        await self.notifications.notify_system_status(
            "started",
            f"Paper trading: {self.config.broker.paper_trading}",
        )

        logger.info(
            "Trading bot started",
            paper=self.config.broker.paper_trading,
            symbols=len(self.config.symbols),
        )

    async def stop(self) -> None:
        """Stop the trading bot gracefully."""
        logger.info("Stopping trading bot...")
        self._running = False

        # Cancel all pending orders
        if self.order_manager:
            await self.order_manager.cancel_all_orders()

        # Stop market data
        if self.market_data:
            await self.market_data.stop()

        # Disconnect broker
        if self.broker:
            await self.broker.disconnect()

        # Stop notifications
        if self.notifications:
            await self.notifications.notify_system_status("stopped", "Graceful shutdown")
            await self.notifications.stop()

        logger.info("Trading bot stopped")

    async def run(self) -> None:
        """Main trading loop."""
        logger.info("Entering main trading loop")

        while self._running:
            try:
                # Check if market is open
                if not self.broker.is_market_open():
                    await self._wait_for_market_open()
                    continue

                # Run one trading cycle
                await self._trading_cycle()

                # Wait before next cycle (1 second)
                await asyncio.sleep(1)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("Error in trading loop", error=str(e))
                await asyncio.sleep(5)

    async def _trading_cycle(self) -> None:
        """Execute one trading cycle."""
        # Update metrics
        await self.risk_manager.update_metrics()

        # Check if trading is halted
        if self.risk_manager.is_trading_halted():
            return

        # Sync positions with broker
        await self.position_manager.sync_with_broker()

        # Check exits for existing positions
        await self._check_exits()

        # Look for new opportunities
        await self._look_for_entries()

    async def _check_exits(self) -> None:
        """Check if any positions should be exited."""
        for position in self.position_manager.get_all_positions():
            symbol = position.symbol
            current_price = self.market_data.get_current_price(symbol)

            if current_price is None:
                continue

            # Check stop/target triggers
            stop_trigger = self.position_manager.check_stops(symbol, current_price)

            if stop_trigger:
                logger.info(
                    "Stop triggered",
                    symbol=symbol,
                    type=stop_trigger["type"],
                    price=current_price,
                )
                await self._exit_position(symbol, current_price, stop_trigger["type"])
                continue

            # Check strategy exit signals
            exit_signal = self.strategy_manager.check_exits(
                symbol=symbol,
                entry_price=position.entry_price,
                position_side="long",
                entry_strategy=position.strategy,
            )

            if exit_signal:
                logger.info(
                    "Strategy exit signal",
                    symbol=symbol,
                    reason=exit_signal.reason,
                )
                await self._exit_position(symbol, current_price, exit_signal.reason)

    async def _look_for_entries(self) -> None:
        """Look for new trading opportunities."""
        # Get current positions
        positions = await self.broker.get_positions()

        # Don't look for entries if at max positions
        if len(positions) >= self.config.trading.max_positions:
            return

        # Analyze all symbols
        analyses = self.strategy_manager.get_all_signals(self.config.symbols)

        for analysis in analyses:
            if analysis.best_signal is None:
                continue

            signal = analysis.best_signal

            # Skip if already in position
            if self.position_manager.has_position(signal.symbol):
                continue

            # Check portfolio constraints
            position_value = self.config.trading.capital * self.config.trading.max_position_pct
            account = await self.broker.get_account()

            approved, reason = self.portfolio_manager.check_new_position(
                symbol=signal.symbol,
                position_value=position_value,
                current_positions=positions,
                equity=account.equity,
            )

            if not approved:
                logger.debug(
                    "Position rejected by portfolio manager",
                    symbol=signal.symbol,
                    reason=reason,
                )
                continue

            # Assess risk
            risk_assessment = self.risk_manager.assess_trade(signal, positions)

            if not risk_assessment.approved:
                logger.debug(
                    "Position rejected by risk manager",
                    symbol=signal.symbol,
                    reason=risk_assessment.rejection_reason,
                )
                continue

            # Execute the trade
            logger.info(
                "Executing signal",
                symbol=signal.symbol,
                strategy=signal.strategy_name,
                confidence=signal.confidence,
                shares=risk_assessment.shares,
            )

            ticket = await self.order_manager.execute_signal(signal, risk_assessment)

            if ticket.status == "submitted":
                # Update strategy state
                self.strategy_manager.set_position_state(
                    symbol=signal.symbol,
                    in_position=True,
                    entry_price=signal.price,
                    position_side="long",
                    strategy_name=signal.strategy_name,
                )

            # Only take one trade per cycle to manage risk
            break

    async def _exit_position(self, symbol: str, price: float, reason: str) -> None:
        """Exit a position."""
        position = self.position_manager.get_position(symbol)
        if position is None:
            return

        ticket = await self.order_manager.execute_exit(symbol, position.qty)

        if ticket and ticket.status == "submitted":
            # Position will be closed when order fills
            pass

    def _on_order_fill(self, ticket) -> None:
        """Handle order fill events."""
        signal = ticket.signal

        if signal.signal_type.value == "buy":
            # Open position tracking
            self.position_manager.open_position(
                signal=signal,
                fill_price=ticket.fill_price,
                qty=ticket.filled_qty,
                stop_loss=signal.suggested_stop_loss,
                take_profit=signal.suggested_take_profit,
            )

            # Update strategy state
            self.strategy_manager.set_position_state(
                symbol=signal.symbol,
                in_position=True,
                entry_price=ticket.fill_price,
                position_side="long",
                strategy_name=signal.strategy_name,
            )

        elif signal.signal_type.value == "sell":
            # Close position tracking
            closed = self.position_manager.close_position(
                symbol=signal.symbol,
                exit_price=ticket.fill_price,
                exit_reason=signal.reason,
            )

            if closed:
                # Record trade for risk management
                self.risk_manager.record_trade(closed.pnl, closed.pnl > 0)

                # Update strategy state
                self.strategy_manager.set_position_state(
                    symbol=signal.symbol,
                    in_position=False,
                )

    async def _wait_for_market_open(self) -> None:
        """Wait for market to open."""
        clock = await self.broker.get_clock()
        next_open = clock.get("next_open")

        if next_open:
            logger.info("Market closed, waiting for open", next_open=str(next_open))

        # Check every minute
        await asyncio.sleep(60)


async def main(config_path: str = "config/settings.yaml") -> None:
    """Main entry point."""
    # Load configuration
    config = Config.load(config_path)

    # Setup logging
    setup_logger(
        level=config.logging.level,
        log_file=config.logging.file,
    )

    logger.info("Configuration loaded", symbols=len(config.symbols))

    # Create and run bot
    bot = TradingBot(config)

    # Handle shutdown signals
    loop = asyncio.get_event_loop()

    def shutdown_handler():
        logger.info("Shutdown signal received")
        asyncio.create_task(bot.stop())

    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, shutdown_handler)

    try:
        await bot.start()
        await bot.run()
    except KeyboardInterrupt:
        pass
    finally:
        await bot.stop()


def run() -> None:
    """Entry point for console script."""
    import argparse

    parser = argparse.ArgumentParser(description="Stock Scalping Trading Bot")
    parser.add_argument(
        "--config",
        "-c",
        default="config/settings.yaml",
        help="Path to configuration file",
    )
    parser.add_argument(
        "--paper",
        action="store_true",
        help="Force paper trading mode",
    )

    args = parser.parse_args()

    asyncio.run(main(args.config))


if __name__ == "__main__":
    run()
