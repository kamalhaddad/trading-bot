"""Risk management for trading operations."""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any

from ..broker.base import BaseBroker, AccountInfo, Position
from ..config import Config
from ..strategies.base import Signal
from ..utils.logger import get_logger


logger = get_logger(__name__)


@dataclass
class RiskMetrics:
    """Current risk metrics."""
    equity: float = 0.0
    cash: float = 0.0
    buying_power: float = 0.0
    positions_value: float = 0.0
    positions_count: int = 0
    daily_pnl: float = 0.0
    daily_pnl_pct: float = 0.0
    daily_trades: int = 0
    max_daily_pnl: float = 0.0
    min_daily_pnl: float = 0.0
    open_risk: float = 0.0  # Total risk in open positions
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class TradeRisk:
    """Risk assessment for a potential trade."""
    approved: bool
    position_size: float  # Dollar amount
    shares: int
    risk_amount: float  # Max loss in dollars
    risk_pct: float  # Max loss as % of equity
    rejection_reason: str | None = None


@dataclass
class DailyStats:
    """Daily trading statistics."""
    date: str
    starting_equity: float
    current_equity: float
    pnl: float = 0.0
    pnl_pct: float = 0.0
    trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    max_drawdown: float = 0.0
    max_equity: float = 0.0
    last_loss_time: datetime | None = None


class RiskManager:
    """Manages risk for the trading bot."""

    def __init__(self, config: Config, broker: BaseBroker):
        self.config = config
        self.broker = broker

        # Risk parameters from config
        self.max_position_pct = config.trading.max_position_pct
        self.max_positions = config.trading.max_positions
        self.daily_loss_limit_pct = config.trading.daily_loss_limit_pct
        self.stop_loss_pct = config.risk.stop_loss_pct
        self.max_daily_trades = config.risk.max_daily_trades
        self.cooldown_after_loss = config.risk.cooldown_after_loss

        # State
        self._daily_stats: DailyStats | None = None
        self._metrics: RiskMetrics = RiskMetrics()
        self._trading_halted = False
        self._halt_reason: str | None = None

    async def initialize(self) -> None:
        """Initialize risk manager with current account state."""
        account = await self.broker.get_account()
        positions = await self.broker.get_positions()

        today = datetime.now().strftime("%Y-%m-%d")
        self._daily_stats = DailyStats(
            date=today,
            starting_equity=account.equity,
            current_equity=account.equity,
            max_equity=account.equity,
        )

        await self.update_metrics()
        logger.info(
            "Risk manager initialized",
            equity=account.equity,
            positions=len(positions),
        )

    async def update_metrics(self) -> RiskMetrics:
        """Update current risk metrics."""
        account = await self.broker.get_account()
        positions = await self.broker.get_positions()

        positions_value = sum(p.market_value for p in positions)
        open_risk = sum(abs(p.unrealized_pnl) for p in positions if p.unrealized_pnl < 0)

        self._metrics = RiskMetrics(
            equity=account.equity,
            cash=account.cash,
            buying_power=account.buying_power,
            positions_value=positions_value,
            positions_count=len(positions),
            daily_pnl=self._daily_stats.pnl if self._daily_stats else 0,
            daily_pnl_pct=self._daily_stats.pnl_pct if self._daily_stats else 0,
            daily_trades=self._daily_stats.trades if self._daily_stats else 0,
            max_daily_pnl=self._daily_stats.max_equity - self._daily_stats.starting_equity if self._daily_stats else 0,
            min_daily_pnl=self._daily_stats.max_drawdown if self._daily_stats else 0,
            open_risk=open_risk,
        )

        # Update daily stats
        if self._daily_stats:
            self._daily_stats.current_equity = account.equity
            self._daily_stats.pnl = account.equity - self._daily_stats.starting_equity
            self._daily_stats.pnl_pct = self._daily_stats.pnl / self._daily_stats.starting_equity

            # Track max equity and drawdown
            if account.equity > self._daily_stats.max_equity:
                self._daily_stats.max_equity = account.equity
            drawdown = (self._daily_stats.max_equity - account.equity) / self._daily_stats.max_equity
            if drawdown > self._daily_stats.max_drawdown:
                self._daily_stats.max_drawdown = drawdown

        return self._metrics

    def assess_trade(
        self,
        signal: Signal,
        current_positions: list[Position],
    ) -> TradeRisk:
        """Assess risk for a potential trade.

        Args:
            signal: Trading signal
            current_positions: Current open positions

        Returns:
            TradeRisk with approval status and sizing
        """
        # Check if trading is halted
        if self._trading_halted:
            return TradeRisk(
                approved=False,
                position_size=0,
                shares=0,
                risk_amount=0,
                risk_pct=0,
                rejection_reason=f"Trading halted: {self._halt_reason}",
            )

        # Check daily loss limit
        if self._daily_stats:
            if self._daily_stats.pnl_pct <= -self.daily_loss_limit_pct:
                self._halt_trading(f"Daily loss limit reached: {self._daily_stats.pnl_pct*100:.2f}%")
                return TradeRisk(
                    approved=False,
                    position_size=0,
                    shares=0,
                    risk_amount=0,
                    risk_pct=0,
                    rejection_reason="Daily loss limit reached",
                )

        # Check max daily trades
        if self._daily_stats and self._daily_stats.trades >= self.max_daily_trades:
            return TradeRisk(
                approved=False,
                position_size=0,
                shares=0,
                risk_amount=0,
                risk_pct=0,
                rejection_reason=f"Max daily trades reached: {self.max_daily_trades}",
            )

        # Check cooldown after loss
        if self._daily_stats and self._daily_stats.last_loss_time:
            cooldown_end = self._daily_stats.last_loss_time + timedelta(seconds=self.cooldown_after_loss)
            if datetime.now() < cooldown_end:
                remaining = (cooldown_end - datetime.now()).seconds
                return TradeRisk(
                    approved=False,
                    position_size=0,
                    shares=0,
                    risk_amount=0,
                    risk_pct=0,
                    rejection_reason=f"Cooldown after loss: {remaining}s remaining",
                )

        # Check max positions
        if len(current_positions) >= self.max_positions:
            return TradeRisk(
                approved=False,
                position_size=0,
                shares=0,
                risk_amount=0,
                risk_pct=0,
                rejection_reason=f"Max positions reached: {self.max_positions}",
            )

        # Check if already in position for this symbol
        if any(p.symbol == signal.symbol for p in current_positions):
            return TradeRisk(
                approved=False,
                position_size=0,
                shares=0,
                risk_amount=0,
                risk_pct=0,
                rejection_reason="Already in position for this symbol",
            )

        # Calculate position size
        position_size = self._calculate_position_size(signal)

        # Calculate shares
        shares = int(position_size / signal.price)
        if shares <= 0:
            return TradeRisk(
                approved=False,
                position_size=0,
                shares=0,
                risk_amount=0,
                risk_pct=0,
                rejection_reason="Position size too small",
            )

        # Calculate risk
        actual_position = shares * signal.price
        stop_loss = signal.suggested_stop_loss or signal.price * (1 - self.stop_loss_pct)
        risk_per_share = signal.price - stop_loss
        risk_amount = shares * risk_per_share
        risk_pct = risk_amount / self._metrics.equity if self._metrics.equity > 0 else 0

        return TradeRisk(
            approved=True,
            position_size=actual_position,
            shares=shares,
            risk_amount=risk_amount,
            risk_pct=risk_pct,
        )

    def _calculate_position_size(self, signal: Signal) -> float:
        """Calculate position size based on signal and risk parameters."""
        base_size = self._metrics.equity * self.max_position_pct

        # Adjust based on signal confidence
        confidence_factor = signal.confidence if signal.confidence else 0.5
        if signal.suggested_qty_pct:
            confidence_factor = signal.suggested_qty_pct

        # Scale position by confidence
        position_size = base_size * confidence_factor

        # Cap at max position size
        max_size = self._metrics.equity * self.max_position_pct
        position_size = min(position_size, max_size)

        # Ensure we have enough buying power
        position_size = min(position_size, self._metrics.buying_power * 0.95)

        return position_size

    def record_trade(self, pnl: float, is_win: bool) -> None:
        """Record a completed trade.

        Args:
            pnl: Profit/loss from the trade
            is_win: Whether the trade was profitable
        """
        if self._daily_stats:
            self._daily_stats.trades += 1
            if is_win:
                self._daily_stats.winning_trades += 1
            else:
                self._daily_stats.losing_trades += 1
                self._daily_stats.last_loss_time = datetime.now()

            logger.info(
                "Trade recorded",
                pnl=pnl,
                is_win=is_win,
                daily_trades=self._daily_stats.trades,
                daily_pnl=self._daily_stats.pnl,
            )

    def _halt_trading(self, reason: str) -> None:
        """Halt trading."""
        self._trading_halted = True
        self._halt_reason = reason
        logger.warning("Trading halted", reason=reason)

    def resume_trading(self) -> None:
        """Resume trading after halt."""
        self._trading_halted = False
        self._halt_reason = None
        logger.info("Trading resumed")

    def is_trading_halted(self) -> bool:
        """Check if trading is halted."""
        return self._trading_halted

    def get_halt_reason(self) -> str | None:
        """Get reason for trading halt."""
        return self._halt_reason

    def reset_daily(self) -> None:
        """Reset daily statistics (call at start of trading day)."""
        if self._metrics.equity > 0:
            today = datetime.now().strftime("%Y-%m-%d")
            self._daily_stats = DailyStats(
                date=today,
                starting_equity=self._metrics.equity,
                current_equity=self._metrics.equity,
                max_equity=self._metrics.equity,
            )
            self._trading_halted = False
            self._halt_reason = None
            logger.info("Daily stats reset", equity=self._metrics.equity)

    def get_metrics(self) -> RiskMetrics:
        """Get current risk metrics."""
        return self._metrics

    def get_daily_stats(self) -> DailyStats | None:
        """Get daily statistics."""
        return self._daily_stats

    def check_position_risk(
        self,
        position: Position,
        current_price: float,
    ) -> dict[str, Any]:
        """Check risk for an existing position.

        Args:
            position: Current position
            current_price: Current market price

        Returns:
            Dict with risk assessment
        """
        pnl_pct = (current_price - position.avg_entry_price) / position.avg_entry_price

        # Check if stop loss should trigger
        stop_triggered = pnl_pct <= -self.stop_loss_pct

        # Calculate max loss from here
        stop_price = position.avg_entry_price * (1 - self.stop_loss_pct)
        max_loss = (current_price - stop_price) * position.qty

        return {
            "symbol": position.symbol,
            "pnl_pct": pnl_pct,
            "stop_triggered": stop_triggered,
            "max_loss_from_here": max_loss,
            "current_unrealized_pnl": position.unrealized_pnl,
        }
