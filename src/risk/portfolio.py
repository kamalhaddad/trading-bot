"""Portfolio-level risk management."""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

import numpy as np

from ..broker.base import Position
from ..utils.logger import get_logger


logger = get_logger(__name__)


@dataclass
class PortfolioMetrics:
    """Portfolio-level metrics."""
    total_value: float = 0.0
    total_unrealized_pnl: float = 0.0
    total_unrealized_pnl_pct: float = 0.0
    largest_position_pct: float = 0.0
    sector_exposure: dict[str, float] = field(default_factory=dict)
    concentration_score: float = 0.0  # 0-1, higher = more concentrated
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class CorrelationResult:
    """Result of correlation check."""
    symbol1: str
    symbol2: str
    correlation: float
    is_correlated: bool


class PortfolioManager:
    """Manages portfolio-level risk and analysis."""

    def __init__(
        self,
        max_sector_exposure: float = 0.4,
        max_correlation: float = 0.7,
        max_position_pct: float = 0.2,
    ):
        self.max_sector_exposure = max_sector_exposure
        self.max_correlation = max_correlation
        self.max_position_pct = max_position_pct

        # Symbol to sector mapping (simplified)
        self._sector_map: dict[str, str] = {
            "AAPL": "Technology",
            "MSFT": "Technology",
            "GOOGL": "Technology",
            "AMZN": "Consumer",
            "META": "Technology",
            "NVDA": "Technology",
            "TSLA": "Automotive",
            "AMD": "Technology",
            "NFLX": "Entertainment",
            "SPY": "Index",
            "QQQ": "Index",
            "INTC": "Technology",
            "CRM": "Technology",
            "ORCL": "Technology",
            "ADBE": "Technology",
            "PYPL": "Fintech",
            "SQ": "Fintech",
            "COIN": "Crypto",
            "SHOP": "E-commerce",
            "UBER": "Transportation",
        }

        # Historical returns for correlation (would be populated from data)
        self._returns_cache: dict[str, list[float]] = {}

    def analyze_portfolio(self, positions: list[Position]) -> PortfolioMetrics:
        """Analyze current portfolio.

        Args:
            positions: List of current positions

        Returns:
            PortfolioMetrics with analysis
        """
        if not positions:
            return PortfolioMetrics()

        total_value = sum(p.market_value for p in positions)
        total_unrealized_pnl = sum(p.unrealized_pnl for p in positions)

        # Calculate position weights
        weights = {p.symbol: p.market_value / total_value for p in positions}
        largest_position_pct = max(weights.values()) if weights else 0

        # Calculate sector exposure
        sector_exposure: dict[str, float] = {}
        for p in positions:
            sector = self._sector_map.get(p.symbol, "Other")
            sector_exposure[sector] = sector_exposure.get(sector, 0) + weights.get(p.symbol, 0)

        # Calculate concentration (Herfindahl Index)
        concentration = sum(w ** 2 for w in weights.values())

        total_cost = sum(p.avg_entry_price * p.qty for p in positions)
        pnl_pct = total_unrealized_pnl / total_cost if total_cost > 0 else 0

        return PortfolioMetrics(
            total_value=total_value,
            total_unrealized_pnl=total_unrealized_pnl,
            total_unrealized_pnl_pct=pnl_pct,
            largest_position_pct=largest_position_pct,
            sector_exposure=sector_exposure,
            concentration_score=concentration,
        )

    def check_new_position(
        self,
        symbol: str,
        position_value: float,
        current_positions: list[Position],
        equity: float,
    ) -> tuple[bool, str | None]:
        """Check if a new position would violate portfolio rules.

        Args:
            symbol: Symbol to add
            position_value: Value of new position
            current_positions: Current positions
            equity: Total equity

        Returns:
            (approved, rejection_reason)
        """
        # Check position size
        position_pct = position_value / equity
        if position_pct > self.max_position_pct:
            return False, f"Position too large: {position_pct*100:.1f}% > {self.max_position_pct*100:.1f}%"

        # Check sector exposure
        sector = self._sector_map.get(symbol, "Other")
        current_sector_exposure = self._get_sector_exposure(current_positions, equity)
        new_sector_exposure = current_sector_exposure.get(sector, 0) + position_pct

        if new_sector_exposure > self.max_sector_exposure:
            return False, f"Sector exposure too high: {sector} at {new_sector_exposure*100:.1f}%"

        # Check correlation with existing positions
        for pos in current_positions:
            correlation = self._estimate_correlation(symbol, pos.symbol)
            if correlation > self.max_correlation:
                return False, f"Too correlated with {pos.symbol}: {correlation:.2f}"

        return True, None

    def _get_sector_exposure(
        self,
        positions: list[Position],
        equity: float,
    ) -> dict[str, float]:
        """Get current sector exposure."""
        exposure: dict[str, float] = {}
        for p in positions:
            sector = self._sector_map.get(p.symbol, "Other")
            pct = p.market_value / equity if equity > 0 else 0
            exposure[sector] = exposure.get(sector, 0) + pct
        return exposure

    def _estimate_correlation(self, symbol1: str, symbol2: str) -> float:
        """Estimate correlation between two symbols.

        Uses cached returns if available, otherwise uses heuristics.
        """
        # Check cache
        if symbol1 in self._returns_cache and symbol2 in self._returns_cache:
            r1 = self._returns_cache[symbol1]
            r2 = self._returns_cache[symbol2]
            if len(r1) >= 20 and len(r2) >= 20:
                min_len = min(len(r1), len(r2))
                return np.corrcoef(r1[-min_len:], r2[-min_len:])[0, 1]

        # Use sector-based heuristic
        sector1 = self._sector_map.get(symbol1, "Other")
        sector2 = self._sector_map.get(symbol2, "Other")

        if sector1 == sector2:
            # Same sector - assume moderate to high correlation
            if sector1 == "Technology":
                return 0.75
            elif sector1 == "Index":
                return 0.9
            else:
                return 0.65
        elif sector1 == "Index" or sector2 == "Index":
            # Index correlates with most stocks
            return 0.6
        else:
            # Different sectors - assume low correlation
            return 0.3

    def update_returns(self, symbol: str, returns: list[float]) -> None:
        """Update cached returns for correlation calculation.

        Args:
            symbol: Symbol
            returns: List of daily returns
        """
        self._returns_cache[symbol] = returns

    def get_rebalancing_suggestions(
        self,
        positions: list[Position],
        equity: float,
        target_weights: dict[str, float] | None = None,
    ) -> list[dict[str, Any]]:
        """Get suggestions for rebalancing portfolio.

        Args:
            positions: Current positions
            equity: Total equity
            target_weights: Optional target weights per symbol

        Returns:
            List of rebalancing suggestions
        """
        suggestions = []

        if not positions:
            return suggestions

        # Calculate current weights
        current_weights = {p.symbol: p.market_value / equity for p in positions}

        # Check for oversized positions
        for symbol, weight in current_weights.items():
            if weight > self.max_position_pct:
                excess_pct = weight - self.max_position_pct
                excess_value = excess_pct * equity
                suggestions.append({
                    "action": "reduce",
                    "symbol": symbol,
                    "reason": "Position too large",
                    "current_weight": weight,
                    "target_weight": self.max_position_pct,
                    "reduce_by_value": excess_value,
                })

        # Check sector concentration
        sector_exposure = self._get_sector_exposure(positions, equity)
        for sector, exposure in sector_exposure.items():
            if exposure > self.max_sector_exposure:
                sector_positions = [p for p in positions if self._sector_map.get(p.symbol) == sector]
                suggestions.append({
                    "action": "reduce_sector",
                    "sector": sector,
                    "reason": "Sector overexposed",
                    "current_exposure": exposure,
                    "max_exposure": self.max_sector_exposure,
                    "positions": [p.symbol for p in sector_positions],
                })

        return suggestions

    def set_sector(self, symbol: str, sector: str) -> None:
        """Set sector mapping for a symbol."""
        self._sector_map[symbol] = sector

    def get_sector(self, symbol: str) -> str:
        """Get sector for a symbol."""
        return self._sector_map.get(symbol, "Other")
