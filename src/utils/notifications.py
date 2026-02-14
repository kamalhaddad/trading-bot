"""Notification system for trading alerts."""

import asyncio
import os
from dataclasses import dataclass
from datetime import datetime
from typing import Any

import aiohttp

from .logger import get_logger


logger = get_logger(__name__)


@dataclass
class TradeNotification:
    """Trade notification data."""
    symbol: str
    action: str  # BUY, SELL, STOP_LOSS, etc.
    price: float
    qty: float
    pnl: float | None = None
    strategy: str | None = None
    message: str | None = None


class NotificationManager:
    """Manages notifications via Slack/Discord."""

    def __init__(
        self,
        enabled: bool = False,
        slack_webhook_url: str | None = None,
    ):
        self.enabled = enabled
        self.slack_webhook_url = slack_webhook_url or os.getenv("SLACK_WEBHOOK_URL")
        self._session: aiohttp.ClientSession | None = None

    async def start(self) -> None:
        """Initialize the notification manager."""
        if self.enabled and self.slack_webhook_url:
            self._session = aiohttp.ClientSession()
            logger.info("Notification manager started")

    async def stop(self) -> None:
        """Cleanup the notification manager."""
        if self._session:
            await self._session.close()
            self._session = None

    async def notify_trade(self, notification: TradeNotification) -> None:
        """Send a trade notification."""
        if not self.enabled:
            return

        emoji = self._get_emoji(notification.action)
        color = self._get_color(notification.action, notification.pnl)

        message = self._format_trade_message(notification)

        await self._send_slack(
            text=f"{emoji} {notification.action}: {notification.symbol}",
            blocks=[
                {
                    "type": "section",
                    "text": {
                        "type": "mrkdwn",
                        "text": message,
                    },
                },
            ],
            color=color,
        )

    async def notify_error(self, error_type: str, message: str, symbol: str | None = None) -> None:
        """Send an error notification."""
        if not self.enabled:
            return

        text = f"*Error: {error_type}*\n{message}"
        if symbol:
            text += f"\nSymbol: {symbol}"

        await self._send_slack(
            text=text,
            color="danger",
        )

    async def notify_daily_summary(
        self,
        date: str,
        total_trades: int,
        winning_trades: int,
        total_pnl: float,
        starting_equity: float,
        ending_equity: float,
    ) -> None:
        """Send daily trading summary."""
        if not self.enabled:
            return

        win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0
        pnl_pct = (total_pnl / starting_equity * 100) if starting_equity > 0 else 0

        emoji = "" if total_pnl >= 0 else ""

        message = f"""
*Daily Trading Summary - {date}*

{emoji} *P&L: ${total_pnl:,.2f} ({pnl_pct:+.2f}%)*

 *Trades:* {total_trades}
 *Winners:* {winning_trades}
 *Win Rate:* {win_rate:.1f}%

 *Starting Equity:* ${starting_equity:,.2f}
 *Ending Equity:* ${ending_equity:,.2f}
"""

        color = "good" if total_pnl >= 0 else "danger"

        await self._send_slack(text=message, color=color)

    async def notify_risk_alert(self, alert_type: str, message: str) -> None:
        """Send a risk management alert."""
        if not self.enabled:
            return

        await self._send_slack(
            text=f"*Risk Alert: {alert_type}*\n{message}",
            color="warning",
        )

    async def notify_system_status(self, status: str, message: str) -> None:
        """Send system status notification."""
        if not self.enabled:
            return

        emoji = "" if status == "started" else "" if status == "stopped" else ""

        await self._send_slack(
            text=f"{emoji} *System {status.title()}*\n{message}",
            color="good" if status == "started" else "#808080",
        )

    def _format_trade_message(self, notification: TradeNotification) -> str:
        """Format trade notification message."""
        parts = [
            f"*{notification.action}* {notification.symbol}",
            f"Qty: {notification.qty}",
            f"Price: ${notification.price:.2f}",
        ]

        if notification.pnl is not None:
            pnl_str = f"${notification.pnl:+,.2f}"
            parts.append(f"P&L: {pnl_str}")

        if notification.strategy:
            parts.append(f"Strategy: {notification.strategy}")

        if notification.message:
            parts.append(notification.message)

        return "\n".join(parts)

    def _get_emoji(self, action: str) -> str:
        """Get emoji for action type."""
        emojis = {
            "BUY": "",
            "SELL": "",
            "STOP_LOSS": "",
            "TAKE_PROFIT": "",
            "TRAILING_STOP": "",
        }
        return emojis.get(action, "")

    def _get_color(self, action: str, pnl: float | None) -> str:
        """Get color for notification."""
        if pnl is not None:
            return "good" if pnl >= 0 else "danger"
        if action == "BUY":
            return "#2196F3"  # Blue
        return "#808080"  # Gray

    async def _send_slack(
        self,
        text: str,
        blocks: list[dict] | None = None,
        color: str = "good",
    ) -> None:
        """Send message to Slack."""
        if not self._session or not self.slack_webhook_url:
            return

        payload: dict[str, Any] = {
            "text": text,
        }

        if blocks:
            payload["attachments"] = [
                {
                    "color": color,
                    "blocks": blocks,
                }
            ]

        try:
            async with self._session.post(
                self.slack_webhook_url,
                json=payload,
                timeout=aiohttp.ClientTimeout(total=10),
            ) as response:
                if response.status != 200:
                    logger.warning(
                        "Failed to send Slack notification",
                        status=response.status,
                    )
        except Exception as e:
            logger.warning("Error sending Slack notification", error=str(e))
