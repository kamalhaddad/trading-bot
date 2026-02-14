#!/usr/bin/env python3
"""Run the trading bot in paper trading mode."""

import asyncio
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.main import main


if __name__ == "__main__":
    print("Starting trading bot in paper trading mode...")
    print("Press Ctrl+C to stop")
    print()

    asyncio.run(main("config/settings.yaml"))
