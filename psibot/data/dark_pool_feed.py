"""
data/dark_pool_feed.py — Dark Pool (ATS) Data Ingestion
========================================================
Dark pool fraction is the 4th component of GBP (weight 0.10).

From the article (Section 8.1):
  'Dark pool fraction: expectations refusing acoustic coupling.
   High dark pool fraction = large optical phonon sector building
   pressure below the gap threshold.'

CCDR Expectation Field Architecture — Version 1.0
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from datetime import datetime
from typing import Optional
import numpy as np

log = logging.getLogger("psibot.data.darkpool")


@dataclass
class DarkPoolData:
    """
    FINRA ATS (dark pool) trading volume data.
    dark_pool_fraction = dark pool volume / total volume (0.0 to 1.0)
    """
    timestamp: datetime
    symbol: str
    dark_pool_volume: float             # shares traded in dark pools
    total_volume: float                 # total shares traded
    dark_pool_fraction: float           # dark_pool_volume / total_volume
    rolling_90d_avg_fraction: float     # 90-day moving average of fraction
    dark_pool_ratio: float              # current / 90d avg — used in GBP

    @classmethod
    def from_volumes(
        cls,
        symbol: str,
        dark_vol: float,
        total_vol: float,
        rolling_90d_avg: float,
    ) -> "DarkPoolData":
        fraction = dark_vol / max(total_vol, 1.0)
        ratio = fraction / max(rolling_90d_avg, 0.001)
        return cls(
            timestamp=datetime.utcnow(),
            symbol=symbol,
            dark_pool_volume=dark_vol,
            total_volume=total_vol,
            dark_pool_fraction=fraction,
            rolling_90d_avg_fraction=rolling_90d_avg,
            dark_pool_ratio=ratio,
        )


class DarkPoolFeed:
    """
    Adapter for dark pool / ATS data.
    Primary source: FINRA ATS transparency data (daily, lagged 1 day).
    """

    def __init__(self, provider: str = "csv", config: dict = None):
        self.provider = provider
        self.config = config or {}
        self._fraction_history: list[float] = []

    async def get_dark_pool_data(self, symbol: str) -> DarkPoolData:
        """Fetch latest dark pool fraction for symbol."""
        try:
            if self.provider == "csv":
                return self._load_from_csv(symbol)
            elif self.provider == "finra_ats":
                return await self._fetch_finra_ats(symbol)
            else:
                return self._make_neutral_data(symbol)
        except Exception as e:
            log.error("DarkPoolFeed.get_dark_pool_data(%s) failed: %s", symbol, e)
            return self._make_neutral_data(symbol)

    def _load_from_csv(self, symbol: str) -> DarkPoolData:
        """Load dark pool data from FINRA ATS CSV file."""
        import pandas as pd
        data_dir = self.config.get("data_dir", "data/dark_pool")
        filepath = os.path.join(data_dir, f"{symbol}_ats.csv")
        if not os.path.exists(filepath):
            log.warning("Dark pool CSV not found: %s — using neutral", filepath)
            return self._make_neutral_data(symbol)

        df = pd.read_csv(filepath, parse_dates=["date"])
        df = df.sort_values("date")

        # Latest entry
        latest = df.iloc[-1]
        dark_vol = float(latest["dark_pool_volume"])
        total_vol = float(latest["total_volume"])

        # 90-day rolling average of fraction
        fractions = (df["dark_pool_volume"] / df["total_volume"]).values
        rolling_avg = float(np.mean(fractions[-90:])) if len(fractions) >= 10 else 0.35

        return DarkPoolData.from_volumes(symbol, dark_vol, total_vol, rolling_avg)

    async def _fetch_finra_ats(self, symbol: str) -> DarkPoolData:
        """FINRA ATS transparency API (placeholder)."""
        log.warning("FINRA ATS integration not yet implemented — using neutral")
        return self._make_neutral_data(symbol)

    def _make_neutral_data(self, symbol: str) -> DarkPoolData:
        """Neutral dark pool data: fraction at 90-day average → ratio=1.0."""
        return DarkPoolData(
            timestamp=datetime.utcnow(),
            symbol=symbol,
            dark_pool_volume=350_000_000.0,
            total_volume=1_000_000_000.0,
            dark_pool_fraction=0.35,
            rolling_90d_avg_fraction=0.35,
            dark_pool_ratio=1.0,  # neutral: current = 90d average
        )
