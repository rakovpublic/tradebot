"""
data/dark_pool_feed.py — Dark Pool / Institutional Flow Data Ingestion
========================================================================
Dark pool signal is the 4th component of GBP (weight 0.05).

T5 hypothesis result: FINRA ATS volume has no directional encoding (buys + sells net to
zero). Variance Risk Premium (VIX²−RV²) is the empirically validated institutional flow
proxy (Bollerslev, Tauchen, Zhou 2009): when VRP_var is above its rolling median, implied
variance exceeds realized variance → fear excess → contrarian bullish.

Provider options:
  'vrp'      — Variance Risk Premium from yfinance (^VIX + ^GSPC) [recommended]
  'csv'      — FINRA ATS CSV files (dark_pool_volume / total_volume ratio)
  'finra_ats'— FINRA ATS API (not yet implemented)

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
        """Fetch latest dark pool / VRP signal for symbol."""
        try:
            if self.provider == "vrp":
                return await self._fetch_vrp(symbol)
            elif self.provider == "csv":
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

    async def _fetch_vrp(self, symbol: str) -> DarkPoolData:
        """
        Variance Risk Premium (VIX²−RV²) as institutional dark-flow proxy.
        T5 validated: variance premium > rolling 252d median → contrarian bullish signal.
        dark_pool_ratio = vrp_var / vrp_252d_median; ratio>1 → above-average fear → bullish.
        """
        try:
            import pandas as pd
            import yfinance as yf
            from datetime import timedelta
            end = datetime.utcnow()
            start = end - timedelta(days=400)  # 400d covers 252d rolling median + buffer
            raw = yf.download(["^VIX", "^GSPC"], start=start, end=end,
                               progress=False, auto_adjust=True)["Close"]
            if isinstance(raw.columns, pd.MultiIndex):
                raw.columns = [c[0] if isinstance(c, tuple) else c for c in raw.columns]
            vix  = raw["^VIX"].dropna()
            spx  = raw["^GSPC"].dropna()
            # Realized variance: 21-day annualized
            rv_dec   = spx.pct_change().rolling(21).std() * np.sqrt(252)
            vrp_var  = (vix / 100)**2 - rv_dec**2
            vrp_var  = vrp_var.dropna()
            if len(vrp_var) < 50:
                log.warning("VRP: insufficient data (%d obs) — neutral", len(vrp_var))
                return self._make_neutral_data(symbol)
            vrp_median = float(vrp_var.rolling(252, min_periods=50).median().iloc[-1])
            vrp_current = float(vrp_var.iloc[-1])
            ratio = vrp_current / vrp_median if vrp_median > 0 else 1.0
            log.debug("VRP dark-flow: vrp_var=%.5f median=%.5f ratio=%.3f (%s)",
                      vrp_current, vrp_median, ratio,
                      "bullish (fear excess)" if ratio > 1 else "neutral/bearish")
            return DarkPoolData(
                timestamp=datetime.utcnow(),
                symbol=symbol,
                dark_pool_volume=vrp_current,       # repurposed field: VRP_var value
                total_volume=vrp_median,            # repurposed field: rolling median
                dark_pool_fraction=min(max(vrp_current, 0), 1),
                rolling_90d_avg_fraction=vrp_median,
                dark_pool_ratio=ratio,
            )
        except Exception as e:
            log.warning("VRP fetch failed (%s) — using neutral", e)
            return self._make_neutral_data(symbol)

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
