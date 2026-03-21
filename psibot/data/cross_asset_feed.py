"""
data/cross_asset_feed.py — Cross-Asset Return Data (27-Asset Universe)
=======================================================================
Feeds the 27-asset return matrix used by L3 (D_eff computation).
D_eff is the leading systemic risk indicator — its decline precedes crashes.

From the article (Section 5):
  'D_eff = -log(Σ λi²) / log(N) ... D_eff decreasing over time →
   approaching saturation → crash risk rising'

CCDR Expectation Field Architecture — Version 1.0
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Optional
import numpy as np
import pandas as pd

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from helpers import ASSET_UNIVERSE_27, rolling_window_returns

log = logging.getLogger("psibot.data.crossasset")


@dataclass
class CrossAssetData:
    """
    Cross-asset return matrix and derived acoustic signals.
    Used by L3 (D_eff) and L5 (breadth, momentum).
    """
    timestamp: datetime
    assets: list[str]
    prices: pd.DataFrame                  # index=date, columns=assets
    returns_matrix: np.ndarray            # shape (window_days, N_assets)
    window_days: int = 60

    # L5 acoustic data (extracted from primary price series — SPX)
    spx_prices: Optional[pd.Series] = None
    momentum_20d: float = 0.0
    momentum_60d: float = 0.0
    momentum_252d: float = 0.0            # 12-month momentum; T3 regime thresholds apply
    vix_term_structure: float = 0.0       # VIX3M − VIX; T8: >0=contango=bullish regime
    breadth: float = 0.5                  # % assets above 50d MA
    volume_ratio: float = 1.0

    def compute_breadth(self) -> float:
        """Fraction of assets above their 50-day moving average."""
        if self.prices.empty or len(self.prices) < 50:
            return 0.5
        ma50 = self.prices.rolling(50).mean().iloc[-1]
        current = self.prices.iloc[-1]
        above = (current > ma50).sum()
        return float(above) / max(len(self.assets), 1)

    def compute_momentum(self, lookback: int = 20) -> float:
        """Price momentum for primary instrument (SPX)."""
        if self.spx_prices is None or len(self.spx_prices) < lookback + 1:
            return 0.0
        ret = (self.spx_prices.iloc[-1] / self.spx_prices.iloc[-(lookback + 1)]) - 1.0
        return float(ret)


class CrossAssetFeed:
    """
    Adapter for 27-asset universe price data.
    Supports: CSV flat files, Yahoo Finance (for testing), Bloomberg.
    """

    def __init__(self, provider: str = "csv", config: dict = None):
        self.provider = provider
        self.config = config or {}
        self._price_cache: Optional[pd.DataFrame] = None
        self._cache_time: Optional[datetime] = None

    async def get_returns_matrix(
        self,
        window_days: int = 60,
        assets: list[str] = None,
    ) -> CrossAssetData:
        """Fetch cross-asset return matrix for D_eff computation."""
        if assets is None:
            assets = ASSET_UNIVERSE_27

        try:
            prices = await self._fetch_prices(assets)
            if prices is None or prices.empty:
                log.error("CrossAssetFeed: no price data — returning synthetic")
                return self._make_synthetic_data(assets, window_days)

            returns_matrix = rolling_window_returns(prices, window=window_days)
            spx_col = "SPX" if "SPX" in prices.columns else prices.columns[0]
            spx_series = prices[spx_col] if spx_col in prices.columns else None

            data = CrossAssetData(
                timestamp=datetime.utcnow(),
                assets=list(prices.columns),
                prices=prices,
                returns_matrix=returns_matrix,
                window_days=window_days,
                spx_prices=spx_series,
            )
            data.breadth = data.compute_breadth()
            data.momentum_20d = data.compute_momentum(20)
            data.momentum_60d = data.compute_momentum(60)
            data.momentum_252d = data.compute_momentum(252)
            data.vix_term_structure = await self._fetch_vix_term_structure()
            return data

        except Exception as e:
            log.error("CrossAssetFeed.get_returns_matrix failed: %s", e)
            return self._make_synthetic_data(assets, window_days)

    async def _fetch_prices(self, assets: list[str]) -> Optional[pd.DataFrame]:
        """Fetch price data based on configured provider."""
        if self.provider == "csv":
            return self._load_from_csv(assets)
        elif self.provider == "yfinance":
            return await self._fetch_yfinance(assets)
        elif self.provider == "bloomberg":
            return await self._fetch_bloomberg(assets)
        else:
            return None

    def _load_from_csv(self, assets: list[str]) -> Optional[pd.DataFrame]:
        """Load prices from CSV file at config['data_dir']/prices.csv"""
        data_dir = self.config.get("data_dir", "data/prices")
        filepath = os.path.join(data_dir, "universe_prices.csv")
        if not os.path.exists(filepath):
            log.warning("Price CSV not found: %s", filepath)
            return None
        try:
            df = pd.read_csv(filepath, index_col=0, parse_dates=True)
            # Keep only available assets
            available = [a for a in assets if a in df.columns]
            if len(available) < 5:
                log.warning("Only %d of %d assets available in CSV", len(available), len(assets))
            return df[available] if available else None
        except Exception as e:
            log.error("CrossAssetFeed CSV load failed: %s", e)
            return None

    async def _fetch_yfinance(self, assets: list[str]) -> Optional[pd.DataFrame]:
        """Yahoo Finance (for development/testing only)."""
        try:
            import yfinance as yf
            # Map internal asset codes to Yahoo tickers
            ticker_map = {
                "SPX": "^GSPC", "NDX": "^NDX", "RUT": "^RUT",
                "SXXP": "^STOXX50E", "NKY": "^N225",
                "VIX": "^VIX", "GOLD": "GC=F", "CRUDE": "CL=F",
                "DXY": "DX-Y.NYB", "US10Y": "^TNX",
            }
            tickers = [ticker_map.get(a, a) for a in assets]
            available = [a for a in assets if a in ticker_map]
            yf_tickers = [ticker_map[a] for a in available]

            end = datetime.utcnow()
            start = end - timedelta(days=300)  # 300d to cover 252d momentum + buffer
            data = yf.download(yf_tickers, start=start, end=end, progress=False)["Close"]
            data.columns = available
            return data.dropna(how="all")
        except Exception as e:
            log.error("yfinance fetch failed: %s", e)
            return None

    async def _fetch_bloomberg(self, assets: list[str]) -> Optional[pd.DataFrame]:
        """Bloomberg data (placeholder)."""
        log.warning("Bloomberg integration not yet implemented")
        return None

    async def _fetch_vix_term_structure(self) -> float:
        """
        Fetch current VIX term structure: VIX3M − VIX.
        T8 (70.6% accuracy): contango (>0) = bullish 3-month regime; backwardation (<0) = bearish.
        Returns 0.0 on any failure (neutral, no regime penalty applied).
        """
        try:
            import yfinance as yf
            from datetime import timedelta
            end = datetime.utcnow()
            start = end - timedelta(days=10)
            raw = yf.download(["^VIX", "^VIX3M"], start=start, end=end,
                               progress=False, auto_adjust=True)["Close"]
            if isinstance(raw.columns, pd.MultiIndex):
                raw.columns = [c[0] if isinstance(c, tuple) else c for c in raw.columns]
            vix_latest  = float(raw["^VIX"].dropna().iloc[-1])
            vix3m_latest = float(raw["^VIX3M"].dropna().iloc[-1])
            vts = vix3m_latest - vix_latest
            log.debug("VTS: VIX3M=%.2f VIX=%.2f → VTS=%.2f (%s)",
                      vix3m_latest, vix_latest, vts,
                      "contango" if vts > 0 else "backwardation")
            return vts
        except Exception as e:
            log.warning("VIX term structure fetch failed (%s) — defaulting to 0.0", e)
            return 0.0

    def _make_synthetic_data(self, assets: list[str], window_days: int) -> CrossAssetData:
        """
        Synthetic cross-asset data for testing.
        Generates a plausible D_eff ~ 12-16 (normal markets).
        """
        N = len(assets)
        rng = np.random.default_rng(42)

        # Generate mildly correlated returns (D_eff ~12)
        # 3 common factors explaining ~30% of variance
        n_factors = 3
        factor_loads = rng.standard_normal((N, n_factors)) * 0.3
        factors = rng.standard_normal((window_days + 1, n_factors))
        idiosyncratic = rng.standard_normal((window_days + 1, N)) * 0.01
        prices_raw = np.exp(np.cumsum(factor_loads @ factors.T + idiosyncratic.T, axis=1))
        prices = pd.DataFrame(prices_raw.T, columns=assets)
        returns_matrix = rolling_window_returns(prices, window=window_days)

        # Create a synthetic SPX series
        spx_prices = pd.Series(prices.iloc[:, 0].values)

        data = CrossAssetData(
            timestamp=datetime.utcnow(),
            assets=assets,
            prices=prices,
            returns_matrix=returns_matrix,
            window_days=window_days,
            spx_prices=spx_prices,
            momentum_20d=0.01,
            momentum_60d=0.03,
            momentum_252d=0.08,  # synthetic normal regime (>+5% T3 threshold)
            vix_term_structure=2.0,  # synthetic contango (typical normal market)
            breadth=0.65,
        )
        return data
