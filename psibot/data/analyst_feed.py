"""
data/analyst_feed.py — Analyst and Survey Data Ingestion
=========================================================
Sources for L2 Phase Agent:
  - IBES forward EPS dispersion → disorder parameter DP
  - AAII / Investors Intelligence surveys → order parameter OP

From the article (Section 8.1):
  'Analyst forecast dispersion: measures the disorder parameter of the
   expectation condensate. High dispersion = approaching grain boundary.'
  'Survey data: measures the order parameter direction.'

CCDR Expectation Field Architecture — Version 1.0
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, date
from typing import Optional
import numpy as np

log = logging.getLogger("psibot.data.analyst")


@dataclass
class AnalystData:
    """
    Analyst forecast data for computing the disorder parameter DP.
    DP = σ(EPS forecasts) / |mean(EPS forecast)|
    """
    timestamp: datetime
    symbol: str                              # e.g. "SPX_INDEX" or "SPX_CONSTITUENTS"
    forward_12m_eps_by_analyst: np.ndarray  # array of individual analyst estimates
    num_analysts: int = 0
    eps_mean: float = 0.0
    eps_std: float = 0.0
    age_business_days: int = 0              # staleness of data

    def __post_init__(self):
        if len(self.forward_12m_eps_by_analyst) > 0:
            self.num_analysts = len(self.forward_12m_eps_by_analyst)
            self.eps_mean = float(np.mean(self.forward_12m_eps_by_analyst))
            self.eps_std = float(np.std(self.forward_12m_eps_by_analyst))

    @property
    def is_stale(self) -> bool:
        return self.age_business_days > 5

    def disorder_parameter(self) -> float:
        """DP = σ(EPS) / |mean(EPS)|"""
        if abs(self.eps_mean) < 1e-6 or self.num_analysts < 2:
            return float("nan")
        return self.eps_std / abs(self.eps_mean)


@dataclass
class SurveyData:
    """
    Sentiment survey data for computing the order parameter OP.
    OP = (% bullish - % bearish) / 100
    Sources: AAII (weekly), Investors Intelligence (weekly), institutional (periodic).
    """
    timestamp: datetime

    # AAII (American Association of Individual Investors)
    aaii_bull: float = 0.0     # % bullish
    aaii_bear: float = 0.0     # % bearish
    aaii_neutral: float = 0.0  # % neutral
    aaii_date: Optional[date] = None

    # Investors Intelligence
    ii_bull: float = 0.0
    ii_bear: float = 0.0
    ii_correction: float = 0.0
    ii_date: Optional[date] = None

    # Institutional (NAAIM, BofA Fund Manager Survey, etc.)
    inst_bull: float = 0.0
    inst_bear: float = 0.0
    inst_date: Optional[date] = None

    def order_parameter(self) -> float:
        """
        OP ∈ [-1, +1]: weighted order parameter from all survey sources.
        WEIGHTS: AAII=0.3, II=0.4, Institutional=0.3
        """
        weights = {'aaii': 0.3, 'ii': 0.4, 'inst': 0.3}
        ops = {}

        if self.aaii_bull + self.aaii_bear > 0:
            ops['aaii'] = (self.aaii_bull - self.aaii_bear) / 100.0
        if self.ii_bull + self.ii_bear > 0:
            ops['ii'] = (self.ii_bull - self.ii_bear) / 100.0
        if self.inst_bull + self.inst_bear > 0:
            ops['inst'] = (self.inst_bull - self.inst_bear) / 100.0

        if not ops:
            return 0.0

        total_weight = sum(weights[k] for k in ops)
        op = sum(weights[k] * ops[k] for k in ops) / total_weight
        return float(np.clip(op, -1.0, 1.0))


class AnalystFeed:
    """
    Adapter for analyst and survey data sources.
    Supports: IBES (via Refinitiv/LSEG), manual CSV import.
    """

    def __init__(self, provider: str = "csv", config: dict = None):
        self.provider = provider
        self.config = config or {}
        self._analyst_cache: Optional[AnalystData] = None
        self._survey_cache: Optional[SurveyData] = None

    async def get_analyst_data(self, symbol: str) -> Optional[AnalystData]:
        """Fetch latest analyst EPS dispersion data."""
        try:
            if self.provider == "csv":
                return self._load_analyst_from_csv(symbol)
            elif self.provider == "ibes":
                return await self._fetch_ibes(symbol)
            else:
                log.warning("Unknown analyst provider: %s", self.provider)
                return self._make_neutral_analyst_data(symbol)
        except Exception as e:
            log.error("AnalystFeed.get_analyst_data(%s) failed: %s", symbol, e)
            return self._make_neutral_analyst_data(symbol)

    async def get_survey_data(self) -> Optional[SurveyData]:
        """Fetch latest sentiment survey data."""
        try:
            if self.provider == "csv":
                return self._load_survey_from_csv()
            else:
                return self._make_neutral_survey_data()
        except Exception as e:
            log.error("AnalystFeed.get_survey_data() failed: %s", e)
            return self._make_neutral_survey_data()

    def _load_analyst_from_csv(self, symbol: str) -> Optional[AnalystData]:
        """Load analyst EPS estimates from CSV."""
        import pandas as pd
        data_dir = self.config.get("data_dir", "data/analyst")
        filepath = os.path.join(data_dir, f"{symbol}_eps_estimates.csv")
        if not os.path.exists(filepath):
            log.warning("Analyst CSV not found: %s — using neutral", filepath)
            return self._make_neutral_analyst_data(symbol)

        df = pd.read_csv(filepath)
        estimates = df["forward_12m_eps"].dropna().values
        return AnalystData(
            timestamp=datetime.utcnow(),
            symbol=symbol,
            forward_12m_eps_by_analyst=estimates,
        )

    def _load_survey_from_csv(self) -> Optional[SurveyData]:
        """Load survey data from CSV."""
        import pandas as pd
        data_dir = self.config.get("data_dir", "data/surveys")
        aaii_path = os.path.join(data_dir, "aaii_latest.csv")
        if not os.path.exists(aaii_path):
            return self._make_neutral_survey_data()
        df = pd.read_csv(aaii_path)
        latest = df.iloc[-1]
        return SurveyData(
            timestamp=datetime.utcnow(),
            aaii_bull=float(latest.get("bull", 33.0)),
            aaii_bear=float(latest.get("bear", 33.0)),
            aaii_neutral=float(latest.get("neutral", 34.0)),
        )

    async def _fetch_ibes(self, symbol: str) -> Optional[AnalystData]:
        """IBES/Refinitiv integration (placeholder)."""
        log.warning("IBES integration not yet implemented — using neutral data")
        return self._make_neutral_analyst_data(symbol)

    def _make_neutral_analyst_data(self, symbol: str) -> AnalystData:
        """Conservative neutral analyst data when source unavailable."""
        return AnalystData(
            timestamp=datetime.utcnow(),
            symbol=symbol,
            forward_12m_eps_by_analyst=np.array([10.0, 10.5, 9.8, 10.2, 10.3, 9.9]),
            age_business_days=0,
        )

    def _make_neutral_survey_data(self) -> SurveyData:
        """Neutral survey data (equal bull/bear = OP=0.0)."""
        return SurveyData(
            timestamp=datetime.utcnow(),
            aaii_bull=33.0,
            aaii_bear=33.0,
            aaii_neutral=34.0,
            ii_bull=40.0,
            ii_bear=40.0,
            ii_correction=20.0,
            inst_bull=50.0,
            inst_bear=50.0,
        )


import os
