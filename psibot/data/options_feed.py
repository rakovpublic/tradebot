"""
data/options_feed.py — Options Surface Ingestion & Normalisation (SK-03)
========================================================================
Ingest, validate, and normalise implied volatility surfaces.
The options surface is the PRIMARY DATA SOURCE — the direct measurement
of the expectation field wavefunction ψ_exp.

CCDR Expectation Field Architecture — Version 1.0
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional
import numpy as np
from scipy.interpolate import CubicSpline

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from helpers import validate_options_surface

log = logging.getLogger("psibot.data.options")

# Standard tenors in trading days
STANDARD_TENORS_DAYS = [7, 14, 30, 91, 182, 365, 730]


@dataclass
class OptionsSurface:
    """
    Options implied volatility surface for a single underlying.
    This object IS the quantum wavefunction measurement apparatus.

    From the article (Section 7):
      'The vol surface is the measurement apparatus for ψ_exp.'
      'Changes in the vol surface = changes in the expectation condensate phase.'
    """
    underlying: str
    timestamp: datetime
    spot: float
    tenors_days: list[int]
    strikes: dict[int, np.ndarray]      # tenor_days → strike prices
    iv: dict[int, np.ndarray]           # tenor_days → implied vols
    bid_iv: dict[int, np.ndarray]       # for spread quality check
    ask_iv: dict[int, np.ndarray]
    open_interest: dict[int, np.ndarray]

    # Validation metadata
    is_valid: bool = False
    validation_issues: list[str] = field(default_factory=list)

    def validate(self) -> bool:
        """
        Validate surface quality: arbitrage-free checks, spread checks,
        sufficient strike coverage.
        """
        all_issues = []
        for tenor in self.tenors_days:
            if tenor not in self.strikes or tenor not in self.iv:
                all_issues.append(f"Tenor {tenor}d: missing data")
                continue
            ok, issues = validate_options_surface(
                self.strikes[tenor], self.iv[tenor], self.spot, tenor
            )
            all_issues.extend([f"Tenor {tenor}d: {i}" for i in issues])

        # Check bid-ask spread quality
        for tenor in self.tenors_days:
            if tenor not in self.bid_iv or tenor not in self.ask_iv:
                continue
            spreads = self.ask_iv[tenor] - self.bid_iv[tenor]
            if np.any(spreads < 0):
                all_issues.append(f"Tenor {tenor}d: negative bid-ask spread")
            if np.median(spreads) > 0.05:
                all_issues.append(f"Tenor {tenor}d: wide spreads (median={np.median(spreads):.3f})")

        self.validation_issues = all_issues
        self.is_valid = len(all_issues) == 0
        if all_issues:
            log.warning("Options surface validation: %d issues for %s",
                        len(all_issues), self.underlying)
        return self.is_valid

    def skew_at_tenor(self, tenor_days: int) -> float:
        """
        IV(0.9S, T) - IV(1.1S, T) — condensate chirality proxy.
        Positive → bearish (put wing elevated), Negative → bullish.
        Corresponds to: psi_skew in CondensateState.
        """
        if tenor_days not in self.strikes:
            return 0.0
        strikes = self.strikes[tenor_days]
        ivs = self.iv[tenor_days]
        if len(strikes) < 3:
            return 0.0
        try:
            cs = CubicSpline(strikes, ivs, extrapolate=True)
            iv_90 = float(cs(0.9 * self.spot))
            iv_110 = float(cs(1.1 * self.spot))
            return iv_90 - iv_110
        except Exception as e:
            log.warning("skew_at_tenor(%d) error: %s", tenor_days, e)
            return 0.0

    def kurtosis_excess_at_tenor(self, tenor_days: int) -> float:
        """
        IV(0.8S,T) + IV(1.2S,T) - 2*IV(S,T) — grain boundary proximity proxy.
        Higher → fatter tails → nearer grain boundary.
        """
        if tenor_days not in self.strikes:
            return 0.0
        strikes = self.strikes[tenor_days]
        ivs = self.iv[tenor_days]
        try:
            cs = CubicSpline(strikes, ivs, extrapolate=True)
            return float(cs(0.8 * self.spot) + cs(1.2 * self.spot) - 2 * cs(self.spot))
        except Exception as e:
            log.warning("kurtosis_excess_at_tenor(%d) error: %s", tenor_days, e)
            return 0.0

    def atm_iv_by_tenor(self) -> dict[int, float]:
        """ATM implied vol at each tenor — for term structure classification."""
        result = {}
        for tenor in self.tenors_days:
            if tenor not in self.strikes:
                continue
            strikes = self.strikes[tenor]
            ivs = self.iv[tenor]
            try:
                cs = CubicSpline(strikes, ivs, extrapolate=False)
                result[tenor] = float(cs(self.spot))
            except Exception:
                atm_idx = np.argmin(np.abs(strikes - self.spot))
                result[tenor] = float(ivs[atm_idx])
        return result

    def interpolate_strike(self, tenor_days: int, strike: float) -> Optional[float]:
        """Interpolate IV at arbitrary strike for a given tenor."""
        if tenor_days not in self.strikes:
            return None
        try:
            cs = CubicSpline(self.strikes[tenor_days], self.iv[tenor_days], extrapolate=True)
            val = float(cs(strike))
            return max(val, 0.001)
        except Exception:
            return None


class OptionsFeed:
    """
    Data source adapter for options surfaces.
    Supports: LiveVol API, Bloomberg OVDV, IBKR TWS, CSV (backtesting).
    """

    def __init__(self, provider: str = "csv", config: dict = None):
        self.provider = provider
        self.config = config or {}
        self._cache: dict[str, OptionsSurface] = {}
        self._last_fetch: dict[str, datetime] = {}

    async def get_surface(self, symbol: str) -> Optional[OptionsSurface]:
        """Fetch and return validated options surface for symbol."""
        try:
            if self.provider == "csv":
                return self._load_from_csv(symbol)
            elif self.provider == "livevol":
                return await self._fetch_livevol(symbol)
            elif self.provider == "ibkr":
                return await self._fetch_ibkr(symbol)
            elif self.provider == "bloomberg":
                return await self._fetch_bloomberg(symbol)
            else:
                log.error("Unknown options provider: %s", self.provider)
                return None
        except Exception as e:
            log.error("OptionsFeed.get_surface(%s) failed: %s", symbol, e)
            return None

    def _load_from_csv(self, symbol: str) -> Optional[OptionsSurface]:
        """
        Load surface from CSV file at config['data_dir']/<symbol>_surface.csv
        CSV format: tenor_days, strike, iv, bid_iv, ask_iv, open_interest
        """
        import pandas as pd
        data_dir = self.config.get("data_dir", "data/surfaces")
        filepath = os.path.join(data_dir, f"{symbol}_surface.csv")

        if not os.path.exists(filepath):
            log.warning("Surface CSV not found: %s", filepath)
            return None

        df = pd.read_csv(filepath)
        df = df.sort_values(["tenor_days", "strike"])

        tenors = sorted(df["tenor_days"].unique().tolist())
        strikes = {t: df[df["tenor_days"] == t]["strike"].values for t in tenors}
        iv = {t: df[df["tenor_days"] == t]["iv"].values for t in tenors}
        bid_iv = {t: df[df["tenor_days"] == t].get("bid_iv", df[df["tenor_days"] == t]["iv"] * 0.99).values for t in tenors}
        ask_iv = {t: df[df["tenor_days"] == t].get("ask_iv", df[df["tenor_days"] == t]["iv"] * 1.01).values for t in tenors}
        oi = {t: df[df["tenor_days"] == t].get("open_interest", np.ones(len(strikes[t]))).values for t in tenors}

        spot = self.config.get("spot", {}).get(symbol, df["strike"].median())

        surface = OptionsSurface(
            underlying=symbol,
            timestamp=datetime.utcnow(),
            spot=spot,
            tenors_days=tenors,
            strikes=strikes,
            iv=iv,
            bid_iv=bid_iv,
            ask_iv=ask_iv,
            open_interest=oi,
        )
        surface.validate()
        return surface

    async def _fetch_livevol(self, symbol: str) -> Optional[OptionsSurface]:
        """LiveVol API integration (placeholder — implement with API credentials)."""
        log.warning("LiveVol integration not yet implemented for %s", symbol)
        return None

    async def _fetch_ibkr(self, symbol: str) -> Optional[OptionsSurface]:
        """IBKR TWS API integration (placeholder)."""
        log.warning("IBKR integration not yet implemented for %s", symbol)
        return None

    async def _fetch_bloomberg(self, symbol: str) -> Optional[OptionsSurface]:
        """Bloomberg OVDV integration (placeholder)."""
        log.warning("Bloomberg integration not yet implemented for %s", symbol)
        return None

    def build_synthetic_surface(
        self,
        underlying: str,
        spot: float,
        atm_vol: float = 0.20,
        skew: float = -0.02,
        kurtosis: float = 0.01,
        tenors_days: list[int] = None,
    ) -> OptionsSurface:
        """
        Build a synthetic surface for testing/backtesting.
        Models a mildly skewed vol surface with controllable shape parameters.
        """
        if tenors_days is None:
            tenors_days = STANDARD_TENORS_DAYS

        strikes_dict = {}
        iv_dict = {}
        bid_dict = {}
        ask_dict = {}
        oi_dict = {}

        for tenor in tenors_days:
            t_years = tenor / 252.0
            moneyness_range = np.linspace(0.7, 1.3, 25)
            K = moneyness_range * spot

            # Simplified quadratic vol surface: IV(m) = atm + skew*(m-1) + kurtosis*(m-1)^2
            m = moneyness_range - 1.0
            iv_slice = atm_vol * (1 + skew * m / atm_vol + kurtosis * m**2 / atm_vol)
            iv_slice = np.clip(iv_slice, 0.01, 2.0)
            spread = 0.005 + 0.01 * np.abs(m)

            strikes_dict[tenor] = K
            iv_dict[tenor] = iv_slice
            bid_dict[tenor] = np.clip(iv_slice - spread / 2, 0.005, None)
            ask_dict[tenor] = iv_slice + spread / 2
            oi_dict[tenor] = np.ones(len(K)) * 1000.0

        surface = OptionsSurface(
            underlying=underlying,
            timestamp=datetime.utcnow(),
            spot=spot,
            tenors_days=tenors_days,
            strikes=strikes_dict,
            iv=iv_dict,
            bid_iv=bid_dict,
            ask_iv=ask_dict,
            open_interest=oi_dict,
        )
        surface.validate()
        return surface
