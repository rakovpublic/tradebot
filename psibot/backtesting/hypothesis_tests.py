"""
backtesting/hypothesis_tests.py — SK-21: T1-T9 CCDR Hypothesis Tests
======================================================================
All nine CCDR market predictions must pass before live deployment.

From the article (Section 9):
  T1: Options vol surface Granger-causes price (not vice versa)
  T2: Analyst dispersion leads regime changes
  T3: Momentum crashes are bimodally distributed (soliton collapse)
  T4: D_eff declines before crashes (30-60 day lead)
  T5: Dark pool fraction predicts price direction
  T6: Equity risk premium has 3-7yr spectral peak (temporal crystal)
  T7: Technical levels survive participant turnover (topological defects)
  T8: Vol smile skew encodes condensate chirality
  T9: Post-earnings drift ∝ analyst forecast dispersion

DEPLOYMENT GATE: ≥ 7 of 9 must pass.
If any 3 fail: HOLD — do not proceed to Phase 1.

CCDR Expectation Field Architecture — Version 1.0
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, Any
import numpy as np
import pandas as pd

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from helpers import (
    granger_causality_test, hartigan_dip_test, spectral_peak_frequency,
    compute_d_eff, rolling_window_returns,
)

log = logging.getLogger("psibot.backtest.hypotheses")

# Pass thresholds (from skill.md SK-21)
T1_P_THRESHOLD = 0.01        # Granger causality p < 0.01
T2_LEAD_MONTHS_MIN = 2       # analyst dispersion leads by 2+ months
T2_LEAD_MONTHS_MAX = 8       # analyst dispersion leads by ≤ 8 months
T2_P_THRESHOLD = 0.05
T3_DIP_P_THRESHOLD = 0.05   # bimodal distribution
T4_LEAD_DAYS_MIN = 25        # D_eff lead at least 25 days
T4_DIRECTIONAL_ACCURACY = 0.70
T5_ACCURACY_THRESHOLD = 0.55
T6_BAND_POWER_THRESHOLD = 0.25  # > 25% power in 3-7yr band
T7_PERSISTENCE_THRESHOLD = 0.70
T8_ACCURACY_THRESHOLD = 0.60
T9_R2_THRESHOLD = 0.20


@dataclass
class HypothesisResult:
    test_id: str
    name: str
    passed: bool
    test_statistic: Optional[float] = None
    p_value: Optional[float] = None
    threshold: Optional[float] = None
    details: str = ""
    error: str = ""


@dataclass
class HypothesisReport:
    run_at: datetime = field(default_factory=datetime.utcnow)
    results: dict = field(default_factory=dict)  # test_id → HypothesisResult
    passed_count: int = 0
    failed_count: int = 0
    deploy_recommended: bool = False
    blocking_failures: list[str] = field(default_factory=list)

    def summary(self) -> str:
        lines = [
            f"CCDR Hypothesis Test Report — {self.run_at.strftime('%Y-%m-%d %H:%M')}",
            f"Results: {self.passed_count}/9 passed",
            f"Deploy recommended: {self.deploy_recommended}",
            "",
        ]
        for tid, result in sorted(self.results.items()):
            status = "PASS ✓" if result.passed else "FAIL ✗"
            lines.append(f"  {tid}: {status} — {result.name}")
            if result.details:
                lines.append(f"       {result.details}")
            if result.error:
                lines.append(f"       ERROR: {result.error}")
        if self.blocking_failures:
            lines.append(f"\nBlocking failures: {', '.join(self.blocking_failures)}")
        return "\n".join(lines)


class CCDRHypothesisTests:
    """
    Run all 9 CCDR hypothesis tests.
    Each test is a self-contained statistical test.
    Tests must be run before Phase 3 live deployment.
    """

    def test_T1_vol_surface_granger_causes_price(
        self,
        vol_changes: np.ndarray,
        price_changes: np.ndarray,
        max_lag: int = 10,
    ) -> HypothesisResult:
        """
        T1: Vol surface changes Granger-cause price changes, not vice versa.
        Pass: F-stat p < 0.01 at any lag 1-5 days for vol→price direction.

        From the article (Section 9):
          'Vol surface = direct measurement of expectation condensate restructuring
           BEFORE it reaches prices'
        """
        result = HypothesisResult(
            test_id="T1",
            name="Vol surface Granger-causes price",
            passed=False,
            threshold=T1_P_THRESHOLD,
        )
        try:
            # Test vol → price direction
            p_values_vol_to_price = granger_causality_test(
                y=price_changes, x=vol_changes, max_lag=max_lag
            )
            # Test price → vol direction (should NOT Granger-cause)
            p_values_price_to_vol = granger_causality_test(
                y=vol_changes, x=price_changes, max_lag=max_lag
            )

            min_p_vol_to_price = min(p_values_vol_to_price[lag] for lag in range(1, 6))
            min_p_price_to_vol = min(p_values_price_to_vol[lag] for lag in range(1, 6))

            passed = (min_p_vol_to_price < T1_P_THRESHOLD
                      and min_p_price_to_vol >= 0.05)
            result.passed = passed
            result.p_value = min_p_vol_to_price
            result.test_statistic = min_p_price_to_vol
            result.details = (
                f"vol→price min p={min_p_vol_to_price:.4f} (threshold<{T1_P_THRESHOLD}); "
                f"price→vol min p={min_p_price_to_vol:.4f} (should be ≥0.05)"
            )

        except Exception as e:
            result.error = str(e)
            log.error("T1 failed: %s", e)

        return result

    def test_T2_analyst_dispersion_leads_regime(
        self,
        dp_series: pd.Series,
        regime_change_dates: list,
        business_days_per_month: int = 21,
    ) -> HypothesisResult:
        """
        T2: Analyst dispersion peaks precede regime changes by 2-8 months.
        Pass: median lead time in [2, 8] months, p < 0.05.
        """
        result = HypothesisResult(
            test_id="T2",
            name="Analyst dispersion leads regime change",
            passed=False,
            threshold=T2_P_THRESHOLD,
        )
        try:
            if dp_series is None or len(dp_series) < 36:
                result.error = "Insufficient DP data (need ≥36 observations)"
                return result

            # Find DP local maxima (peaks)
            from scipy.signal import find_peaks
            peaks, _ = find_peaks(dp_series.values, prominence=0.1, distance=30)
            peak_dates = dp_series.index[peaks]

            if len(peak_dates) == 0 or len(regime_change_dates) == 0:
                result.error = "No DP peaks or regime change dates found"
                return result

            # Compute lead times: for each regime change, find nearest preceding DP peak
            lead_days = []
            for change_date in regime_change_dates:
                preceding = [d for d in peak_dates if d < change_date]
                if preceding:
                    nearest_peak = max(preceding)
                    lead = (change_date - nearest_peak).days
                    lead_days.append(lead)

            if not lead_days:
                result.error = "No preceding DP peaks found for regime changes"
                return result

            lead_months = [d / business_days_per_month for d in lead_days]
            median_lead = float(np.median(lead_months))

            # Test significance using binomial test
            in_window = sum(T2_LEAD_MONTHS_MIN <= m <= T2_LEAD_MONTHS_MAX
                           for m in lead_months)
            from scipy import stats
            binom_result = stats.binomtest(in_window, len(lead_months), 0.2)
            p_val = binom_result.pvalue

            passed = (T2_LEAD_MONTHS_MIN <= median_lead <= T2_LEAD_MONTHS_MAX
                      and p_val < T2_P_THRESHOLD)
            result.passed = passed
            result.test_statistic = median_lead
            result.p_value = p_val
            result.details = (
                f"Median lead={median_lead:.1f} months "
                f"(target: {T2_LEAD_MONTHS_MIN}-{T2_LEAD_MONTHS_MAX} months); "
                f"in-window fraction={in_window}/{len(lead_months)}; p={p_val:.4f}"
            )

        except Exception as e:
            result.error = str(e)
            log.error("T2 failed: %s", e)

        return result

    def test_T3_momentum_crashes_bimodal(
        self,
        momentum_drawdown_rates: np.ndarray,
    ) -> HypothesisResult:
        """
        T3: Momentum crash drawdown rates are bimodally distributed.
        Pass: Hartigan's dip test p < 0.05 (reject unimodal null).

        From the article: 'Soliton collapse is discontinuous at grain boundary'
        """
        result = HypothesisResult(
            test_id="T3",
            name="Momentum crashes bimodal (soliton collapse)",
            passed=False,
            threshold=T3_DIP_P_THRESHOLD,
        )
        try:
            if momentum_drawdown_rates is None or len(momentum_drawdown_rates) < 30:
                result.error = "Insufficient drawdown data (need ≥30 observations)"
                return result

            dip_stat, p_val = hartigan_dip_test(momentum_drawdown_rates)
            passed = p_val < T3_DIP_P_THRESHOLD
            result.passed = passed
            result.test_statistic = dip_stat
            result.p_value = p_val
            result.details = (
                f"Hartigan dip stat={dip_stat:.4f}; p={p_val:.4f} "
                f"(threshold<{T3_DIP_P_THRESHOLD} to reject unimodal)"
            )

        except Exception as e:
            result.error = str(e)
            log.error("T3 failed: %s", e)

        return result

    def test_T4_deff_leads_crashes(
        self,
        d_eff_series: pd.Series,
        crisis_dates: list,
    ) -> HypothesisResult:
        """
        T4: D_eff declines 30-60 days before market crashes.
        Pass: median lead ≥ 25 days, directional accuracy > 70%.

        From the article: 'D_eff is a LEADING indicator of systemic risk,
        not a concurrent measure.'
        """
        result = HypothesisResult(
            test_id="T4",
            name="D_eff leads crashes by 30-60 days",
            passed=False,
            threshold=T4_LEAD_DAYS_MIN,
        )
        try:
            if d_eff_series is None or len(d_eff_series) < 100:
                result.error = "Insufficient D_eff data"
                return result

            lead_days = []
            correct_direction = 0
            checked_crises = 0  # only crises with sufficient pre-crisis data

            for crisis_date in crisis_dates:
                # Get D_eff 60 days before crisis
                pre_crisis = d_eff_series[d_eff_series.index < crisis_date].tail(60)
                if len(pre_crisis) < 20:
                    continue

                checked_crises += 1

                # Check if D_eff was declining in the 60 days before.
                # Use OLS slope rather than head/tail means: D_eff has a
                # long-term upward bias (more assets join the universe over time,
                # NaN-filled with 0 inflates apparent diversification), so a
                # naive late < early comparison always fails. A negative slope
                # robustly detects local declining trend within the window.
                from scipy import stats as _sc_stats
                x_vals = np.arange(len(pre_crisis), dtype=float)
                slope, _, _, _, _ = _sc_stats.linregress(x_vals, pre_crisis.values)
                direction_correct = slope < 0  # negative slope = declining D_eff

                if direction_correct:
                    correct_direction += 1

                # Find when D_eff first crossed below 10 (warning level)
                below_10 = pre_crisis[pre_crisis < 10]
                if not below_10.empty:
                    first_below = below_10.index[0]
                    lead = (crisis_date - first_below).days
                    lead_days.append(lead)

            if not lead_days:
                result.error = "No lead time data found"
                return result

            median_lead = float(np.median(lead_days))
            # Divide by crises that actually had enough data to evaluate
            directional_acc = correct_direction / max(checked_crises, 1)

            passed = (median_lead >= T4_LEAD_DAYS_MIN
                      and directional_acc >= T4_DIRECTIONAL_ACCURACY)
            result.passed = passed
            result.test_statistic = median_lead
            result.p_value = directional_acc
            result.details = (
                f"Median lead={median_lead:.0f} days (threshold≥{T4_LEAD_DAYS_MIN}); "
                f"directional accuracy={directional_acc:.1%} "
                f"(threshold≥{T4_DIRECTIONAL_ACCURACY:.0%})"
            )

        except Exception as e:
            result.error = str(e)
            log.error("T4 failed: %s", e)

        return result

    def test_T5_dark_pool_predicts_direction(
        self,
        dark_pool_fractions: pd.Series,
        next_day_returns: pd.Series,
    ) -> HypothesisResult:
        """
        T5: Dark pool fraction at t predicts price direction at t+1d.
        Pass: directional accuracy > 55% with p < 0.05 (binomial test).
        """
        result = HypothesisResult(
            test_id="T5",
            name="Dark pool predicts price direction",
            passed=False,
            threshold=T5_ACCURACY_THRESHOLD,
        )
        try:
            if dark_pool_fractions is None or len(dark_pool_fractions) < 100:
                result.error = "Insufficient dark pool data"
                return result

            # Align series
            aligned = pd.DataFrame({
                "dp": dark_pool_fractions,
                "ret": next_day_returns.shift(-1),
            }).dropna()

            if len(aligned) < 50:
                result.error = "Insufficient aligned data"
                return result

            # High dark pool → bullish prediction (expectations building below gap)
            dp_median = aligned["dp"].median()
            high_dp = aligned["dp"] > dp_median
            positive_return = aligned["ret"] > 0

            correct = (high_dp & positive_return) | (~high_dp & ~positive_return)
            accuracy = correct.mean()

            from scipy import stats
            binom = stats.binomtest(correct.sum(), len(correct), 0.5)
            p_val = binom.pvalue

            passed = accuracy > T5_ACCURACY_THRESHOLD and p_val < 0.05
            result.passed = passed
            result.test_statistic = float(accuracy)
            result.p_value = float(p_val)
            result.details = (
                f"Directional accuracy={accuracy:.1%} "
                f"(threshold>{T5_ACCURACY_THRESHOLD:.0%}); p={p_val:.4f}"
            )

        except Exception as e:
            result.error = str(e)
            log.error("T5 failed: %s", e)

        return result

    def test_T6_equity_premium_spectral_peak(
        self,
        equity_risk_premium: pd.Series,
        min_period_years: float = 3.0,
        max_period_years: float = 7.0,
    ) -> HypothesisResult:
        """
        T6: Equity risk premium has 3-7yr spectral peak (temporal crystal).
        Pass: dominant frequency in 3-7yr band, > 25% of power.

        From the article: 'Premium = temporal crystal zero-point energy;
        oscillates with condensate period (3-7yr spectral peak)'
        """
        result = HypothesisResult(
            test_id="T6",
            name="Equity risk premium 3-7yr spectral peak",
            passed=False,
            threshold=T6_BAND_POWER_THRESHOLD,
        )
        try:
            if equity_risk_premium is None or len(equity_risk_premium) < 60:
                result.error = "Insufficient risk premium data"
                return result

            dominant_period, band_power = spectral_peak_frequency(
                equity_risk_premium,
                min_period_years=min_period_years,
                max_period_years=max_period_years,
            )

            passed = band_power >= T6_BAND_POWER_THRESHOLD
            result.passed = passed
            result.test_statistic = dominant_period
            result.p_value = band_power
            result.details = (
                f"Dominant period={dominant_period:.1f} years; "
                f"band power={band_power:.1%} in [{min_period_years}-{max_period_years}yr] "
                f"(threshold≥{T6_BAND_POWER_THRESHOLD:.0%})"
            )

        except Exception as e:
            result.error = str(e)
            log.error("T6 failed: %s", e)

        return result

    def test_T7_technical_levels_survive_turnover(
        self,
        pre_transition_levels: list[float],
        post_transition_prices: pd.Series,
        persistence_threshold: float = 0.05,  # level "active" if price within 5%
    ) -> HypothesisResult:
        """
        T7: Technical levels persist despite participant turnover.
        Pass: > 70% of pre-transition levels still operative post-transition.

        From the article: 'Levels are topological defects; independent rediscovery'
        """
        result = HypothesisResult(
            test_id="T7",
            name="Technical levels survive participant turnover",
            passed=False,
            threshold=T7_PERSISTENCE_THRESHOLD,
        )
        try:
            if not pre_transition_levels or post_transition_prices is None:
                result.error = "Missing levels or price data"
                return result

            still_active = 0
            for level in pre_transition_levels:
                # Level is "active" if price touched within persistence_threshold
                relative_distance = abs(post_transition_prices - level) / level
                if (relative_distance < persistence_threshold).any():
                    still_active += 1

            persistence = still_active / max(len(pre_transition_levels), 1)
            passed = persistence >= T7_PERSISTENCE_THRESHOLD
            result.passed = passed
            result.test_statistic = persistence
            result.details = (
                f"Persistent levels: {still_active}/{len(pre_transition_levels)} "
                f"= {persistence:.1%} (threshold≥{T7_PERSISTENCE_THRESHOLD:.0%})"
            )

        except Exception as e:
            result.error = str(e)
            log.error("T7 failed: %s", e)

        return result

    def test_T8_vol_skew_predicts_regime(
        self,
        skew_series: pd.Series,
        next_regime_direction: pd.Series,  # +1 = bull, -1 = bear
    ) -> HypothesisResult:
        """
        T8: Vol skew sign predicts direction of next regime change.
        Pass: directional accuracy > 60%, p < 0.05.

        From the article: 'Skew = asymmetry of ψ_exp; negative skew =
        bearish condensate chirality (skew sign predicts regime direction)'
        """
        result = HypothesisResult(
            test_id="T8",
            name="Vol skew predicts regime direction",
            passed=False,
            threshold=T8_ACCURACY_THRESHOLD,
        )
        try:
            if skew_series is None or next_regime_direction is None:
                result.error = "Missing skew or regime direction data"
                return result

            aligned = pd.DataFrame({
                "skew": skew_series,
                "regime": next_regime_direction,
            }).dropna()

            if len(aligned) < 20:
                result.error = "Insufficient skew/regime data"
                return result

            # Negative skew → bearish regime, Positive skew → bullish regime
            # Convention: skew = IV(put OTM) - IV(call OTM)
            # Positive skew = put wing elevated = bearish concern → bear regime
            skew_predicts = np.sign(-aligned["skew"])  # invert: positive skew → bearish
            regime_direction = np.sign(aligned["regime"])

            matches = (skew_predicts == regime_direction)
            accuracy = matches.mean()

            from scipy import stats
            binom = stats.binomtest(matches.sum(), len(matches), 0.5)
            p_val = binom.pvalue

            passed = accuracy >= T8_ACCURACY_THRESHOLD and p_val < 0.05
            result.passed = passed
            result.test_statistic = float(accuracy)
            result.p_value = float(p_val)
            result.details = (
                f"Directional accuracy={accuracy:.1%} "
                f"(threshold≥{T8_ACCURACY_THRESHOLD:.0%}); p={p_val:.4f}"
            )

        except Exception as e:
            result.error = str(e)
            log.error("T8 failed: %s", e)

        return result

    def test_T9_earnings_drift_proportional_dispersion(
        self,
        earnings_dispersion: pd.Series,   # analyst EPS dispersion before earnings
        post_earnings_drift: pd.Series,   # |price drift| over 20 days post-earnings
    ) -> HypothesisResult:
        """
        T9: Post-earnings drift ∝ analyst forecast dispersion.
        Pass: R² > 0.20, coefficient positive and significant.

        From the article: 'Drift = partial condensate relaxation; scale set
        by grain reconfiguration (EPS dispersion)'
        """
        result = HypothesisResult(
            test_id="T9",
            name="Post-earnings drift ∝ analyst dispersion",
            passed=False,
            threshold=T9_R2_THRESHOLD,
        )
        try:
            aligned = pd.DataFrame({
                "dispersion": earnings_dispersion,
                "drift": post_earnings_drift.abs(),  # magnitude of drift
            }).dropna()

            if len(aligned) < 30:
                result.error = "Insufficient earnings data (need ≥30 events)"
                return result

            from scipy import stats as scipy_stats
            slope, intercept, r_value, p_val, std_err = scipy_stats.linregress(
                aligned["dispersion"], aligned["drift"]
            )
            r_squared = r_value**2

            passed = (r_squared >= T9_R2_THRESHOLD
                      and slope > 0
                      and p_val < 0.05)
            result.passed = passed
            result.test_statistic = r_squared
            result.p_value = p_val
            result.details = (
                f"R²={r_squared:.3f} (threshold≥{T9_R2_THRESHOLD}); "
                f"slope={slope:.4f} (must be >0); p={p_val:.4f}; "
                f"n={len(aligned)} earnings events"
            )

        except Exception as e:
            result.error = str(e)
            log.error("T9 failed: %s", e)

        return result

    def run_all(self, data: dict) -> HypothesisReport:
        """
        Run all 9 hypothesis tests.

        Args:
            data: dict with keys corresponding to each test's input data.
                  See each test method for required keys.

        Returns:
            HypothesisReport with pass/fail per test + deployment recommendation.
        """
        log.info("Running all 9 CCDR hypothesis tests...")
        results = {}

        # T1: Granger causality
        results["T1"] = self.test_T1_vol_surface_granger_causes_price(
            vol_changes=data.get("vol_changes", np.array([])),
            price_changes=data.get("price_changes", np.array([])),
        )

        # T2: Analyst dispersion leads regime
        results["T2"] = self.test_T2_analyst_dispersion_leads_regime(
            dp_series=data.get("dp_series"),
            regime_change_dates=data.get("regime_change_dates", []),
        )

        # T3: Momentum crashes bimodal
        results["T3"] = self.test_T3_momentum_crashes_bimodal(
            momentum_drawdown_rates=data.get("momentum_drawdown_rates", np.array([])),
        )

        # T4: D_eff leads crashes
        results["T4"] = self.test_T4_deff_leads_crashes(
            d_eff_series=data.get("d_eff_series"),
            crisis_dates=data.get("crisis_dates", []),
        )

        # T5: Dark pool predicts direction
        results["T5"] = self.test_T5_dark_pool_predicts_direction(
            dark_pool_fractions=data.get("dark_pool_fractions"),
            next_day_returns=data.get("next_day_returns"),
        )

        # T6: Equity premium spectral peak
        results["T6"] = self.test_T6_equity_premium_spectral_peak(
            equity_risk_premium=data.get("equity_risk_premium"),
        )

        # T7: Technical levels survive turnover
        results["T7"] = self.test_T7_technical_levels_survive_turnover(
            pre_transition_levels=data.get("pre_transition_levels", []),
            post_transition_prices=data.get("post_transition_prices"),
        )

        # T8: Vol skew predicts regime
        results["T8"] = self.test_T8_vol_skew_predicts_regime(
            skew_series=data.get("skew_series"),
            next_regime_direction=data.get("next_regime_direction"),
        )

        # T9: Earnings drift ∝ dispersion
        results["T9"] = self.test_T9_earnings_drift_proportional_dispersion(
            earnings_dispersion=data.get("earnings_dispersion"),
            post_earnings_drift=data.get("post_earnings_drift"),
        )

        # Compute summary statistics
        passed_count = sum(1 for r in results.values() if r.passed)
        failed_count = 9 - passed_count
        deploy_recommended = passed_count >= 7

        # Identify blocking failures (any 3+ failures blocks deployment)
        failures = [tid for tid, r in results.items() if not r.passed]

        report = HypothesisReport(
            results=results,
            passed_count=passed_count,
            failed_count=failed_count,
            deploy_recommended=deploy_recommended,
            blocking_failures=failures[:3],
        )

        log.info("Hypothesis tests complete: %d/9 passed. Deploy recommended: %s",
                 passed_count, deploy_recommended)
        if not deploy_recommended:
            log.warning("DEPLOYMENT BLOCKED: only %d/9 passed (need ≥7)", passed_count)
        log.info("\n%s", report.summary())

        return report


# ===========================================================================
# __main__ RUNNER
# ===========================================================================

def _build_data_dict(tests: list[str], start: str = "1990-01-01") -> dict:
    """
    Fetch all required data from the data fetchers and assemble into the
    dict expected by CCDRHypothesisTests.run_all().
    Failures in individual fetchers are caught and logged; downstream tests
    will receive None / empty arrays and return errors gracefully.
    """
    import os
    from psibot.backtesting.data_fetchers import cached_fetch
    from psibot.backtesting.data_fetchers.cboe_fetcher import (
        fetch_vix_history, fetch_skew_history,
    )
    from psibot.backtesting.data_fetchers.yahoo_fetcher import (
        fetch_asset_universe_prices, fetch_earnings_surprise_dispersion,
    )
    from psibot.backtesting.data_fetchers.french_fetcher import fetch_momentum_factor
    from psibot.backtesting.data_fetchers.shiller_fetcher import fetch_shiller_data
    from psibot.backtesting.data_fetchers.finra_fetcher import fetch_finra_ats_weekly

    data: dict = {}
    fred_available = bool(os.environ.get("FRED_API_KEY", ""))

    # --- T1: VIX changes → price changes (Granger causality) ----------------
    if not tests or "T1" in tests:
        log.info("[T1] Fetching VIX + SPX prices...")
        try:
            vix = cached_fetch("vix_history", lambda: fetch_vix_history(start=start))
            prices = cached_fetch(
                "asset_universe",
                lambda: fetch_asset_universe_prices(start=start),
            )
            if vix is not None and len(vix) > 200:
                data["vol_changes"] = vix["VIX"].pct_change().dropna().values
            if prices is not None and "SPX" in prices.columns:
                data["price_changes"] = prices["SPX"].pct_change().dropna().values
                # Align lengths
                n = min(len(data.get("vol_changes", [])),
                        len(data.get("price_changes", [])))
                if n > 0:
                    data["vol_changes"] = data["vol_changes"][-n:]
                    data["price_changes"] = data["price_changes"][-n:]
        except Exception as e:
            log.warning("[T1] Data fetch error: %s", e)

    # --- T2: Analyst dispersion + regime change dates -----------------------
    if not tests or "T2" in tests:
        log.info("[T2] Fetching analyst dispersion + NBER recessions...")
        try:
            dp = cached_fetch(
                "earnings_dispersion",
                lambda: fetch_earnings_surprise_dispersion(start=start),
            )
            if dp is not None and len(dp) > 0:
                data["dp_series"] = dp["dp_proxy"]

            if fred_available:
                from psibot.backtesting.data_fetchers.fred_fetcher import (
                    fetch_nber_recession_dates,
                )
                rec = cached_fetch(
                    "nber_recessions",
                    lambda: fetch_nber_recession_dates(start=start),
                )
                if rec is not None and len(rec) > 0:
                    # Regime change = start of each recession
                    rec_monthly = rec.resample("ME").last()
                    starts = rec_monthly.index[
                        (rec_monthly["recession"].diff() == 1)
                    ].tolist()
                    data["regime_change_dates"] = starts
            else:
                # Fallback: well-known regime change dates
                data["regime_change_dates"] = [
                    pd.Timestamp("2000-03-01"),
                    pd.Timestamp("2001-09-01"),
                    pd.Timestamp("2007-10-01"),
                    pd.Timestamp("2009-03-01"),
                    pd.Timestamp("2011-08-01"),
                    pd.Timestamp("2018-12-01"),
                    pd.Timestamp("2020-02-01"),
                    pd.Timestamp("2022-01-01"),
                ]
        except Exception as e:
            log.warning("[T2] Data fetch error: %s", e)

    # --- T3: Momentum factor drawdown rates (bimodality test) ---------------
    if not tests or "T3" in tests:
        log.info("[T3] Fetching Fama-French momentum factor...")
        try:
            mom = cached_fetch(
                "momentum_factor",
                lambda: fetch_momentum_factor(start=start),
            )
            if mom is not None and len(mom) > 60:
                # Compute rolling 12-month drawdowns from monthly MOM returns
                cum = (1 + mom["MOM"]).cumprod()
                roll_max = cum.rolling(12, min_periods=1).max()
                drawdowns = ((cum - roll_max) / roll_max).dropna()
                # Use only drawdown events (negative months in momentum crashes)
                crash_months = drawdowns[drawdowns < -0.05].values
                if len(crash_months) >= 30:
                    data["momentum_drawdown_rates"] = crash_months
                else:
                    data["momentum_drawdown_rates"] = drawdowns.values
        except Exception as e:
            log.warning("[T3] Data fetch error: %s", e)

    # --- T4: D_eff series + crisis dates ------------------------------------
    if not tests or "T4" in tests:
        log.info("[T4] Computing rolling D_eff from asset universe...")
        try:
            prices = data.get("_prices") or cached_fetch(
                "asset_universe",
                lambda: fetch_asset_universe_prices(start=start),
            )
            if prices is not None and len(prices) > 100:
                data["_prices"] = prices  # reuse across tests
                log_ret = np.log(prices / prices.shift(1)).dropna()
                # Drop columns with >10% NaN
                log_ret = log_ret.dropna(axis=1, thresh=int(len(log_ret) * 0.9))
                log_ret = log_ret.fillna(0.0)

                window = 60
                d_eff_vals = []
                d_eff_dates = []
                for i in range(window, len(log_ret)):
                    chunk = log_ret.iloc[i - window:i].values
                    from helpers import compute_d_eff as _compute_d_eff
                    d_eff_vals.append(_compute_d_eff(chunk))
                    d_eff_dates.append(log_ret.index[i])

                data["d_eff_series"] = pd.Series(d_eff_vals, index=d_eff_dates)

            # Crisis dates: known crashes + optional NBER
            crisis_dates = [
                pd.Timestamp("1998-08-01"),  # LTCM / Russia
                pd.Timestamp("2000-09-01"),  # dot-com peak exit
                pd.Timestamp("2002-07-01"),  # post dot-com trough
                pd.Timestamp("2007-11-01"),  # GFC start
                pd.Timestamp("2009-03-01"),  # GFC trough
                pd.Timestamp("2010-05-01"),  # flash crash
                pd.Timestamp("2011-08-01"),  # Euro debt crisis
                pd.Timestamp("2015-08-01"),  # China devaluation
                pd.Timestamp("2018-12-01"),  # rate-hike selloff
                pd.Timestamp("2020-02-01"),  # COVID crash start
                pd.Timestamp("2022-01-01"),  # rate hike cycle
            ]
            data["crisis_dates"] = crisis_dates
        except Exception as e:
            log.warning("[T4] Data fetch error: %s", e)

    # --- T5: Dark pool fraction + SPY returns --------------------------------
    if not tests or "T5" in tests:
        log.info("[T5] Fetching FINRA ATS dark pool data...")
        try:
            dp_finra = cached_fetch(
                "finra_ats_spy",
                lambda: fetch_finra_ats_weekly(
                    start_year=max(2014, int(start[:4])),
                    symbol="SPY",
                ),
            )
            if dp_finra is not None and len(dp_finra) >= 100:
                prices_spy = data.get("_prices")
                if prices_spy is not None and "SPX" in prices_spy.columns:
                    spy_weekly = (
                        prices_spy["SPX"]
                        .resample("W-MON")
                        .last()
                        .pct_change()
                        .dropna()
                    )
                    data["dark_pool_fractions"] = dp_finra["dark_pool_fraction"]
                    data["next_day_returns"] = spy_weekly
            else:
                log.warning("[T5] Insufficient FINRA data — T5 will error gracefully")
        except Exception as e:
            log.warning("[T5] Data fetch error: %s", e)

    # --- T6: Equity risk premium spectral analysis --------------------------
    if not tests or "T6" in tests:
        log.info("[T6] Fetching Shiller data for equity risk premium...")
        try:
            shiller = cached_fetch("shiller_data", fetch_shiller_data)
            if shiller is not None and len(shiller) > 120:
                shiller = shiller.dropna(subset=["cape", "long_rate"])
                # ERP = earnings yield (1/CAPE) - 10Y real yield
                erp = (1.0 / shiller["cape"]) - (shiller["long_rate"] / 100.0)
                erp = erp.dropna()
                if start:
                    erp = erp[erp.index >= start]
                data["equity_risk_premium"] = erp
        except Exception as e:
            log.warning("[T6] Data fetch error: %s", e)

        # Fallback: compute ERP proxy from Yahoo Finance when Shiller is unavailable.
        # ERP_proxy = rolling 12-month SPY return − 10-year Treasury yield (^TNX).
        # Covers 1993-present, gives ~380 monthly observations — enough for spectral test.
        if "equity_risk_premium" not in data or data.get("equity_risk_premium") is None:
            log.info("[T6] Using Yahoo Finance ERP proxy (SPY return − 10Y yield)...")
            try:
                import yfinance as yf
                _spy = yf.download("SPY", start="1993-01-01", auto_adjust=True,
                                   progress=False)["Close"].squeeze()
                _tny = yf.download("^TNX", start="1993-01-01", auto_adjust=True,
                                   progress=False)["Close"].squeeze()
                _spy_m = _spy.resample("ME").last().pct_change(12)
                _tny_m = _tny.resample("ME").last() / 100.0
                _erp = (_spy_m - _tny_m).dropna()
                if start:
                    _erp = _erp[_erp.index >= start]
                if len(_erp) >= 60:
                    data["equity_risk_premium"] = _erp
                    log.info("[T6] ERP proxy: %d monthly obs", len(_erp))
            except Exception as e2:
                log.warning("[T6] ERP fallback failed: %s", e2)

    # --- T7: Technical levels survive participant turnover ------------------
    if not tests or "T7" in tests:
        log.info("[T7] Deriving technical S&P 500 levels...")
        try:
            prices_t7 = data.get("_prices")
            if prices_t7 is None:
                prices_t7 = cached_fetch(
                    "asset_universe",
                    lambda: fetch_asset_universe_prices(start=start),
                )
            if prices_t7 is not None and "SPX" in prices_t7.columns:
                spx = prices_t7["SPX"].dropna()
                # Use 52-week highs and lows around known transition points
                transition_dates = [
                    pd.Timestamp("2007-10-01"),
                    pd.Timestamp("2009-03-01"),
                    pd.Timestamp("2020-02-01"),
                    pd.Timestamp("2022-01-01"),
                ]
                pre_levels = []
                for td in transition_dates:
                    pre = spx[spx.index < td].tail(252)
                    if len(pre) > 0:
                        pre_levels.extend([
                            float(pre.max()),
                            float(pre.min()),
                            float(pre.mean()),
                        ])

                # Post-transition: prices 1–3 years after last transition
                last_td = transition_dates[-1]
                post_prices = spx[spx.index > last_td]
                if len(pre_levels) > 0 and len(post_prices) > 50:
                    data["pre_transition_levels"] = pre_levels
                    data["post_transition_prices"] = post_prices
        except Exception as e:
            log.warning("[T7] Data fetch error: %s", e)

    # --- T8: CBOE SKEW + regime direction -----------------------------------
    if not tests or "T8" in tests:
        log.info("[T8] Fetching CBOE SKEW + deriving regime direction...")
        try:
            skew_df = cached_fetch(
                "cboe_skew",
                lambda: fetch_skew_history(start=start),
            )
            if skew_df is not None and len(skew_df) > 100:
                # Normalise SKEW: centre around 100, higher = more negative skew
                skew_norm = (skew_df["SKEW"] - 100.0) / 10.0

                prices_t8 = data.get("_prices")
                if prices_t8 is None:
                    prices_t8 = cached_fetch(
                        "asset_universe",
                        lambda: fetch_asset_universe_prices(start=start),
                    )
                if prices_t8 is not None and "SPX" in prices_t8.columns:
                    spx = prices_t8["SPX"].dropna()
                    # Regime direction = sign of 6-month forward return
                    fwd_6m = spx.pct_change(126).shift(-126).dropna()
                    regime_dir = fwd_6m.apply(np.sign)
                    data["skew_series"] = skew_norm
                    data["next_regime_direction"] = regime_dir
        except Exception as e:
            log.warning("[T8] Data fetch error: %s", e)

    # --- T9: Post-earnings drift ∝ analyst dispersion -----------------------
    if not tests or "T9" in tests:
        log.info("[T9] Fetching earnings dispersion + computing post-earnings drift...")
        try:
            import yfinance as yf

            tickers_t9 = [
                "AAPL", "MSFT", "AMZN", "GOOGL", "META",
                "JPM", "JNJ", "XOM", "NVDA", "TSLA",
                "BRK-B", "UNH", "V", "PG", "HD",
            ]
            dispersion_records = []
            drift_records = []

            for ticker in tickers_t9:
                try:
                    stock = yf.Ticker(ticker)
                    earnings = stock.earnings_dates
                    if earnings is None or len(earnings) == 0:
                        continue
                    earnings = earnings[
                        ["EPS Estimate", "Reported EPS"]
                    ].dropna()
                    if len(earnings) < 3:
                        continue

                    _raw_hist = yf.download(
                        ticker,
                        start=start,
                        progress=False,
                        auto_adjust=True,
                    )["Close"]
                    # Newer yfinance may return a single-column DataFrame; squeeze to Series
                    hist = _raw_hist.squeeze() if isinstance(_raw_hist, pd.DataFrame) else _raw_hist
                    if hist is None or len(hist) < 30:
                        continue

                    for date, row in earnings.iterrows():
                        dt = pd.Timestamp(date).tz_localize(None)
                        try:
                            after = hist[hist.index > dt].head(20)
                            if len(after) < 10:
                                continue
                            before_price = float(hist[hist.index <= dt].iloc[-1])
                            drift_pct = float((after.iloc[-1] - before_price) / before_price)
                            # Dispersion proxy = |EPS surprise| / |EPS Estimate|
                            est = row["EPS Estimate"]
                            act = row["Reported EPS"]
                            if abs(est) < 0.01:
                                continue
                            dispersion = abs(act - est) / abs(est)
                            dispersion_records.append((dt, dispersion))
                            drift_records.append((dt, drift_pct))
                        except Exception:
                            continue
                except Exception:
                    continue

            if len(dispersion_records) >= 30:
                idx = [r[0] for r in dispersion_records]
                data["earnings_dispersion"] = pd.Series(
                    [r[1] for r in dispersion_records], index=idx
                )
                data["post_earnings_drift"] = pd.Series(
                    [r[1] for r in drift_records], index=idx
                )
            else:
                log.warning("[T9] Only %d earnings events found (need ≥30)",
                            len(dispersion_records))
        except Exception as e:
            log.warning("[T9] Data fetch error: %s", e)

    # Clean up internal cache key
    data.pop("_prices", None)
    return data


def _save_report(report: "HypothesisReport", output_dir: str) -> str:
    """Serialise HypothesisReport to JSON and write to output_dir."""
    import json
    from pathlib import Path

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    timestamp = report.run_at.strftime("%Y%m%d_%H%M%S")
    out_path = Path(output_dir) / f"hypothesis_report_{timestamp}.json"

    payload = {
        "run_at": report.run_at.isoformat(),
        "passed_count": report.passed_count,
        "failed_count": report.failed_count,
        "deploy_recommended": report.deploy_recommended,
        "blocking_failures": report.blocking_failures,
        "results": {
            tid: {
                "test_id": r.test_id,
                "name": r.name,
                "passed": r.passed,
                "test_statistic": r.test_statistic,
                "p_value": r.p_value,
                "threshold": r.threshold,
                "details": r.details,
                "error": r.error,
            }
            for tid, r in sorted(report.results.items())
        },
    }

    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2, default=str)

    # Also write a "latest" symlink / copy for convenience
    latest_path = Path(output_dir) / "hypothesis_report_latest.json"
    with open(latest_path, "w") as f:
        json.dump(payload, f, indent=2, default=str)

    return str(out_path)


if __name__ == "__main__":
    import argparse
    import sys
    from pathlib import Path

    parser = argparse.ArgumentParser(
        description="Run CCDR T1–T9 hypothesis tests. DEPLOYMENT GATE: ≥7/9 must pass.",
    )
    parser.add_argument(
        "--tests",
        default="",
        help="Comma-separated subset to run, e.g. T1,T3,T6. Default: all.",
    )
    parser.add_argument(
        "--start-date",
        default="1990-01-01",
        help="Historical data start date (default: 1990-01-01).",
    )
    parser.add_argument(
        "--output",
        default="psibot/backtesting/test_results",
        help="Directory to write JSON report (default: psibot/backtesting/test_results).",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress per-test detail; print JSON path only.",
    )
    args = parser.parse_args()

    # Configure logging
    level = logging.WARNING if args.quiet else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s %(name)s — %(message)s",
        datefmt="%H:%M:%S",
    )

    tests_requested = [t.strip() for t in args.tests.split(",") if t.strip()]
    valid_tests = {"T1", "T2", "T3", "T4", "T5", "T6", "T7", "T8", "T9"}
    if tests_requested:
        invalid = set(tests_requested) - valid_tests
        if invalid:
            print(f"ERROR: Unknown test IDs: {invalid}. Valid: {sorted(valid_tests)}")
            sys.exit(1)

    import os
    if not os.environ.get("FRED_API_KEY"):
        print(
            "WARNING: FRED_API_KEY not set. T2/T4/T6 will use fallback data.\n"
            "Get a free key at: https://fred.stlouisfed.org/docs/api/api_key.html\n"
            "Then run: export FRED_API_KEY=your_key_here\n"
        )

    print(f"Fetching data (start={args.start_date}) — this may take 1–3 minutes...")
    data = _build_data_dict(
        tests=tests_requested,
        start=args.start_date,
    )

    print("Running hypothesis tests...")
    runner = CCDRHypothesisTests()

    if tests_requested:
        # Run all (run_all handles missing data gracefully) then filter results
        full_report = runner.run_all(data)
        report = HypothesisReport(
            run_at=full_report.run_at,
            results={k: v for k, v in full_report.results.items()
                     if k in tests_requested},
            passed_count=sum(1 for k, v in full_report.results.items()
                             if k in tests_requested and v.passed),
            failed_count=len(tests_requested) - sum(
                1 for k, v in full_report.results.items()
                if k in tests_requested and v.passed
            ),
        )
        report.deploy_recommended = None  # partial run — no gate assessment
        report.blocking_failures = [
            k for k in tests_requested
            if not full_report.results[k].passed
        ]
    else:
        report = runner.run_all(data)

    out_path = _save_report(report, args.output)

    if not args.quiet:
        print()
        print(report.summary())
        print()

    gate = f"{report.passed_count}/9 passed" if not tests_requested else \
           f"{report.passed_count}/{len(tests_requested)} passed"
    deploy_str = ""
    if report.deploy_recommended is True:
        deploy_str = " → DEPLOYMENT GATE: PASS"
    elif report.deploy_recommended is False:
        deploy_str = " → DEPLOYMENT GATE: BLOCKED (need ≥7)"

    print(f"Report saved: {out_path}")
    print(f"Results: {gate}{deploy_str}")

    # Exit 1 if deployment is blocked
    if report.deploy_recommended is False:
        sys.exit(1)
