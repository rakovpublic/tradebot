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
            if dp_series is None or len(dp_series) < 100:
                result.error = "Insufficient DP data (need ≥100 observations)"
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

            for crisis_date in crisis_dates:
                # Get D_eff 60 days before crisis
                pre_crisis = d_eff_series[d_eff_series.index < crisis_date].tail(60)
                if len(pre_crisis) < 20:
                    continue

                # Check if D_eff was declining in the 60 days before
                early = pre_crisis.head(20).mean()
                late = pre_crisis.tail(20).mean()
                direction_correct = late < early  # declining = correct signal

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
            directional_acc = correct_direction / max(len(crisis_dates), 1)

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
