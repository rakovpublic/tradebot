"""
tests/test_l3.py — Unit tests for Layer 3: D_eff Holographic Monitor
=====================================================================
All tests must pass before L3 goes live (Phase 1 gate).

CCDR Expectation Field Architecture — Version 1.0
"""

import asyncio
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import numpy as np
import pandas as pd
import pytest

from helpers import (
    BotMode, compute_d_eff, compute_d_eff_trend, d_eff_to_bot_mode,
    d_eff_to_size_factor, CCDR_THRESHOLDS,
)
from psibot.state.condensate_state import CondensateState
from psibot.pipeline import l3_holo_monitor


class _SyntheticCAData:
    """Simple cross-asset data wrapper for tests."""
    def __init__(self, returns_matrix):
        self.returns_matrix = returns_matrix
        self.momentum_20d = 0.01
        self.momentum_60d = 0.02
        self.breadth = 0.6
        self.volume_ratio = 1.0


def make_returns_matrix(N=27, window=60, d_eff_target=12.0, rng_seed=42):
    """
    Generate synthetic return matrix with approximate target D_eff.
    Higher correlation → lower D_eff (approaching crisis).
    """
    rng = np.random.default_rng(rng_seed)

    # Approximate: more factors = more D_eff
    n_factors = max(1, int(d_eff_target))

    factors = rng.standard_normal((window, min(n_factors, N)))
    loads = rng.standard_normal((N, min(n_factors, N))) * 0.3
    idio = rng.standard_normal((window, N)) * 0.01

    returns = factors @ loads.T + idio
    return returns.astype(float)


class TestDeffComputation:
    """Test compute_d_eff core algorithm (SK-07)."""

    def test_d_eff_range_normal_markets(self):
        """D_eff should be in [1, 27] for any valid returns matrix."""
        returns = make_returns_matrix(N=27, window=60, d_eff_target=12.0)
        d_eff = compute_d_eff(returns)
        assert 1.0 <= d_eff <= 27.0, f"D_eff={d_eff} out of range [1, 27]"

    def test_d_eff_lower_when_highly_correlated(self):
        """Highly correlated returns should give lower (or equal) D_eff."""
        rng = np.random.default_rng(42)

        # Diverse returns: many independent factors
        factors_diverse = rng.standard_normal((60, 27))
        returns_diverse = factors_diverse  # fully independent

        # Highly correlated: one common factor dominates
        common = rng.standard_normal((60, 1))
        idio = rng.standard_normal((60, 27)) * 0.1
        returns_corr = common * np.ones((1, 27)) + idio

        d_eff_diverse = compute_d_eff(returns_diverse)
        d_eff_corr = compute_d_eff(returns_corr)

        assert d_eff_diverse >= d_eff_corr, \
            f"Diverse D_eff ({d_eff_diverse:.4f}) should be ≥ correlated D_eff ({d_eff_corr:.4f})"

    def test_d_eff_returns_conservative_on_small_data(self):
        """D_eff should return 5.0 (conservative) when data < 10 rows."""
        returns = np.random.randn(5, 27)
        d_eff = compute_d_eff(returns)
        assert d_eff == 5.0, f"Expected 5.0 for small data, got {d_eff}"

    def test_d_eff_handles_nan_gracefully(self):
        """D_eff should handle NaN/Inf in returns without crashing."""
        returns = make_returns_matrix(N=27, window=60)
        returns[5, 3] = np.nan
        returns[10, 7] = np.inf
        try:
            d_eff = compute_d_eff(returns)
            assert np.isfinite(d_eff), f"D_eff must be finite, got {d_eff}"
        except Exception as e:
            pytest.fail(f"compute_d_eff raised exception with NaN data: {e}")


class TestDeffTrend:
    """Test D_eff trend computation (SK-08)."""

    def test_declining_trend_is_negative(self):
        """Declining D_eff series should give negative slope."""
        history = pd.Series(list(range(15, 5, -1)), dtype=float)  # 15,14,...,6
        trend = compute_d_eff_trend(history, window=10)
        assert trend < 0, f"Declining D_eff should give negative trend, got {trend}"

    def test_rising_trend_is_positive(self):
        """Rising D_eff series should give positive slope."""
        history = pd.Series(list(range(5, 20)), dtype=float)  # 5,6,...,19
        trend = compute_d_eff_trend(history, window=10)
        assert trend > 0, f"Rising D_eff should give positive trend, got {trend}"

    def test_trend_zero_on_insufficient_data(self):
        """Trend should be 0.0 when < window observations."""
        history = pd.Series([10.0, 11.0, 9.0])
        trend = compute_d_eff_trend(history, window=20)
        assert trend == 0.0, f"Expected 0.0 for insufficient data, got {trend}"


class TestDeffBotMode:
    """Test D_eff → BotMode mapping."""

    def test_guardian_below_3(self):
        assert d_eff_to_bot_mode(2.9) == BotMode.GUARDIAN
        assert d_eff_to_bot_mode(1.0) == BotMode.GUARDIAN

    def test_scout_between_3_and_5(self):
        assert d_eff_to_bot_mode(3.1) == BotMode.SCOUT
        assert d_eff_to_bot_mode(5.0) == BotMode.SCOUT

    def test_hunter_above_5(self):
        assert d_eff_to_bot_mode(5.1) == BotMode.HUNTER
        assert d_eff_to_bot_mode(20.0) == BotMode.HUNTER


class TestDeffSizeFactor:
    """Test D_eff → size factor mapping."""

    def test_zero_at_floor(self):
        assert d_eff_to_size_factor(3.0) == 0.0
        assert d_eff_to_size_factor(1.0) == 0.0

    def test_one_at_ceiling(self):
        assert d_eff_to_size_factor(20.0) == 1.0
        assert d_eff_to_size_factor(25.0) == 1.0

    def test_monotone_increasing(self):
        d_effs = [3.0, 5.0, 8.0, 12.0, 15.0, 20.0]
        factors = [d_eff_to_size_factor(d) for d in d_effs]
        for i in range(len(factors) - 1):
            assert factors[i] <= factors[i+1], \
                f"Size factor not monotone: {factors}"


class TestL3Pipeline:
    """Integration tests for the L3 pipeline layer."""

    def test_l3_run_populates_state(self):
        """L3 must populate d_eff, d_eff_trend_10d, bot_mode_from_deff."""
        returns = make_returns_matrix(N=27, window=60, d_eff_target=12.0)
        state = CondensateState()
        ca_data = _SyntheticCAData(returns)

        updated = asyncio.get_event_loop().run_until_complete(
            l3_holo_monitor.run(state, ca_data)
        )

        assert 1.0 <= updated.d_eff <= 27.0
        assert np.isfinite(updated.d_eff_trend_10d)
        assert updated.bot_mode_from_deff in (BotMode.SCOUT, BotMode.HUNTER, BotMode.GUARDIAN)
        assert not updated.l3_failed

    def test_l3_failsafe_on_empty_data(self):
        """L3 must apply failsafe (d_eff=5.0) when returns matrix is empty."""
        state = CondensateState()
        ca_data = _SyntheticCAData(np.array([]))

        updated = asyncio.get_event_loop().run_until_complete(
            l3_holo_monitor.run(state, ca_data)
        )

        assert updated.d_eff == 5.0
        assert updated.bot_mode_from_deff == BotMode.SCOUT
        assert updated.l3_failed is True

    def test_l3_latency_under_50ms(self):
        """SK-03: D_eff computation must complete in < 50ms (27×27 eigenvalue)."""
        import time
        returns = make_returns_matrix(N=27, window=60)
        state = CondensateState()
        ca_data = _SyntheticCAData(returns)

        times = []
        for _ in range(10):
            start = time.perf_counter()
            asyncio.get_event_loop().run_until_complete(
                l3_holo_monitor.run(state, ca_data)
            )
            times.append(time.perf_counter() - start)

        p95_ms = np.percentile(times, 95) * 1000
        assert p95_ms < 50, f"L3 latency p95 = {p95_ms:.1f}ms (threshold: 50ms)"
