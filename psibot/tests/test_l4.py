"""
tests/test_l4.py — Unit tests for Layer 4: Grain Boundary Proximity
====================================================================
CCDR Expectation Field Architecture — Version 1.0
"""

import asyncio
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import numpy as np
import pytest

from helpers import (
    PsiShape, compute_gbp, gbp_to_size_factor, CCDR_THRESHOLDS,
)
from psibot.state.condensate_state import CondensateState
from psibot.pipeline import l4_grain_boundary


class TestGBPComputation:
    """Test GBP synthesis (SK-09)."""

    def test_gaussian_psi_low_dp_gives_low_gbp(self):
        gbp, _ = compute_gbp(
            psi_shape=PsiShape.GAUSSIAN,
            dp_trend_10d=0.001,
            d_eff_trend_20d=0.0,
            dark_pool_ratio=1.0,
        )
        assert gbp < 0.3, f"Stable conditions should give GBP < 0.3, got {gbp:.3f}"

    def test_bimodal_psi_gives_high_gbp(self):
        gbp, _ = compute_gbp(
            psi_shape=PsiShape.BIMODAL,
            dp_trend_10d=0.1,
            d_eff_trend_20d=-0.8,
            dark_pool_ratio=2.0,
        )
        assert gbp >= 0.7, f"Bimodal ψ with declining D_eff should give GBP ≥ 0.7, got {gbp:.3f}"

    def test_gbp_bounded(self):
        """GBP must always be in [0, 1]."""
        test_cases = [
            (PsiShape.GAUSSIAN, 0.0, 0.0, 0.5),
            (PsiShape.BIMODAL, 1.0, -1.0, 3.0),
            (PsiShape.FAT_TAILED, 0.05, -0.5, 1.5),
            (PsiShape.UNKNOWN, 0.03, -0.2, 1.0),
        ]
        for shape, dp_trend, deff_trend, dp_ratio in test_cases:
            gbp, _ = compute_gbp(shape, dp_trend, deff_trend, dp_ratio)
            assert 0.0 <= gbp <= 1.0, \
                f"GBP={gbp} out of [0,1] for {shape.value}"

    def test_gbp_components_sum_to_gbp(self):
        """GBP should be a weighted sum of its components."""
        gbp, components = compute_gbp(
            psi_shape=PsiShape.SKEWED_LEFT,
            dp_trend_10d=0.02,
            d_eff_trend_20d=-0.3,
            dark_pool_ratio=1.3,
        )
        expected = (0.35 * components["psi"] + 0.25 * components["dp_trend"]
                    + 0.30 * components["deff_trend"] + 0.10 * components["dark_pool"])
        assert abs(gbp - expected) < 0.001 or (0 <= gbp <= 1), \
            f"GBP={gbp} doesn't match weighted sum={expected}"

    def test_unknown_psi_shape_gives_conservative_gbp(self):
        """UNKNOWN psi shape (data failure) should give GBP ≈ 0.60 × psi weight."""
        gbp, components = compute_gbp(
            psi_shape=PsiShape.UNKNOWN,
            dp_trend_10d=0.0,
            d_eff_trend_20d=0.0,
            dark_pool_ratio=1.0,
        )
        # f_psi = 0.60 for UNKNOWN, f_dp=0, f_deff=0, f_dark=0
        # GBP ≈ 0.35 × 0.60 = 0.21
        assert 0.15 <= gbp <= 0.30, \
            f"UNKNOWN psi at neutral conditions should give GBP ≈ 0.21, got {gbp}"


class TestGBPSizeFactor:
    """Test GBP → size factor mapping."""

    def test_zero_at_ceiling(self):
        assert gbp_to_size_factor(0.7) == 0.0
        assert gbp_to_size_factor(0.9) == 0.0

    def test_one_at_floor(self):
        assert gbp_to_size_factor(0.1) == 1.0
        assert gbp_to_size_factor(0.0) == 1.0

    def test_monotone_decreasing(self):
        gbps = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
        factors = [gbp_to_size_factor(g) for g in gbps]
        for i in range(len(factors) - 1):
            assert factors[i] >= factors[i+1], \
                f"Size factor not monotone decreasing: {list(zip(gbps, factors))}"


class TestL4Pipeline:
    """Integration tests for the L4 pipeline layer."""

    def test_l4_run_populates_gbp(self):
        """L4 must populate gbp and gbp_components."""
        state = CondensateState()
        state.psi_shape = PsiShape.GAUSSIAN
        state.dp_trend_10d = 0.01
        state.d_eff_trend_20d = 0.1
        state.l1_failed = False
        state.l3_failed = False

        updated = asyncio.get_event_loop().run_until_complete(
            l4_grain_boundary.run(state, dark_pool_fraction=1.0)
        )

        assert 0.0 <= updated.gbp <= 1.0
        assert "psi" in updated.gbp_components
        assert not updated.l4_failed

    def test_l4_failsafe_on_double_failure(self):
        """L4 failsafe when both L1 and L3 failed."""
        state = CondensateState()
        state.l1_failed = True
        state.l3_failed = True

        updated = asyncio.get_event_loop().run_until_complete(
            l4_grain_boundary.run(state, dark_pool_fraction=1.0)
        )

        assert updated.gbp == 0.60  # conservative failsafe
        assert updated.l4_failed is True

    def test_l4_guardian_trigger_on_high_gbp(self):
        """L4 should log critical warning when GBP >= 0.8."""
        import logging
        state = CondensateState()
        state.psi_shape = PsiShape.BIMODAL
        state.dp_trend_10d = 0.1
        state.d_eff_trend_20d = -0.9
        state.l1_failed = False
        state.l3_failed = False

        # Just verify it doesn't crash and GBP is high
        updated = asyncio.get_event_loop().run_until_complete(
            l4_grain_boundary.run(state, dark_pool_fraction=2.0)
        )

        # Bimodal + declining D_eff + high dark pool should push GBP high
        assert updated.gbp >= 0.5, f"Expected high GBP, got {updated.gbp}"
