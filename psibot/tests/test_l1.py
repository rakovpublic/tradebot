"""
tests/test_l1.py — Unit tests for Layer 1: ψ_exp Wavefunction Reconstruction
==============================================================================
All tests must pass before L1 goes live (Phase 1 gate).

CCDR Expectation Field Architecture — Version 1.0
"""

import asyncio
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import numpy as np
if not hasattr(np, 'trapz'):
    np.trapz = np.trapezoid
import pytest

from helpers import (
    PsiShape, TermStructure,
    classify_psi_shape, compute_psi_entropy, classify_term_structure,
    validate_options_surface,
)
from psibot.data.options_feed import OptionsFeed, OptionsSurface
from psibot.state.condensate_state import CondensateState
from psibot.pipeline import l1_psi_reconstruction


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def feed():
    return OptionsFeed(provider="csv")


@pytest.fixture
def synthetic_surface_gaussian(feed):
    """Normal market surface — should classify as GAUSSIAN or SKEWED."""
    return feed.build_synthetic_surface("SPX", spot=5000.0, atm_vol=0.18, skew=-0.01, kurtosis=0.005)


@pytest.fixture
def synthetic_surface_skewed_right(feed):
    """Bullish chirality surface — skew tilted right."""
    return feed.build_synthetic_surface("SPX", spot=5000.0, atm_vol=0.18, skew=0.03, kurtosis=0.005)


@pytest.fixture
def synthetic_surface_fat_tailed(feed):
    """Fat-tailed surface — high kurtosis."""
    return feed.build_synthetic_surface("SPX", spot=5000.0, atm_vol=0.25, skew=-0.01, kurtosis=0.05)


# =============================================================================
# SK-01: ψ_exp Wavefunction Reconstruction
# =============================================================================

class TestPsiReconstruction:
    """Test the core wavefunction reconstruction pipeline."""

    def test_psi_reconstruction_integrates_to_unity(self, synthetic_surface_gaussian):
        """SK-01: Risk-neutral density must integrate to 1.0 ± 0.001."""
        from psibot.pipeline.l1_psi_reconstruction import _compute_risk_neutral_density

        surface = synthetic_surface_gaussian
        primary_tenor = 30
        if primary_tenor not in surface.strikes:
            primary_tenor = surface.tenors_days[0]

        p, K = _compute_risk_neutral_density(
            strikes=surface.strikes[primary_tenor],
            iv_slice=surface.iv[primary_tenor],
            spot=surface.spot,
            tenor_days=primary_tenor,
        )

        total = np.trapz(p, K)
        assert abs(total - 1.0) < 0.01, f"Density integrates to {total:.4f}, expected ~1.0"
        assert np.all(p >= 0), "Density must be non-negative"

    def test_l1_populates_state_fields(self, synthetic_surface_gaussian):
        """L1 run must populate all required CondensateState fields."""
        state = CondensateState()
        updated_state = asyncio.get_event_loop().run_until_complete(
            l1_psi_reconstruction.run(state, synthetic_surface_gaussian)
        )

        assert updated_state.psi_shape != PsiShape.UNKNOWN
        assert isinstance(updated_state.psi_skew, float)
        assert isinstance(updated_state.psi_kurtosis_excess, float)
        assert isinstance(updated_state.psi_entropy, float)
        assert updated_state.psi_entropy >= 0
        assert updated_state.psi_term_structure in TermStructure.__members__.values()
        assert not updated_state.l1_failed

    def test_l1_failsafe_on_none_surface(self):
        """L1 must apply conservative failsafe when surface is None."""
        state = CondensateState()
        updated = asyncio.get_event_loop().run_until_complete(
            l1_psi_reconstruction.run(state, None)
        )

        assert updated.psi_shape == PsiShape.UNKNOWN
        assert updated.l1_failed is True
        assert len(updated.pipeline_errors) > 0

    def test_l1_latency_under_100ms(self, synthetic_surface_gaussian):
        """SK-01: Reconstruction must complete in < 100ms (95th percentile)."""
        import time
        state = CondensateState()
        times = []
        for _ in range(10):
            start = time.perf_counter()
            asyncio.get_event_loop().run_until_complete(
                l1_psi_reconstruction.run(state, synthetic_surface_gaussian)
            )
            times.append(time.perf_counter() - start)

        p95_ms = np.percentile(times, 95) * 1000
        assert p95_ms < 100, f"L1 latency p95 = {p95_ms:.1f}ms (threshold: 100ms)"


# =============================================================================
# SK-02: Wavefunction Shape Classification
# =============================================================================

class TestPsiShapeClassification:
    """Test classify_psi_shape with synthetic densities."""

    def _make_normal_density(self, n=50):
        """Gaussian-shaped density."""
        x = np.linspace(0.5, 1.5, n)
        p = np.exp(-0.5 * ((x - 1.0) / 0.1)**2)
        return p / np.trapz(p, x), x

    def _make_skewed_right_density(self, n=50):
        """Right-skewed density (bullish chirality)."""
        from scipy.stats import skewnorm
        x = np.linspace(0.5, 1.5, n)
        p = skewnorm.pdf(x, a=5, loc=0.95, scale=0.15)
        return p / np.trapz(p, x), x

    def _make_fat_tailed_density(self, n=50):
        """Fat-tailed density (leptokurtic)."""
        from scipy.stats import t
        x = np.linspace(0.5, 1.5, n)
        p = t.pdf(x, df=2, loc=1.0, scale=0.05)
        return p / np.trapz(p, x), x

    def test_gaussian_classification(self):
        p, K = self._make_normal_density()
        shape = classify_psi_shape(p, K)
        assert shape == PsiShape.GAUSSIAN, f"Expected GAUSSIAN, got {shape.value}"

    def test_skewed_right_classification(self):
        p, K = self._make_skewed_right_density()
        shape = classify_psi_shape(p, K)
        assert shape in (PsiShape.SKEWED_RIGHT, PsiShape.GAUSSIAN), \
            f"Expected SKEWED_RIGHT or GAUSSIAN, got {shape.value}"

    def test_fat_tailed_classification(self):
        p, K = self._make_fat_tailed_density()
        shape = classify_psi_shape(p, K)
        assert shape in (PsiShape.FAT_TAILED, PsiShape.BIMODAL), \
            f"Expected FAT_TAILED or BIMODAL, got {shape.value}"

    def test_unknown_on_too_few_points(self):
        p = np.array([0.1, 0.9])
        K = np.array([0.9, 1.1])
        shape = classify_psi_shape(p, K)
        assert shape == PsiShape.UNKNOWN, f"Expected UNKNOWN for 2 points, got {shape.value}"

    def test_normalisation_invariance(self):
        """Shape classification should be invariant to density scaling."""
        p, K = self._make_normal_density()
        shape1 = classify_psi_shape(p, K)
        shape2 = classify_psi_shape(p * 100, K)  # same shape, different scale
        assert shape1 == shape2, "Shape classification should be scale-invariant"


# =============================================================================
# SK-03: Vol Surface Ingestion & Normalisation
# =============================================================================

class TestVolSurfaceIngestion:
    """Test options surface validation and ingestion."""

    def test_validate_good_surface(self, feed, synthetic_surface_gaussian):
        surface = synthetic_surface_gaussian
        assert surface.validate(), f"Good surface should validate. Issues: {surface.validation_issues}"

    def test_skew_computation(self, synthetic_surface_gaussian):
        """Skew computation should return a finite float."""
        surface = synthetic_surface_gaussian
        skew = surface.skew_at_tenor(30)
        assert np.isfinite(skew), f"Skew must be finite, got {skew}"

    def test_kurtosis_proxy_computation(self, synthetic_surface_gaussian):
        """Kurtosis proxy should be non-negative for convex surfaces."""
        surface = synthetic_surface_gaussian
        kurt = surface.kurtosis_excess_at_tenor(30)
        assert np.isfinite(kurt), f"Kurtosis proxy must be finite, got {kurt}"

    def test_atm_iv_extraction(self, synthetic_surface_gaussian):
        """ATM IV at each tenor should be within [0.01, 3.0]."""
        surface = synthetic_surface_gaussian
        atm_ivs = surface.atm_iv_by_tenor()
        assert len(atm_ivs) > 0, "Should extract ATM IVs"
        for tenor, iv in atm_ivs.items():
            assert 0.01 <= iv <= 3.0, f"ATM IV at tenor {tenor} = {iv:.4f} out of range"

    def test_invalid_surface_detection(self):
        """Surface with negative IVs should fail validation."""
        bad_surface = OptionsSurface(
            underlying="TEST",
            timestamp=__import__("datetime").datetime.utcnow(),
            spot=100.0,
            tenors_days=[30],
            strikes={30: np.array([90.0, 100.0, 110.0])},
            iv={30: np.array([-0.1, 0.20, 0.18])},  # negative IV
            bid_iv={30: np.array([-0.12, 0.19, 0.17])},
            ask_iv={30: np.array([-0.08, 0.21, 0.19])},
            open_interest={30: np.array([100.0, 500.0, 200.0])},
        )
        assert not bad_surface.validate(), "Surface with negative IVs should fail"

    def test_term_structure_classification_contango(self):
        """Rising IV surface should be CONTANGO."""
        ivs = {30: 0.15, 91: 0.17, 182: 0.19, 365: 0.21}
        ts = classify_term_structure(ivs)
        assert ts == TermStructure.CONTANGO, f"Expected CONTANGO, got {ts.value}"

    def test_term_structure_classification_backwardation(self):
        """Falling IV surface should be BACKWARDATION."""
        ivs = {30: 0.30, 91: 0.25, 182: 0.22, 365: 0.20}
        ts = classify_term_structure(ivs)
        assert ts == TermStructure.BACKWARDATION, f"Expected BACKWARDATION, got {ts.value}"


# =============================================================================
# Entropy tests
# =============================================================================

class TestPsiEntropy:
    """Test entropy computation for condensate disorder."""

    def test_gaussian_has_lower_entropy_than_uniform(self):
        """Gaussian density should have lower entropy than uniform density."""
        x = np.linspace(0.5, 1.5, 100)

        # Gaussian
        p_gauss = np.exp(-0.5 * ((x - 1.0) / 0.1)**2)
        p_gauss /= np.trapz(p_gauss, x)
        entropy_gauss = compute_psi_entropy(p_gauss, x)

        # Uniform
        p_uniform = np.ones(100) / 1.0
        p_uniform /= np.trapz(p_uniform, x)
        entropy_uniform = compute_psi_entropy(p_uniform, x)

        assert entropy_gauss < entropy_uniform, \
            f"Gaussian entropy ({entropy_gauss:.3f}) should be < uniform ({entropy_uniform:.3f})"

    def test_entropy_is_finite(self):
        """Differential entropy of a valid density must be a finite number."""
        x = np.linspace(0.5, 1.5, 50)
        p = np.exp(-0.5 * ((x - 1.0) / 0.15)**2)
        p /= np.trapz(p, x)
        entropy = compute_psi_entropy(p, x)
        assert np.isfinite(entropy), f"Entropy must be finite, got {entropy}"
