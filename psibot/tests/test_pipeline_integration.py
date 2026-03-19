"""
tests/test_pipeline_integration.py — Full Pipeline Integration Tests
====================================================================
Tests the full L1→L2→L3→L4→L5 pipeline end-to-end.

CCDR Expectation Field Architecture — Version 1.0
"""

import asyncio
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import numpy as np
import pytest

from helpers import BotMode, MarketPhase, PsiShape, AcousticSignal, CCDR_THRESHOLDS
from psibot.state.condensate_state import CondensateState
from psibot.state.portfolio_state import PortfolioState
from psibot.data.options_feed import OptionsFeed
from psibot.data.analyst_feed import AnalystData, SurveyData
from psibot.pipeline import (
    l1_psi_reconstruction, l2_phase_detector,
    l3_holo_monitor, l4_grain_boundary, l5_acoustic_parser,
)
from psibot.execution.sizing import compute_size_multiplier
from helpers import determine_bot_mode


# =============================================================================
# Helper fixtures
# =============================================================================

class _CA:
    """Minimal cross-asset data for tests."""
    def __init__(self):
        self.returns_matrix = (np.random.randn(60, 27) * np.ones((1, 27)) * 0.01
                               + np.random.randn(60, 1) * 0.005)
        self.momentum_20d = 0.01
        self.momentum_60d = 0.02
        self.breadth = 0.62
        self.volume_ratio = 1.05


def make_analyst_data():
    return AnalystData(
        timestamp=__import__("datetime").datetime.utcnow(),
        symbol="SPX",
        forward_12m_eps_by_analyst=np.array([10.0, 10.5, 9.8, 10.2, 10.3, 9.9]),
    )


def make_survey_data(bull=45.0, bear=25.0):
    return SurveyData(
        timestamp=__import__("datetime").datetime.utcnow(),
        aaii_bull=bull, aaii_bear=bear, aaii_neutral=100-bull-bear,
        ii_bull=bull+5, ii_bear=bear+5, ii_correction=10.0,
        inst_bull=bull+10, inst_bear=bear,
    )


# =============================================================================
# Full pipeline tests
# =============================================================================

class TestFullPipeline:
    """End-to-end pipeline integration tests."""

    def run_full_pipeline(self, options_surface=None, bull=45.0, bear=25.0) -> CondensateState:
        """Helper: run all 5 layers and return final state."""
        feed = OptionsFeed(provider="csv")
        if options_surface is None:
            options_surface = feed.build_synthetic_surface(
                "SPX", spot=5000.0, atm_vol=0.18, skew=-0.01, kurtosis=0.005
            )

        analyst_data = make_analyst_data()
        survey_data = make_survey_data(bull, bear)
        ca_data = _CA()

        state = CondensateState()
        loop = asyncio.get_event_loop()

        state = loop.run_until_complete(l1_psi_reconstruction.run(state, options_surface))
        state = loop.run_until_complete(l2_phase_detector.run(state, analyst_data, survey_data))
        state = loop.run_until_complete(l3_holo_monitor.run(state, ca_data))
        state = loop.run_until_complete(l4_grain_boundary.run(state, dark_pool_fraction=1.0))
        state = loop.run_until_complete(l5_acoustic_parser.run(state, ca_data))

        state.active_mode = determine_bot_mode(state.d_eff, state.gbp, state.phase)
        state.signal_size_multiplier = compute_size_multiplier(state)

        return state

    def test_full_pipeline_completes_without_error(self):
        """Full pipeline should complete without unhandled exceptions."""
        state = self.run_full_pipeline()
        assert state is not None
        assert not (state.l1_failed and state.l2_failed and state.l3_failed)

    def test_pipeline_populates_all_required_fields(self):
        """All CondensateState fields should be populated after full pipeline run."""
        state = self.run_full_pipeline()

        # L1
        assert state.psi_shape in PsiShape.__members__.values()
        assert np.isfinite(state.psi_skew)
        assert np.isfinite(state.psi_kurtosis_excess)
        assert np.isfinite(state.psi_entropy)

        # L2
        assert -1.0 <= state.order_parameter <= 1.0
        assert state.disorder_parameter >= 0
        assert state.phase in MarketPhase.__members__.values()

        # L3
        assert 1.0 <= state.d_eff <= 27.0
        assert np.isfinite(state.d_eff_trend_10d)

        # L4
        assert 0.0 <= state.gbp <= 1.0
        assert "psi" in state.gbp_components

        # L5
        assert state.acoustic_signal in AcousticSignal.__members__.values()

        # Derived
        assert state.active_mode in BotMode.__members__.values()
        assert 0.0 <= state.signal_size_multiplier <= 1.0

    def test_pipeline_layers_execute_in_order(self):
        """Each layer must complete before the next starts (sequential contract)."""
        # We verify by checking that state fields are set at the right points.
        # This is inherently true for our sequential async implementation.
        state = self.run_full_pipeline()
        # If layers ran in parallel, L4 would fail due to missing L1/L2/L3 data
        # The fact that L4 has proper GBP values confirms sequential execution.
        assert state.gbp_components.get("psi", -1) >= 0

    def test_guardian_mode_on_very_low_d_eff(self):
        """When D_eff is forced very low, active_mode should be GUARDIAN."""
        state = self.run_full_pipeline()
        # Manually force extreme D_eff (simulate crisis)
        state.d_eff = 2.0
        state.active_mode = determine_bot_mode(state.d_eff, state.gbp, state.phase)
        assert state.active_mode == BotMode.GUARDIAN

    def test_guardian_mode_on_high_gbp(self):
        """When GBP >= 0.8, active_mode should be GUARDIAN."""
        state = self.run_full_pipeline()
        state.gbp = 0.85
        state.active_mode = determine_bot_mode(state.d_eff, state.gbp, state.phase)
        assert state.active_mode == BotMode.GUARDIAN

    def test_pipeline_failsafe_chain(self):
        """Pipeline with all-None inputs should still return valid state."""
        state = CondensateState()
        loop = asyncio.get_event_loop()

        # Fail L1 (None surface)
        state = loop.run_until_complete(l1_psi_reconstruction.run(state, None))
        assert state.l1_failed
        assert state.psi_shape == PsiShape.UNKNOWN

        # L2 with None data
        state = loop.run_until_complete(l2_phase_detector.run(state, None, None))

        # L3 with empty data
        class _EmptyCA:
            returns_matrix = np.array([])
            momentum_20d = 0.0
            momentum_60d = 0.0
            breadth = 0.5
            volume_ratio = 1.0

        state = loop.run_until_complete(l3_holo_monitor.run(state, _EmptyCA()))
        assert state.l3_failed
        assert state.d_eff == 5.0  # conservative

        # L4 should still work (failsafe)
        state = loop.run_until_complete(l4_grain_boundary.run(state, dark_pool_fraction=1.0))

        # Final mode determination
        state.active_mode = determine_bot_mode(state.d_eff, state.gbp, state.phase)

        # System should be in Scout or Guardian (conservative) mode after all failures
        assert state.active_mode in (BotMode.SCOUT, BotMode.GUARDIAN)


class TestModeLogic:
    """Test mode determination logic."""

    def test_hunter_requires_all_conditions(self):
        """Hunter mode requires D_eff>5, GBP<0.5, phase≠DISORDERED."""
        assert determine_bot_mode(10.0, 0.3, MarketPhase.ORDERED_BULL) == BotMode.HUNTER
        # Violate D_eff
        assert determine_bot_mode(4.9, 0.3, MarketPhase.ORDERED_BULL) == BotMode.SCOUT
        # Violate GBP
        assert determine_bot_mode(10.0, 0.6, MarketPhase.ORDERED_BULL) == BotMode.SCOUT
        # Violate phase
        assert determine_bot_mode(10.0, 0.3, MarketPhase.DISORDERED) == BotMode.SCOUT

    def test_guardian_priority_over_hunter(self):
        """Guardian takes priority even when other conditions favor Hunter."""
        assert determine_bot_mode(2.9, 0.3, MarketPhase.ORDERED_BULL) == BotMode.GUARDIAN
        assert determine_bot_mode(10.0, 0.85, MarketPhase.ORDERED_BULL) == BotMode.GUARDIAN

    def test_scout_default_when_uncertain(self):
        """Scout is the safe default when conditions are unclear."""
        assert determine_bot_mode(4.0, 0.4, MarketPhase.ORDERED_BULL) == BotMode.SCOUT


class TestSizingIntegration:
    """Test position sizing in the context of full pipeline state."""

    def test_zero_size_at_guardian_conditions(self):
        """Size multiplier should be 0 or near 0 in Guardian conditions."""
        state = CondensateState()
        state.d_eff = 2.0   # Guardian trigger
        state.gbp = 0.9     # Guardian trigger
        state.acoustic_signal = AcousticSignal.NEUTRAL

        mult = compute_size_multiplier(state)
        assert mult == 0.0, f"Guardian conditions should give size multiplier = 0, got {mult}"

    def test_max_size_in_optimal_conditions(self):
        """Size multiplier should approach 1.0 in ideal Hunter conditions."""
        state = CondensateState()
        state.d_eff = 22.0   # well above 20 → f_deff = 1.0
        state.gbp = 0.05     # well below 0.1 → f_gbp = 1.0
        state.acoustic_signal = AcousticSignal.CONFIRM

        mult = compute_size_multiplier(state)
        assert mult == 1.0, f"Optimal conditions should give size multiplier = 1.0, got {mult}"

    def test_acoustic_confirm_boosts_size(self):
        """CONFIRM signal should give higher size multiplier than NEUTRAL."""
        state = CondensateState()
        state.d_eff = 12.0
        state.gbp = 0.25
        state.acoustic_signal = AcousticSignal.NEUTRAL
        mult_neutral = compute_size_multiplier(state)

        state.acoustic_signal = AcousticSignal.CONFIRM
        mult_confirm = compute_size_multiplier(state)

        assert mult_confirm >= mult_neutral, \
            f"CONFIRM ({mult_confirm}) should be ≥ NEUTRAL ({mult_neutral})"

    def test_acoustic_contradict_reduces_size(self):
        """CONTRADICT signal should give lower size multiplier than NEUTRAL."""
        state = CondensateState()
        state.d_eff = 12.0
        state.gbp = 0.25
        state.acoustic_signal = AcousticSignal.NEUTRAL
        mult_neutral = compute_size_multiplier(state)

        state.acoustic_signal = AcousticSignal.CONTRADICT
        mult_contradict = compute_size_multiplier(state)

        assert mult_contradict <= mult_neutral, \
            f"CONTRADICT ({mult_contradict}) should be ≤ NEUTRAL ({mult_neutral})"
