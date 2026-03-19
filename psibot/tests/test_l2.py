"""
tests/test_l2.py — Unit tests for Layer 2: Condensate Phase Detector
====================================================================
CCDR Expectation Field Architecture — Version 1.0
"""

import asyncio
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import numpy as np
import pytest

from helpers import (
    MarketPhase, classify_market_phase,
)
from psibot.data.analyst_feed import AnalystFeed, AnalystData, SurveyData
from psibot.state.condensate_state import CondensateState
from psibot.pipeline import l2_phase_detector


class TestOrderParameter:
    """Test order parameter computation from survey data."""

    def test_bullish_survey_gives_positive_op(self):
        survey = SurveyData(
            timestamp=__import__("datetime").datetime.utcnow(),
            aaii_bull=60.0, aaii_bear=20.0, aaii_neutral=20.0,
            ii_bull=55.0, ii_bear=25.0, ii_correction=20.0,
            inst_bull=65.0, inst_bear=15.0,
        )
        op = survey.order_parameter()
        assert op > 0, f"Bullish survey should give positive OP, got {op}"

    def test_bearish_survey_gives_negative_op(self):
        survey = SurveyData(
            timestamp=__import__("datetime").datetime.utcnow(),
            aaii_bull=20.0, aaii_bear=60.0, aaii_neutral=20.0,
            ii_bull=25.0, ii_bear=55.0, ii_correction=20.0,
            inst_bull=15.0, inst_bear=65.0,
        )
        op = survey.order_parameter()
        assert op < 0, f"Bearish survey should give negative OP, got {op}"

    def test_neutral_survey_gives_op_near_zero(self):
        survey = SurveyData(
            timestamp=__import__("datetime").datetime.utcnow(),
            aaii_bull=33.0, aaii_bear=33.0, aaii_neutral=34.0,
            ii_bull=40.0, ii_bear=40.0, ii_correction=20.0,
            inst_bull=50.0, inst_bear=50.0,
        )
        op = survey.order_parameter()
        assert abs(op) < 0.05, f"Neutral survey should give OP ~0, got {op}"

    def test_op_bounded(self):
        survey = SurveyData(
            timestamp=__import__("datetime").datetime.utcnow(),
            aaii_bull=100.0, aaii_bear=0.0,
            ii_bull=100.0, ii_bear=0.0,
            inst_bull=100.0, inst_bear=0.0,
        )
        op = survey.order_parameter()
        assert -1.0 <= op <= 1.0, f"OP must be in [-1, 1], got {op}"


class TestDisorderParameter:
    """Test disorder parameter (analyst dispersion) computation."""

    def test_high_dispersion_gives_high_dp(self):
        data = AnalystData(
            timestamp=__import__("datetime").datetime.utcnow(),
            symbol="SPX",
            forward_12m_eps_by_analyst=np.array([5.0, 15.0, 20.0, 2.0, 25.0]),  # high spread
        )
        dp = data.disorder_parameter()
        assert dp > 0.5, f"High dispersion should give DP > 0.5, got {dp}"

    def test_low_dispersion_gives_low_dp(self):
        data = AnalystData(
            timestamp=__import__("datetime").datetime.utcnow(),
            symbol="SPX",
            forward_12m_eps_by_analyst=np.array([10.0, 10.1, 9.9, 10.05, 9.95]),
        )
        dp = data.disorder_parameter()
        assert dp < 0.05, f"Low dispersion should give DP < 0.05, got {dp}"

    def test_dp_non_negative(self):
        data = AnalystData(
            timestamp=__import__("datetime").datetime.utcnow(),
            symbol="SPX",
            forward_12m_eps_by_analyst=np.array([10.0, 11.0, 9.0]),
        )
        dp = data.disorder_parameter()
        assert dp >= 0, f"DP must be ≥ 0, got {dp}"


class TestPhaseClassification:
    """Test condensate phase classification."""

    def test_strong_bullish_op_gives_ordered_bull(self):
        phase = classify_market_phase(op=0.6, dp=0.1, op_trend_5d=0.01, dp_trend_5d=-0.01)
        assert phase == MarketPhase.ORDERED_BULL

    def test_strong_bearish_op_gives_ordered_bear(self):
        phase = classify_market_phase(op=-0.6, dp=0.1, op_trend_5d=-0.01, dp_trend_5d=-0.01)
        assert phase == MarketPhase.ORDERED_BEAR

    def test_low_op_high_dp_gives_disordered(self):
        phase = classify_market_phase(op=0.05, dp=0.5, op_trend_5d=0.0, dp_trend_5d=0.05)
        assert phase == MarketPhase.DISORDERED

    def test_rising_op_declining_dp_gives_reordering(self):
        phase = classify_market_phase(op=0.05, dp=0.3, op_trend_5d=0.05, dp_trend_5d=-0.05)
        assert phase == MarketPhase.REORDERING

    def test_ordered_phases_is_ordered(self):
        assert MarketPhase.ORDERED_BULL.is_ordered()
        assert MarketPhase.ORDERED_BEAR.is_ordered()
        assert not MarketPhase.DISORDERED.is_ordered()
        assert not MarketPhase.REORDERING.is_ordered()


class TestL2Pipeline:
    """Integration tests for the L2 pipeline layer."""

    def test_l2_run_populates_state(self):
        feed = AnalystFeed(provider="csv")
        state = CondensateState()

        analyst_data = AnalystData(
            timestamp=__import__("datetime").datetime.utcnow(),
            symbol="SPX",
            forward_12m_eps_by_analyst=np.array([10.0, 10.5, 9.8, 10.2, 10.3]),
        )
        survey_data = SurveyData(
            timestamp=__import__("datetime").datetime.utcnow(),
            aaii_bull=45.0, aaii_bear=25.0, aaii_neutral=30.0,
            ii_bull=50.0, ii_bear=30.0, ii_correction=20.0,
            inst_bull=55.0, inst_bear=25.0,
        )

        updated = asyncio.get_event_loop().run_until_complete(
            l2_phase_detector.run(state, analyst_data, survey_data)
        )

        assert -1.0 <= updated.order_parameter <= 1.0
        assert updated.disorder_parameter >= 0
        assert updated.phase != MarketPhase.UNKNOWN
        assert not updated.l2_failed

    def test_l2_failsafe_preserves_last_known(self):
        """L2 failsafe should preserve last known values."""
        state = CondensateState()
        state.order_parameter = 0.45
        state.disorder_parameter = 0.12
        state.phase = MarketPhase.ORDERED_BULL

        # Pass None analyst_data to trigger error path in failsafe
        # Here we test the failsafe function directly
        from psibot.pipeline.l2_phase_detector import _apply_l2_failsafe
        updated = _apply_l2_failsafe(state, "test failure")

        assert updated.phase == MarketPhase.ORDERED_BULL  # preserved
        assert updated.analyst_data_stale is True
        assert updated.l2_failed is True
