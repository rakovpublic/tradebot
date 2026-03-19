"""
pipeline/l2_phase_detector.py — AGT-02: Condensate Phase Detector (SK-04, SK-05, SK-06)
=========================================================================================
Measures the order parameter (OP) and disorder parameter (DP) from analyst
and survey data, then classifies the condensate phase.

From the article (Section 4.1):
  'Bull and Bear Markets as Condensate Phases — the order parameter is
   collective expectation coherence. The transition is a genuine topological
   phase transition: the order parameter passes through zero before
   re-establishing in the new direction.'

Update frequency: Daily (analyst data) + Weekly (surveys).
CCDR Expectation Field Architecture — Version 1.0
"""

from __future__ import annotations

import logging
from collections import deque
from datetime import datetime
from typing import Optional
import numpy as np

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from helpers import MarketPhase, classify_market_phase

log = logging.getLogger("psibot.pipeline.l2")

# Historical OP/DP for trend computation
_op_history: deque = deque(maxlen=20)  # last 20 business days
_dp_history: deque = deque(maxlen=20)


async def run(state, analyst_data, survey_data=None) -> object:
    """
    Layer 2 main entry point.
    Populates L2 fields in CondensateState.

    Args:
        state: CondensateState
        analyst_data: AnalystData from data/analyst_feed.py
        survey_data: SurveyData from data/analyst_feed.py (optional, weekly)

    Returns:
        Updated CondensateState
    """
    try:
        # Check staleness
        if analyst_data is not None and analyst_data.is_stale:
            log.warning("L2: analyst data stale (%d business days old)",
                        analyst_data.age_business_days)
            state.analyst_data_stale = True
            state.analyst_data_age_days = analyst_data.age_business_days

        # Compute disorder parameter DP from analyst dispersion
        dp = _compute_dp(analyst_data)

        # Compute order parameter OP from survey data
        op = _compute_op(survey_data)

        # Update history for trend computation
        _dp_history.append(dp)
        _op_history.append(op)

        # Compute trends
        op_trend_5d = _compute_trend(_op_history, window=5)
        dp_trend_5d = _compute_trend(_dp_history, window=5)
        dp_trend_10d = _compute_trend(_dp_history, window=10)

        # Classify condensate phase
        phase = classify_market_phase(
            op=op,
            dp=dp,
            op_trend_5d=op_trend_5d,
            dp_trend_5d=dp_trend_5d,
        )

        # Check for phase transition — alert Risk agent if phase changes
        if state.phase != MarketPhase.UNKNOWN and state.phase != phase:
            _handle_phase_transition(state.phase, phase)

        # Write to state
        state.order_parameter = op
        state.disorder_parameter = dp
        state.phase = phase
        state.op_trend_5d = op_trend_5d
        state.dp_trend_5d = dp_trend_5d
        state.dp_trend_10d = dp_trend_10d
        state.l2_failed = False

        log.info("L2 complete: phase=%s OP=%.3f DP=%.4f dp_trend_10d=%.4f",
                 phase.value, op, dp, dp_trend_10d)
        return state

    except Exception as e:
        log.error("L2 phase detection failed: %s — using last known values", e, exc_info=True)
        state.pipeline_errors.append(f"L2: {e}")
        return _apply_l2_failsafe(state, str(e))


def _compute_dp(analyst_data) -> float:
    """
    DP = σ(EPS forecasts) / |mean(EPS forecast)|
    High DP = high dispersion = approaching grain boundary.
    """
    if analyst_data is None:
        return 0.1  # neutral fallback
    dp = analyst_data.disorder_parameter()
    if np.isnan(dp):
        return 0.1
    return float(np.clip(dp, 0.0, 5.0))


def _compute_op(survey_data) -> float:
    """
    OP = (% bullish - % bearish) / 100, weighted across survey sources.
    OP ∈ [-1, +1]: +1 = fully bullish condensate, -1 = fully bearish.
    """
    if survey_data is None:
        return 0.0  # neutral
    return float(survey_data.order_parameter())


def _compute_trend(history: deque, window: int) -> float:
    """Compute change over last `window` observations."""
    if len(history) < 2:
        return 0.0
    effective_window = min(window, len(history))
    values = list(history)
    return float(values[-1] - values[-effective_window])


def _handle_phase_transition(old_phase: MarketPhase, new_phase: MarketPhase) -> None:
    """
    Phase transition event handler.
    In production: publish to event bus for Risk Agent (AGT-07).
    Phase transitions immediately notify the Risk Agent regardless of other conditions.
    """
    log.warning("PHASE TRANSITION: %s → %s", old_phase.value, new_phase.value)
    # Event bus integration point:
    # event_bus.publish("phase_transition", {"old": old_phase, "new": new_phase})
    if new_phase == MarketPhase.DISORDERED:
        log.warning("DISORDERED PHASE DETECTED — grain boundary crossing in progress")


def _apply_l2_failsafe(state, reason: str) -> object:
    """
    L2 failure protocol: use last known values with staleness flag.
    Phase remains at last known value; staleness flagged.
    """
    log.warning("L2 FAILSAFE: %s — using last known OP=%.3f DP=%.4f phase=%s",
                reason, state.order_parameter, state.disorder_parameter, state.phase.value)
    state.analyst_data_stale = True
    state.l2_failed = True
    state.pipeline_errors.append(f"L2 failsafe: {reason}")
    # Phase unchanged — last known value used
    return state
