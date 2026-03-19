"""
pipeline/l3_holo_monitor.py — AGT-03: Holographic Saturation Monitor (SK-07, SK-08)
======================================================================================
Computes D_eff — the effective dimensionality of the cross-asset expectation condensate.
This is the MOST CRITICAL RISK MANAGEMENT SIGNAL in the system.

From the article (Section 5):
  'D_eff is a leading indicator of systemic risk, not a concurrent measure.
   It measures the complexity of the expectation field before the crash
   propagates into the acoustic (price) sector.'

  'D_eff = -log(Σ λi²) / log(N)
   D_eff = 1: crisis (dimensional reduction complete)
   D_eff = N: fully diversified (maximum complexity)'

Hard trigger: D_eff < 3.0 → immediate Guardian activation.

CCDR Expectation Field Architecture — Version 1.0
"""

from __future__ import annotations

import logging
from collections import deque
from datetime import datetime
from typing import Optional
import numpy as np
import pandas as pd

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from helpers import BotMode, compute_d_eff, compute_d_eff_trend, d_eff_to_bot_mode, CCDR_THRESHOLDS

log = logging.getLogger("psibot.pipeline.l3")

# Rolling D_eff history for trend computation
_d_eff_history: deque = deque(maxlen=30)  # 30 trading days of history


async def run(state, cross_asset_data) -> object:
    """
    Layer 3 main entry point.
    Populates L3 fields: d_eff, d_eff_trend_10d, d_eff_trend_20d, bot_mode_from_deff.

    Hard trigger: if d_eff < 3.0 → emit immediate Guardian alert.

    Args:
        state: CondensateState
        cross_asset_data: CrossAssetData from data/cross_asset_feed.py

    Returns:
        Updated CondensateState
    """
    try:
        returns_matrix = cross_asset_data.returns_matrix

        if returns_matrix is None or returns_matrix.size == 0:
            log.error("L3: no returns matrix — applying failsafe")
            return _apply_l3_failsafe(state, "returns matrix empty")

        # Compute D_eff (SK-07)
        d_eff = compute_d_eff(returns_matrix)

        # Update rolling history
        _d_eff_history.append(d_eff)
        d_eff_series = pd.Series(list(_d_eff_history))

        # Compute trend (SK-08)
        d_eff_trend_10d = compute_d_eff_trend(d_eff_series, window=10)
        d_eff_trend_20d = compute_d_eff_trend(d_eff_series, window=20)

        # Determine mode implied by D_eff alone
        bot_mode_from_deff = d_eff_to_bot_mode(d_eff)

        # Check hard Guardian trigger (D_eff < 3.0)
        if d_eff <= CCDR_THRESHOLDS["D_EFF_GUARDIAN"]:
            _emit_guardian_alert(d_eff, "holographic saturation — D_eff ≤ 3.0")

        # Write to state
        state.d_eff = d_eff
        state.d_eff_trend_10d = d_eff_trend_10d
        state.d_eff_trend_20d = d_eff_trend_20d
        state.bot_mode_from_deff = bot_mode_from_deff
        state.l3_failed = False

        log.info("L3 complete: D_eff=%.2f trend_10d=%.3f/day mode_from_deff=%s",
                 d_eff, d_eff_trend_10d, bot_mode_from_deff.value)
        return state

    except Exception as e:
        log.error("L3 D_eff computation failed: %s — applying failsafe", e, exc_info=True)
        state.pipeline_errors.append(f"L3: {e}")
        return _apply_l3_failsafe(state, str(e))


def _emit_guardian_alert(d_eff: float, reason: str) -> None:
    """
    Emit high-priority Guardian trigger event.
    In production: publishes immediately to event bus, does NOT wait for pipeline cycle.
    AGT-07 (Risk) is subscribed and will activate Guardian mode.
    """
    log.critical("GUARDIAN TRIGGER FROM L3: D_eff=%.2f — %s", d_eff, reason)
    # Event bus integration point (production):
    # event_bus.publish("guardian_trigger", {"source": "L3", "d_eff": d_eff, "reason": reason})


def _apply_l3_failsafe(state, reason: str) -> object:
    """
    L3 failure protocol:
    - d_eff = 5.0 (conservative mid-range → Scout mode)
    - Trend = 0.0 (no trend signal available)
    - mode_from_deff = SCOUT (conservative)
    - Block new Hunter positions until D_eff recovers
    """
    log.warning("L3 FAILSAFE: %s — setting d_eff=5.0 (conservative Scout)", reason)
    state.d_eff = 5.0  # conservative: Scout trigger threshold
    state.d_eff_trend_10d = 0.0
    state.d_eff_trend_20d = 0.0
    state.bot_mode_from_deff = BotMode.SCOUT
    state.l3_failed = True
    state.pipeline_errors.append(f"L3 failsafe: {reason}")
    return state


def get_d_eff_history() -> list[float]:
    """Return copy of D_eff rolling history (for backtesting/monitoring)."""
    return list(_d_eff_history)


def inject_d_eff_history(history: list[float]) -> None:
    """Inject historical D_eff values (for backtesting replay)."""
    global _d_eff_history
    _d_eff_history = deque(history, maxlen=30)
