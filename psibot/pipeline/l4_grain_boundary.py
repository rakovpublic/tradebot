"""
pipeline/l4_grain_boundary.py — AGT-04: Grain Boundary Proximity (SK-09)
=========================================================================
Synthesises all upstream signals (L1, L2, L3) plus dark pool data into
the single most important number: GBP ∈ [0, 1].

GBP = w1*f(ψ_shape) + w2*f(DP_trend) + w3*f(D_eff_trend) + w4*f(dark_pool)
Weights: 0.35 / 0.25 / 0.30 / 0.10

From the article (Section 4.2):
  'Grain boundaries = competing narratives with irreconcilable differences'
  'Price = weighted centroid of all grain helicity centers'
  'Alpha lies in identifying grain boundaries before the acoustic sector catches up.'

Hard trigger: GBP >= 0.8 → immediate Guardian activation.

CCDR Expectation Field Architecture — Version 1.0
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Optional
import numpy as np

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from helpers import compute_gbp, CCDR_THRESHOLDS

log = logging.getLogger("psibot.pipeline.l4")


async def run(state, dark_pool_fraction: float) -> object:
    """
    Layer 4 main entry point.
    Synthesises L1+L2+L3 outputs into GBP score.

    Args:
        state: CondensateState (with L1, L2, L3 fields populated)
        dark_pool_fraction: current dark pool ratio (current / 90d avg)
                           from DarkPoolData.dark_pool_ratio

    Returns:
        Updated CondensateState with gbp and gbp_components populated
    """
    try:
        # Validate upstream data is populated
        if state.l1_failed and state.l3_failed:
            log.error("L4: both L1 and L3 failed — applying failsafe")
            return _apply_l4_failsafe(state, "L1 and L3 both failed")

        # Compute GBP components
        gbp, components = compute_gbp(
            psi_shape=state.psi_shape,
            dp_trend_10d=state.dp_trend_10d,
            d_eff_trend_20d=state.d_eff_trend_20d,
            dark_pool_ratio=dark_pool_fraction,
        )

        # Check hard Guardian trigger (GBP >= 0.8)
        if gbp >= CCDR_THRESHOLDS["GBP_GUARDIAN"]:
            _emit_guardian_alert(gbp, state)

        # Check warning threshold (GBP > 0.6 → Scout)
        if gbp > 0.6:
            log.warning("GBP WARNING: gbp=%.3f > 0.6 — grain boundary approaching", gbp)

        # Write to state
        state.gbp = gbp
        state.gbp_components = components
        state.l4_failed = False

        log.info("L4 complete: GBP=%.3f components=%s",
                 gbp, {k: round(v, 3) for k, v in components.items()})
        return state

    except Exception as e:
        log.error("L4 GBP computation failed: %s — applying failsafe", e, exc_info=True)
        state.pipeline_errors.append(f"L4: {e}")
        return _apply_l4_failsafe(state, str(e))


def _emit_guardian_alert(gbp: float, state) -> None:
    """
    Emit high-priority Guardian trigger event.
    GBP >= 0.8 = grain boundary crossing in progress.
    """
    log.critical("GUARDIAN TRIGGER FROM L4: GBP=%.3f >= 0.8 — grain boundary crossing", gbp)
    log.critical("GBP breakdown: psi=%.3f dp=%.3f deff=%.3f dark=%.3f",
                 state.gbp_components.get("psi", 0),
                 state.gbp_components.get("dp_trend", 0),
                 state.gbp_components.get("deff_trend", 0),
                 state.gbp_components.get("dark_pool", 0))
    # Event bus integration point (production):
    # event_bus.publish("guardian_trigger", {"source": "L4", "gbp": gbp, "state": state})


def _apply_l4_failsafe(state, reason: str) -> object:
    """
    L4 failure protocol:
    - gbp = 0.6 (conservative — between warning and Guardian)
    - Block new Hunter signals until GBP recoverable
    """
    log.warning("L4 FAILSAFE: %s — setting GBP=0.60 (conservative)", reason)
    state.gbp = 0.60
    state.gbp_components = {
        "psi": 0.60, "dp_trend": 0.60, "deff_trend": 0.60,
        "dark_pool": 0.60, "gbp": 0.60,
    }
    state.l4_failed = True
    state.pipeline_errors.append(f"L4 failsafe: {reason}")
    return state
