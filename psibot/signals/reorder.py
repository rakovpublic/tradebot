"""
signals/reorder.py — SK-14: Reorder Signal Generation
=======================================================
Reorder = first mover into a new condensate grain after re-ordering.

From the article (Section 4.1):
  'This is what major market turns look like: a period of confused,
   high-volatility, directionless price action before the new trend
   establishes. CCDR identifies it as the disordered phase of the
   condensate during a phase transition.'

Entry: phase==REORDERING (new grain nucleating), psi_shape==GAUSSIAN,
       GBP<0.30, DP declining.
Exit: GBP < 0.2 sustained 3+ days (trade complete), or phase changes.

Signal class: SK-14
CCDR Expectation Field Architecture — Version 1.0
"""

from __future__ import annotations

import logging
import uuid
from collections import deque
from datetime import datetime
from typing import Optional

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from helpers import (
    BotMode, MarketPhase, PsiShape, AcousticSignal,
    SignalClass, SignalDirection, CCDR_THRESHOLDS,
)

log = logging.getLogger("psibot.signals.reorder")

# Reorder entry thresholds (from agents.md SK-14)
REORDER_GBP_MAX = 0.30
REORDER_INSTRUMENTS = ["SPX_FUT", "NDX_FUT", "RUT_FUT"]
REORDER_GBP_EXIT_THRESHOLD = 0.20
REORDER_GBP_EXIT_DAYS = 3


def check_reorder_signal(state, portfolio) -> Optional[dict]:
    """
    Evaluate reorder entry conditions.

    Entry conditions (ALL required):
      □ active_mode == HUNTER
      □ phase == REORDERING (new grain nucleating)
      □ psi_shape == GAUSSIAN (new stable condensate forming)
      □ gbp < 0.30
      □ dp_trend_5d < 0 (disorder declining — new order forming)
    """
    # Must be in Hunter mode
    if state.active_mode != BotMode.HUNTER:
        return None

    # Phase must be REORDERING — new grain actively nucleating
    if state.phase != MarketPhase.REORDERING:
        return None

    # ψ must be Gaussian — new stable condensate forming
    if state.psi_shape != PsiShape.GAUSSIAN:
        return None

    # GBP must be low — new grain interior forming
    if state.gbp > REORDER_GBP_MAX:
        return None

    # DP must be declining — disorder resolving
    if state.dp_trend_5d >= 0:
        return None

    # Max 1 Reorder position at a time
    existing = [p for p in portfolio.open_positions
                if p.signal_class == SignalClass.REORDER]
    if existing:
        return None

    # Direction: follow OP sign (which direction new grain is forming)
    direction = (SignalDirection.LONG
                 if state.order_parameter >= 0
                 else SignalDirection.SHORT)

    signal = {
        "signal_id": str(uuid.uuid4()),
        "signal_class": SignalClass.REORDER,
        "direction": direction,
        "size_multiplier": state.signal_size_multiplier * 0.8,  # size up as grain matures
        "entry_gbp": state.gbp,
        "entry_phase": state.phase,
        "entry_psi_shape": state.psi_shape,
        "entry_d_eff": state.d_eff,
        "instruments": REORDER_INSTRUMENTS,
        "rationale": (
            f"Reorder-{direction.value}: new grain nucleating, "
            f"ψ=GAUSSIAN, GBP={state.gbp:.3f}, "
            f"OP={state.order_parameter:.3f}, "
            f"DP_trend_5d={state.dp_trend_5d:.4f} (declining)"
        ),
        "exit_conditions": {
            "gbp_exit_threshold": REORDER_GBP_EXIT_THRESHOLD,
            "gbp_exit_days": REORDER_GBP_EXIT_DAYS,
            "phase_exit": [MarketPhase.DISORDERED],
        },
        "timestamp": datetime.utcnow(),
    }

    log.info("REORDER SIGNAL: %s | ψ=GAUSSIAN GBP=%.3f OP=%.3f D_eff=%.1f",
             direction.value, state.gbp, state.order_parameter, state.d_eff)
    return signal


def check_reorder_exits(state, open_positions: list, gbp_below_threshold_days: int = 0) -> list[dict]:
    """
    Check exit conditions for Reorder positions.

    Exit conditions:
      - GBP < 0.20 sustained 3+ days (grain fully established, trade complete)
      - Phase changes away from ordered
    """
    exits = []
    for pos in open_positions:
        if pos.signal_class != SignalClass.REORDER or not pos.is_open:
            continue

        reasons = []

        # GBP deeply inside new grain for 3+ days
        if state.gbp < REORDER_GBP_EXIT_THRESHOLD and gbp_below_threshold_days >= REORDER_GBP_EXIT_DAYS:
            reasons.append(
                f"Grain established: GBP={state.gbp:.3f} < {REORDER_GBP_EXIT_THRESHOLD} "
                f"for {gbp_below_threshold_days} days"
            )

        # Phase changed away from ordered
        if state.phase == MarketPhase.DISORDERED:
            reasons.append(f"Phase exit: REORDERING → DISORDERED")

        if reasons:
            exits.append({
                "position_id": pos.position_id,
                "signal_class": SignalClass.REORDER,
                "reasons": reasons,
                "urgency": "normal",
            })
            log.info("REORDER EXIT: %s | %s", pos.position_id[:8], "; ".join(reasons))

    return exits
