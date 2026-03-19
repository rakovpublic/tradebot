"""
signals/transition.py — SK-13: Transition Signal Generation
============================================================
Transition = long volatility at grain boundary crossing.

From the article (Section 4.1):
  'The transition between bull and bear is a genuine topological phase
   transition: the order parameter passes through zero (disordered phase)
   before re-establishing in the new direction.'

Entry: GBP high (grain boundary imminent), DP accelerating, D_eff declining.
Exit: re-ordering signal (|OP| rising), GBP falls below 0.4.

Key property: Transition positions are MAINTAINED during Guardian mode
(volatility positions hedge Guardian risk — they profit from the transition).

Signal class: SK-13
CCDR Expectation Field Architecture — Version 1.0
"""

from __future__ import annotations

import logging
import uuid
from datetime import datetime
from typing import Optional

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from helpers import BotMode, MarketPhase, PsiShape, SignalClass, SignalDirection, CCDR_THRESHOLDS

log = logging.getLogger("psibot.signals.transition")

# Transition entry thresholds (from agents.md SK-13)
TRANSITION_GBP_MIN = 0.65
TRANSITION_DP_TREND_PERCENTILE = 0.90   # 90th percentile threshold
TRANSITION_DEFF_TREND_MAX = -0.3        # declining D_eff required
TRANSITION_INSTRUMENTS = ["VIX_FUT", "VVIX_FUT", "SPX_STRADDLE"]  # vol positions


def check_transition_signal(state, portfolio, dp_trend_90th_pctile: float = 0.05) -> Optional[dict]:
    """
    Evaluate transition entry conditions.

    Entry conditions (ALL required):
      □ active_mode in [HUNTER, SCOUT with dp_trend extreme]
      □ gbp > 0.65
      □ dp_trend_10d > 90th percentile (dispersion accelerating)
      □ d_eff_trend_10d < -0.3 (dimensionality declining)

    The transition signal is special: it can be entered in Scout mode
    if DP trend is extreme, because it's a defensive volatility position.
    """
    # Check mode eligibility
    if state.active_mode == BotMode.GUARDIAN:
        # In Guardian: no new positions, but maintain existing
        return None

    # GBP must be elevated — grain boundary imminent
    if state.gbp < TRANSITION_GBP_MIN:
        return None

    # DP trend must be accelerating (dispersion explosion)
    if state.dp_trend_10d < dp_trend_90th_pctile:
        return None

    # D_eff must be declining
    if state.d_eff_trend_10d >= TRANSITION_DEFF_TREND_MAX:
        return None

    # Cap to 1 Transition position at a time
    existing_transitions = [p for p in portfolio.open_positions
                            if p.signal_class == SignalClass.TRANSITION]
    if len(existing_transitions) >= 1:
        log.debug("Transition: existing position — skipping")
        return None

    # Transition is always directionally long volatility (LONG = long vol)
    direction = SignalDirection.LONG

    signal = {
        "signal_id": str(uuid.uuid4()),
        "signal_class": SignalClass.TRANSITION,
        "direction": direction,
        "size_multiplier": state.signal_size_multiplier * 0.8,  # slightly smaller for vol
        "entry_gbp": state.gbp,
        "entry_phase": state.phase,
        "entry_psi_shape": state.psi_shape,
        "entry_d_eff": state.d_eff,
        "instruments": TRANSITION_INSTRUMENTS,
        "rationale": (
            f"Transition (long vol): GBP={state.gbp:.3f}, "
            f"dp_trend_10d={state.dp_trend_10d:.4f}, "
            f"d_eff_trend_10d={state.d_eff_trend_10d:.3f}/day, "
            f"D_eff={state.d_eff:.1f}"
        ),
        "exit_conditions": {
            "gbp_fall_threshold": 0.40,             # exit when GBP < 0.4 (boundary passed)
            "op_reordering_days": 3,                # |OP| rising for 3+ consecutive days
        },
        "maintained_in_guardian": True,             # Transition survives Guardian mode
        "timestamp": datetime.utcnow(),
    }

    log.info("TRANSITION SIGNAL: long vol | GBP=%.3f dp_trend=%.4f D_eff_trend=%.3f/day",
             state.gbp, state.dp_trend_10d, state.d_eff_trend_10d)
    return signal


def check_transition_exits(state, open_positions: list, op_rising_days: int = 0) -> list[dict]:
    """
    Check exit conditions for Transition (long vol) positions.

    Exit conditions:
      - |OP| rising from trough for 3+ consecutive days (re-ordering)
      - GBP falls below 0.40 (boundary crossing completed)
    """
    exits = []
    for pos in open_positions:
        if pos.signal_class != SignalClass.TRANSITION or not pos.is_open:
            continue

        reasons = []

        # L2 re-ordering: |OP| rising from trough 3+ days
        if op_rising_days >= 3:
            reasons.append(f"Re-ordering: |OP| rising for {op_rising_days} days")

        # GBP normalised — boundary crossing complete
        if state.gbp < 0.40:
            reasons.append(f"GBP recovery: {state.gbp:.3f} < 0.40 (boundary crossed)")

        if reasons:
            exits.append({
                "position_id": pos.position_id,
                "signal_class": SignalClass.TRANSITION,
                "reasons": reasons,
                "urgency": "normal",
            })
            log.info("TRANSITION EXIT: %s | %s", pos.position_id[:8], "; ".join(reasons))

    return exits
