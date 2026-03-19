"""
signals/saturation_hedge.py — SK-15: Saturation Hedge Generation
=================================================================
Saturation-Hedge = defensive position before holographic saturation event.

From the article (Section 5):
  'When market complexity > S_attention:
   Agents cannot maintain independent, differentiated expectations.
   Expectation condensate undergoes dimensional reduction.
   All expectations collapse onto a single low-dimensional representation:
     risk-on vs. risk-off (the 2D attractor of market crises)'

Entry: D_eff declining rapidly OR fat-tailed ψ with low D_eff.
This signal can be entered even approaching Guardian mode.
In Guardian mode: existing hedge maintained; new hedge added at 50% sizing.

Signal class: SK-15
CCDR Expectation Field Architecture — Version 1.0
"""

from __future__ import annotations

import logging
import uuid
from datetime import datetime
from typing import Optional

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from helpers import BotMode, PsiShape, SignalClass, SignalDirection, CCDR_THRESHOLDS

log = logging.getLogger("psibot.signals.saturation_hedge")

# Saturation-Hedge entry thresholds (from agents.md SK-15)
SATURATION_HEDGE_INSTRUMENTS = [
    "VIX_CALL_30D",    # long VIX calls
    "SPX_PUT_10PCT",   # OTM SPX puts
    "GOLD",            # safe haven
    "TLT_FUT",         # long duration bonds
]


def check_saturation_hedge_signal(
    state,
    portfolio,
    guardian_active: bool = False,
) -> Optional[dict]:
    """
    Evaluate saturation hedge entry conditions.

    Entry conditions (ANY triggers entry):
      □ d_eff < 6.0 AND d_eff_trend_10d < -0.5 per day
      □ psi_shape == FAT_TAILED AND d_eff < 8.0

    Special rules:
      - Can be entered in Scout mode (defensive)
      - In Guardian mode: size at 50% normal
      - Max 2 saturation hedge positions at a time
    """
    # Not entered in Guardian mode (Guardian already flattened risk)
    # But Guardian activation GENERATES a saturation hedge (from guardian.py)
    if guardian_active and state.active_mode == BotMode.GUARDIAN:
        # Guardian generates saturation hedge directly; signal agent skips
        return None

    # Check entry conditions
    triggered = False
    trigger_reasons = []

    # Condition A: D_eff low and declining rapidly
    if state.d_eff < 6.0 and state.d_eff_trend_10d < -0.5:
        triggered = True
        trigger_reasons.append(
            f"D_eff={state.d_eff:.1f} < 6.0 AND trend={state.d_eff_trend_10d:.2f}/day < -0.5"
        )

    # Condition B: Fat-tailed ψ with modest D_eff
    if state.psi_shape == PsiShape.FAT_TAILED and state.d_eff < 8.0:
        triggered = True
        trigger_reasons.append(
            f"ψ=FAT_TAILED AND D_eff={state.d_eff:.1f} < 8.0"
        )

    if not triggered:
        return None

    # Cap at 2 saturation hedge positions
    existing = [p for p in portfolio.open_positions
                if p.signal_class == SignalClass.SATURATION_HEDGE]
    if len(existing) >= 2:
        return None

    # Size: 50% in Scout/approaching Guardian, normal in Hunter
    if state.active_mode == BotMode.SCOUT or state.d_eff < 5.0:
        size_mult = state.signal_size_multiplier * 0.5
    else:
        size_mult = state.signal_size_multiplier

    signal = {
        "signal_id": str(uuid.uuid4()),
        "signal_class": SignalClass.SATURATION_HEDGE,
        "direction": SignalDirection.LONG,   # long protection
        "size_multiplier": size_mult,
        "entry_gbp": state.gbp,
        "entry_phase": state.phase,
        "entry_psi_shape": state.psi_shape,
        "entry_d_eff": state.d_eff,
        "instruments": SATURATION_HEDGE_INSTRUMENTS,
        "rationale": "Saturation hedge: " + "; ".join(trigger_reasons),
        "exit_conditions": {
            "d_eff_recovery": 8.0,          # exit when D_eff recovers > 8.0
            "psi_gaussian": True,           # exit when ψ returns to GAUSSIAN
        },
        "timestamp": datetime.utcnow(),
    }

    log.info("SATURATION HEDGE SIGNAL | D_eff=%.1f trend=%.2f/day ψ=%s size×=%.2f",
             state.d_eff, state.d_eff_trend_10d, state.psi_shape.value, size_mult)
    return signal


def check_saturation_hedge_exits(state, open_positions: list) -> list[dict]:
    """
    Exit saturation hedge when crisis conditions have passed.
    """
    exits = []
    for pos in open_positions:
        if pos.signal_class != SignalClass.SATURATION_HEDGE or not pos.is_open:
            continue

        reasons = []

        # D_eff recovered above 8.0 and stabilised
        if state.d_eff > 8.0 and state.d_eff_trend_10d >= 0:
            reasons.append(f"D_eff recovery: {state.d_eff:.1f} > 8.0, trend stable")

        # ψ returned to Gaussian (normal condensate)
        if state.psi_shape == PsiShape.GAUSSIAN:
            reasons.append("ψ returned to GAUSSIAN — condensate normalised")

        if reasons:
            exits.append({
                "position_id": pos.position_id,
                "signal_class": SignalClass.SATURATION_HEDGE,
                "reasons": reasons,
                "urgency": "normal",
            })
            log.info("SATURATION HEDGE EXIT: %s | %s",
                     pos.position_id[:8], "; ".join(reasons))

    return exits
