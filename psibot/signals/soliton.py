"""
signals/soliton.py — SK-12: Soliton Signal Generation
=======================================================
Soliton = topological momentum in ordered condensate.

From the article (Section 6):
  'Momentum: Topological soliton of expectation condensate.
   Propagates dispersion-free by topological protection.'
  'Momentum crashes are abrupt (soliton collapse at grain boundary), not gradual.'

Entry: condensate ordered, chirality defined, GBP low, L5 confirms.
Exit: GBP stop, D_eff stop, ψ turns bimodal, phase change.

Signal class: SK-12 (Soliton-Long / Soliton-Short)
CCDR Expectation Field Architecture — Version 1.0
"""

from __future__ import annotations

import logging
import uuid
from datetime import datetime
from typing import Optional

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from helpers import (
    BotMode, MarketPhase, PsiShape, AcousticSignal,
    SignalClass, SignalDirection, CCDR_THRESHOLDS,
)

_MOM_CRASH  = CCDR_THRESHOLDS["MOM_CRASH_THRESHOLD"]   # -0.20 (T3)
_MOM_NORMAL = CCDR_THRESHOLDS["MOM_NORMAL_THRESHOLD"]  #  0.05 (T3)

log = logging.getLogger("psibot.signals.soliton")

# Soliton entry thresholds
SOLITON_GBP_MAX = 0.30        # must be in deep grain
SOLITON_PHASES = {MarketPhase.ORDERED_BULL, MarketPhase.ORDERED_BEAR}
SOLITON_PSI_SHAPES = {PsiShape.SKEWED_RIGHT, PsiShape.SKEWED_LEFT}
SOLITON_INSTRUMENTS = ["SPX_FUT", "NDX_FUT"]  # primary execution instruments


def check_soliton_signal(state, portfolio) -> Optional[dict]:
    """
    Evaluate soliton entry conditions.

    Entry conditions (ALL required — from agents.md SK-12):
      □ active_mode == HUNTER
      □ gbp < 0.30
      □ phase in [ORDERED_BULL, ORDERED_BEAR]
      □ psi_shape in [SKEWED_RIGHT, SKEWED_LEFT]
      □ No existing Soliton in same direction

    Returns:
        Signal dict if conditions met, None otherwise
    """
    # Condition 1: must be in Hunter mode
    if state.active_mode != BotMode.HUNTER:
        return None

    # Condition 2: GBP must be low (deep grain interior)
    if state.gbp > SOLITON_GBP_MAX:
        return None

    # Condition 3: condensate must be in ordered phase
    if state.phase not in SOLITON_PHASES:
        return None

    # Condition 4: ψ_exp must show defined chirality
    if state.psi_shape not in SOLITON_PSI_SHAPES:
        return None

    # Determine direction from ψ chirality
    direction = (SignalDirection.LONG
                 if state.psi_shape == PsiShape.SKEWED_RIGHT
                 else SignalDirection.SHORT)

    # Condition 5: no existing Soliton in same direction
    for pos in portfolio.open_positions:
        if (pos.signal_class == SignalClass.SOLITON
                and pos.direction == direction):
            log.debug("Soliton: existing %s position — skipping", direction.value)
            return None

    # T3 momentum regime check — bimodal crash structure validated by Hartigan dip test.
    # Crash regime (<-20% 12m MOM): soliton topological protection fails abruptly → block.
    # Gap zone (-20% to +5%):       transitional uncertainty → halve size.
    mom_252d = getattr(state, "momentum_252d", 0.0)
    if mom_252d < _MOM_CRASH:
        log.warning(
            "Soliton BLOCKED: crash regime (mom252d=%.1f%% < %.0f%%) — soliton collapse risk",
            mom_252d * 100, _MOM_CRASH * 100,
        )
        return None

    # Adjust sizing for acoustic contradiction and T3 gap zone
    size_mult = state.signal_size_multiplier
    if mom_252d < _MOM_NORMAL:
        size_mult *= 0.50
        log.info(
            "Soliton: T3 gap-zone momentum (%.1f%%) — size halved to %.2f",
            mom_252d * 100, size_mult,
        )
    if state.acoustic_signal == AcousticSignal.CONTRADICT:
        size_mult *= 0.7
        log.info("Soliton: acoustic CONTRADICT — reducing size to %.2f", size_mult)
    elif state.acoustic_signal == AcousticSignal.CONFIRM:
        log.info("Soliton: acoustic CONFIRM — L5 aligned with ψ chirality")

    signal = {
        "signal_id": str(uuid.uuid4()),
        "signal_class": SignalClass.SOLITON,
        "direction": direction,
        "size_multiplier": size_mult,
        "entry_gbp": state.gbp,
        "entry_phase": state.phase,
        "entry_psi_shape": state.psi_shape,
        "entry_d_eff": state.d_eff,
        "instruments": SOLITON_INSTRUMENTS,
        "rationale": (
            f"Soliton-{direction.value}: "
            f"ψ={state.psi_shape.value}, "
            f"OP={state.order_parameter:.2f}, "
            f"GBP={state.gbp:.3f}, "
            f"D_eff={state.d_eff:.1f}, "
            f"acoustic={state.acoustic_signal.value}"
        ),
        "stop_conditions": {
            "gbp_delta": CCDR_THRESHOLDS["GBP_STOP_DELTA"],
            "d_eff_floor": CCDR_THRESHOLDS["D_EFF_STOP_FLOOR"],
            "phase_change": True,
            "psi_bimodal": True,
        },
        "timestamp": datetime.utcnow(),
    }

    log.info("SOLITON SIGNAL: %s | GBP=%.3f OP=%.2f D_eff=%.1f size×=%.2f",
             direction.value, state.gbp, state.order_parameter,
             state.d_eff, size_mult)
    return signal


def check_soliton_exits(state, open_positions: list) -> list[dict]:
    """
    Check structural stops for all open Soliton positions.
    Returns list of exit decisions.
    """
    exits = []
    for pos in open_positions:
        if pos.signal_class != SignalClass.SOLITON or not pos.is_open:
            continue

        reasons = []

        # GBP stop: grain boundary approach since entry
        if state.gbp > pos.entry_gbp + CCDR_THRESHOLDS["GBP_STOP_DELTA"]:
            reasons.append(f"GBP stop: {state.gbp:.3f} > {pos.entry_gbp:.3f}+{CCDR_THRESHOLDS['GBP_STOP_DELTA']}")

        # D_eff floor
        if state.d_eff < CCDR_THRESHOLDS["D_EFF_STOP_FLOOR"]:
            reasons.append(f"D_eff stop: {state.d_eff:.1f} < {CCDR_THRESHOLDS['D_EFF_STOP_FLOOR']}")

        # Phase transition: ordered → disordered
        if pos.entry_phase.is_ordered() and state.phase == MarketPhase.DISORDERED:
            reasons.append(f"Phase stop: {pos.entry_phase.value} → DISORDERED")

        # ψ bimodal stop: soliton dissolved into grain boundary crossing
        if state.psi_shape == PsiShape.BIMODAL:
            reasons.append("ψ bimodal stop: soliton collapsed at grain boundary")

        if reasons:
            exits.append({
                "position_id": pos.position_id,
                "signal_class": SignalClass.SOLITON,
                "reasons": reasons,
                "urgency": "normal",  # limit order preferred
            })
            log.warning("SOLITON EXIT TRIGGERED [%s]: %s",
                        pos.position_id[:8], "; ".join(reasons))

    return exits
