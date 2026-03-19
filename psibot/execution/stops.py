"""
execution/stops.py — SK-17: CCDR Structural Stop Engine
========================================================
These stops replace all price-level stop-losses entirely.
NEVER use a fixed price stop in this system.

From claude.md:
  'These stops replace all price-level stop-losses. Never use a fixed price stop.'

Stop types:
  1. GBP delta stop: exit when GBP rises above entry_GBP + 0.35
  2. D_eff floor stop: exit all risk positions when D_eff < 4.0
  3. Phase change stop: exit when phase transitions ordered → disordered
  4. ψ bimodal stop: exit soliton trades when ψ turns bimodal

CCDR Expectation Field Architecture — Version 1.0
"""

from __future__ import annotations

import logging

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from helpers import (
    MarketPhase, PsiShape, SignalClass, StopCheckResult,
    check_structural_stops, CCDR_THRESHOLDS,
)

log = logging.getLogger("psibot.execution.stops")


def evaluate_stops(state, portfolio) -> list[dict]:
    """
    Evaluate all CCDR structural stops for all open positions.
    Called every pipeline cycle by AGT-07 (Risk Agent).

    Returns:
        List of exit decisions: [{position_id, signal_class, reasons, urgency}, ...]
    """
    exit_decisions = []

    for position in portfolio.open_positions:
        result = check_structural_stops(
            entry_gbp=position.entry_gbp,
            entry_phase=position.entry_phase,
            signal_class=position.signal_class,
            current_gbp=state.gbp,
            current_d_eff=state.d_eff,
            current_phase=state.phase,
            current_psi_shape=state.psi_shape,
            gbp_stop_delta=CCDR_THRESHOLDS["GBP_STOP_DELTA"],
            d_eff_floor=CCDR_THRESHOLDS["D_EFF_STOP_FLOOR"],
        )

        if result.triggered:
            # Determine urgency: D_eff or GBP Guardian triggers → market order
            urgency = "normal"
            if state.d_eff <= CCDR_THRESHOLDS["D_EFF_GUARDIAN"]:
                urgency = "immediate"  # market order
            if state.gbp >= CCDR_THRESHOLDS["GBP_GUARDIAN"]:
                urgency = "immediate"  # market order

            exit_decisions.append({
                "position_id": position.position_id,
                "signal_class": position.signal_class,
                "reasons": result.reasons,
                "urgency": urgency,
            })

            log.warning("STRUCTURAL STOP [%s/%s urgency=%s]: %s",
                        position.position_id[:8],
                        position.signal_class.value,
                        urgency,
                        "; ".join(result.reasons))

    return exit_decisions


def pre_entry_stop_check(signal: dict, state) -> bool:
    """
    Pre-entry validation: ensure structural stop would not fire immediately on entry.
    Returns True if it's safe to enter (no immediate stop condition).
    """
    # GBP check: ensure we're well within grain
    if state.gbp > CCDR_THRESHOLDS["GBP_HUNTER_MAX"]:
        log.warning("Pre-entry stop check failed: GBP=%.3f too high for new position",
                    state.gbp)
        return False

    # D_eff floor check
    if state.d_eff < CCDR_THRESHOLDS["D_EFF_STOP_FLOOR"] + 1.0:
        log.warning("Pre-entry stop check failed: D_eff=%.1f too close to floor=%.1f",
                    state.d_eff, CCDR_THRESHOLDS["D_EFF_STOP_FLOOR"])
        return False

    # Phase check for Soliton
    if signal.get("signal_class") == SignalClass.SOLITON:
        if state.phase == MarketPhase.DISORDERED:
            log.warning("Pre-entry: Soliton blocked in DISORDERED phase")
            return False

    return True
