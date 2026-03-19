"""
execution/sizing.py — SK-16: Position Sizing Engine
====================================================
Position sizing is a function of D_eff and GBP, modified by L5 acoustic signal.

Formula:
  base = f(D_eff) × f(GBP) × max_risk_usd
  if acoustic=CONFIRM:   final = min(base × 1.2, max_risk_usd)
  if acoustic=CONTRADICT: final = base × 0.7
  if acoustic=NEUTRAL:   final = base

Constraints:
  - Never exceed max_leverage × account_equity on a single position
  - Never exceed remaining portfolio capacity

CCDR Expectation Field Architecture — Version 1.0
"""

from __future__ import annotations

import logging

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from helpers import (
    AcousticSignal, compute_position_size,
    d_eff_to_size_factor, gbp_to_size_factor, CCDR_THRESHOLDS,
)

log = logging.getLogger("psibot.execution.sizing")


def size_order(signal: dict, state, portfolio, config: dict = None) -> float:
    """
    Compute final position size in USD notional.

    Args:
        signal: Signal dict from signal agent
        state: CondensateState
        portfolio: PortfolioState
        config: Override config dict (optional)

    Returns:
        Position size in USD notional (0.0 if blocked)
    """
    if config is None:
        config = {}

    max_risk_usd = config.get("max_risk_usd", portfolio.max_risk_usd)
    max_leverage = config.get("max_leverage", CCDR_THRESHOLDS["MAX_LEVERAGE"])
    max_positions = config.get("max_positions", CCDR_THRESHOLDS["MAX_POSITIONS"])

    # Check portfolio capacity
    remaining_capacity = max_positions - portfolio.position_count
    if remaining_capacity <= 0:
        log.warning("Max positions (%d) reached — blocking new signal", max_positions)
        return 0.0

    # Compute base size using CCDR formula
    base_usd = compute_position_size(
        d_eff=state.d_eff,
        gbp=state.gbp,
        acoustic=state.acoustic_signal,
        max_risk_usd=max_risk_usd,
    )

    # Apply signal-specific size multiplier
    signal_mult = signal.get("size_multiplier", 1.0)
    final_usd = base_usd * signal_mult

    # Never exceed 2× leverage on single position
    max_notional = portfolio.account_equity * max_leverage
    final_usd = min(final_usd, max_notional)

    # Never exceed max_risk_usd
    final_usd = min(final_usd, max_risk_usd)

    if final_usd <= 0:
        log.warning("Sizing: computed size=0 (D_eff=%.1f GBP=%.3f) — signal blocked",
                    state.d_eff, state.gbp)
        return 0.0

    log.info("Sizing: base=%.0f × signal_mult=%.2f = %.0f USD "
             "(D_eff=%.1f GBP=%.3f acoustic=%s)",
             base_usd, signal_mult, final_usd,
             state.d_eff, state.gbp, state.acoustic_signal.value)

    return float(final_usd)


def compute_size_multiplier(state) -> float:
    """
    Compute the signal_size_multiplier field for CondensateState.
    Called at end of pipeline to store in state object.
    """
    from helpers import compute_position_size
    # Compute as fraction of max_risk_usd (normalised 0-1)
    f_deff = d_eff_to_size_factor(state.d_eff)
    f_gbp = gbp_to_size_factor(state.gbp)
    base = f_deff * f_gbp

    if state.acoustic_signal == AcousticSignal.CONFIRM:
        return min(base * 1.2, 1.0)
    elif state.acoustic_signal == AcousticSignal.CONTRADICT:
        return base * 0.7
    return base
