"""
modes/hunter.py — SK-19: Hunter Mode Controller
================================================
Hunter mode: active signal execution.

Hunter conditions (ALL must hold):
  - GBP < 0.5 (deep within grain)
  - D_eff > 5.0 (sufficient dimensionality)
  - Phase != DISORDERED (condensate ordered)

In Hunter mode, all four signal classes are available:
  - Soliton: ordered phase + defined chirality + GBP < 0.3
  - Transition: GBP > 0.65 + DP accelerating (defensive, still active near boundary)
  - Reorder: new grain nucleating + ψ=GAUSSIAN
  - Saturation Hedge: D_eff declining or ψ fat-tailed

CCDR Expectation Field Architecture — Version 1.0
"""

from __future__ import annotations

import logging
from datetime import datetime

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from helpers import BotMode, MarketPhase, PsiShape, SignalClass, CCDR_THRESHOLDS

log = logging.getLogger("psibot.modes.hunter")


class HunterModeController:
    """
    Hunter mode: full signal generation and execution.
    """

    def is_hunter_condition(self, state) -> bool:
        """Verify all Hunter mode conditions are met (programmatic enforcement)."""
        return (
            state.gbp < CCDR_THRESHOLDS["GBP_HUNTER_MAX"]
            and state.d_eff > CCDR_THRESHOLDS["D_EFF_SCOUT"]
            and state.phase != MarketPhase.DISORDERED
            and not state.l1_failed
            and not state.l3_failed
        )

    def evaluate_all_signals(self, state, portfolio) -> list[dict]:
        """
        Evaluate all four signal classes in Hunter mode.
        Returns list of signal dicts ready for execution.
        """
        from psibot.signals.soliton import check_soliton_signal
        from psibot.signals.transition import check_transition_signal
        from psibot.signals.reorder import check_reorder_signal
        from psibot.signals.saturation_hedge import check_saturation_hedge_signal

        # Guard: enforce Hunter conditions before generating any signal
        if not self.is_hunter_condition(state):
            log.warning("Hunter condition check failed — suppressing signals "
                        "(GBP=%.3f D_eff=%.1f phase=%s)",
                        state.gbp, state.d_eff, state.phase.value)
            return []

        signals = []

        # 1. Soliton (topological momentum — primary Hunter signal)
        sig = check_soliton_signal(state, portfolio)
        if sig:
            signals.append(sig)

        # 2. Transition (grain boundary — defensive)
        sig = check_transition_signal(state, portfolio)
        if sig:
            signals.append(sig)

        # 3. Reorder (new grain first mover)
        sig = check_reorder_signal(state, portfolio)
        if sig:
            signals.append(sig)

        # 4. Saturation Hedge (D_eff declining defense)
        sig = check_saturation_hedge_signal(state, portfolio)
        if sig:
            signals.append(sig)

        if signals:
            log.info("Hunter generated %d signal(s): %s",
                     len(signals),
                     [s["signal_class"].value for s in signals])

        return signals

    def evaluate_all_exits(self, state, portfolio, **context) -> list[dict]:
        """Evaluate all structural stops across open positions."""
        from psibot.signals.soliton import check_soliton_exits
        from psibot.signals.transition import check_transition_exits
        from psibot.signals.reorder import check_reorder_exits
        from psibot.signals.saturation_hedge import check_saturation_hedge_exits

        all_exits = []
        positions = portfolio.open_positions

        all_exits.extend(check_soliton_exits(state, positions))
        all_exits.extend(check_transition_exits(
            state, positions,
            op_rising_days=context.get("op_rising_days", 0)
        ))
        all_exits.extend(check_reorder_exits(
            state, positions,
            gbp_below_threshold_days=context.get("gbp_below_threshold_days", 0)
        ))
        all_exits.extend(check_saturation_hedge_exits(state, positions))

        return all_exits

    def position_capacity_check(self, portfolio) -> dict:
        """Check position capacity before generating new signals."""
        open_count = portfolio.position_count
        max_positions = CCDR_THRESHOLDS["MAX_POSITIONS"]
        remaining = max_positions - open_count
        return {
            "open_positions": open_count,
            "max_positions": max_positions,
            "remaining_capacity": remaining,
            "can_open_new": remaining > 0,
        }
