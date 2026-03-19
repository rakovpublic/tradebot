"""
modes/scout.py — SK-18: Scout Mode Controller
==============================================
Scout mode: observe only — no new risk positions.

Scout is the DEFAULT mode when:
  - D_eff is in [3, 5] range (post-crisis or pre-Hunter)
  - Phase is DISORDERED with GBP > 0.6
  - System is uncertain about condensate state
  - Recovering from Guardian mode

In Scout mode:
  - Pipeline runs fully (all 5 layers)
  - No new Soliton or Reorder positions
  - Transition signals allowed (defensive)
  - Saturation Hedge allowed (defensive)
  - Existing positions: structural stops monitored but not force-closed

CCDR Expectation Field Architecture — Version 1.0
"""

from __future__ import annotations

import logging
from datetime import datetime

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from helpers import BotMode, MarketPhase, PsiShape, SignalClass, CCDR_THRESHOLDS

log = logging.getLogger("psibot.modes.scout")


class ScoutModeController:
    """
    Scout mode: full pipeline observation, no new risk positions.
    Promotes to Hunter when conditions clear.
    """

    def is_scout_condition(self, state) -> bool:
        """Check if Scout conditions are met (non-Guardian, non-Hunter)."""
        return state.active_mode == BotMode.SCOUT

    def allowed_signals(self, state) -> list[str]:
        """
        Scout mode allows only defensive signals.
        Returns list of allowed SignalClass values.
        """
        allowed = []

        # Transition (long vol): always allowed in Scout — defensive
        if state.gbp > 0.65 and state.dp_trend_10d > 0:
            allowed.append(SignalClass.TRANSITION.value)

        # Saturation hedge: allowed when D_eff declining
        if state.d_eff < 6.0 and state.d_eff_trend_10d < -0.3:
            allowed.append(SignalClass.SATURATION_HEDGE.value)

        return allowed

    def can_promote_to_hunter(self, state) -> bool:
        """
        Check if conditions have improved enough to switch to Hunter mode.
        Promotion requires strict conditions — err on side of caution.
        """
        return (
            state.d_eff > CCDR_THRESHOLDS["D_EFF_SCOUT"]
            and state.gbp < CCDR_THRESHOLDS["GBP_HUNTER_MAX"]
            and state.phase != MarketPhase.DISORDERED
            and not state.l1_failed
            and not state.l3_failed
        )

    def generate_scout_report(self, state) -> dict:
        """Generate observation report for monitoring."""
        promotion_ready = self.can_promote_to_hunter(state)
        return {
            "mode": BotMode.SCOUT.value,
            "timestamp": datetime.utcnow().isoformat(),
            "d_eff": state.d_eff,
            "gbp": state.gbp,
            "phase": state.phase.value,
            "psi_shape": state.psi_shape.value,
            "promotion_ready": promotion_ready,
            "allowed_signals": self.allowed_signals(state),
            "observation": _describe_scout_observation(state),
        }


def _describe_scout_observation(state) -> str:
    """Human-readable description of current Scout mode observation."""
    lines = []

    if state.d_eff <= 5.0:
        lines.append(f"D_eff={state.d_eff:.1f} — post-crisis or approaching saturation")
    if state.gbp > 0.6:
        lines.append(f"GBP={state.gbp:.3f} — grain boundary approaching")
    if state.phase == MarketPhase.DISORDERED:
        lines.append("Phase=DISORDERED — condensate in transition")
    if state.psi_shape == PsiShape.BIMODAL:
        lines.append("ψ=BIMODAL — grain boundary crossing in progress")
    if state.psi_shape == PsiShape.FAT_TAILED:
        lines.append("ψ=FAT_TAILED — tails widening, approaching saturation")

    if not lines:
        lines.append("Conditions uncertain — observing")

    return "; ".join(lines)
