"""
modes/guardian.py — SK-20: Guardian Mode Controller (AGT-07)
=============================================================
Guardian mode is the HIGHEST PRIORITY agent. Activated on systemic risk events.
It runs in parallel with the pipeline and can interrupt any other agent.

Guardian mode actions:
  1. Flatten all Soliton and Reorder positions (market orders)
  2. Maintain Transition positions (they hedge Guardian risk)
  3. Add Saturation-Hedge at 50% sizing
  4. Block new Hunter/Scout risk positions

Guardian exit requires 48h cooloff + 5 consecutive days of clean conditions.

From the article (Section 5):
  'D_eff decreasing over time → approaching saturation → crash risk rising'
  'All expectations collapse onto risk-on vs risk-off (2D attractor of market crises)'

CCDR Expectation Field Architecture — Version 1.0
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta
from typing import Optional

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from helpers import BotMode, MarketPhase, SignalClass, CCDR_THRESHOLDS

log = logging.getLogger("psibot.modes.guardian")

GUARDIAN_COOLOFF_HOURS = CCDR_THRESHOLDS["GUARDIAN_COOLOFF_H"]
GUARDIAN_CLEAN_DAYS_REQUIRED = 5


class GuardianModeController:
    """
    AGT-07: Risk Agent — Guardian Mode Controller.

    Always active; runs in parallel with main pipeline.
    Can interrupt any other operation via event bus.
    """

    def __init__(self):
        self.guardian_active: bool = False
        self.guardian_activated_at: Optional[datetime] = None
        self.guardian_reason: str = ""
        self.consecutive_clean_days: int = 0
        self.block_new_positions: bool = False

    def check_guardian_triggers(self, state, portfolio) -> bool:
        """
        Check all hard Guardian trigger conditions.

        Triggers:
          - D_eff <= 3.0 (holographic saturation)
          - GBP >= 0.8 (grain boundary crossing)
          - 10-day rolling drawdown <= -5%
        """
        # Hard D_eff trigger
        if state.d_eff <= CCDR_THRESHOLDS["D_EFF_GUARDIAN"]:
            self.activate_guardian(
                state, portfolio,
                f"D_eff below {CCDR_THRESHOLDS['D_EFF_GUARDIAN']} — holographic saturation"
            )
            return True

        # Hard GBP trigger
        if state.gbp >= CCDR_THRESHOLDS["GBP_GUARDIAN"]:
            self.activate_guardian(
                state, portfolio,
                f"GBP >= {CCDR_THRESHOLDS['GBP_GUARDIAN']} — grain boundary crossing in progress"
            )
            return True

        # Drawdown circuit breaker
        drawdown_10d = portfolio.rolling_drawdown(days=10)
        if drawdown_10d <= -CCDR_THRESHOLDS["DRAWDOWN_CB"]:
            self.activate_guardian(
                state, portfolio,
                f"Drawdown circuit breaker: {drawdown_10d:.1%} in rolling 10 days"
            )
            return True

        return False

    def activate_guardian(self, state, portfolio, reason: str) -> None:
        """
        Guardian Mode Activation Protocol:
          1. Log and alert
          2. Mark positions for closure (Soliton + Reorder = market orders)
          3. Maintain Transition positions
          4. Generate Saturation-Hedge at 50% size
          5. Block all new non-defensive positions
        """
        if self.guardian_active:
            log.warning("Guardian already active; new trigger: %s", reason)
            return

        log.critical("=" * 60)
        log.critical("GUARDIAN MODE ACTIVATED: %s", reason)
        log.critical("D_eff=%.2f GBP=%.3f Phase=%s ψ=%s",
                     state.d_eff, state.gbp, state.phase.value, state.psi_shape.value)
        log.critical("=" * 60)

        self.guardian_active = True
        self.guardian_activated_at = datetime.utcnow()
        self.guardian_reason = reason
        self.consecutive_clean_days = 0
        self.block_new_positions = True

        # Mark risk positions for immediate close (execution agent handles order routing)
        positions_to_close = [
            p for p in portfolio.open_positions
            if p.signal_class in (SignalClass.SOLITON, SignalClass.REORDER)
        ]
        for pos in positions_to_close:
            log.critical("Guardian: marking %s %s for immediate closure",
                         pos.signal_class.value, pos.position_id[:8])
            # In production: execution_agent.close_position(pos, order_type="MARKET",
            #                 reason=f"Guardian: {reason}")

        # Transition positions are maintained (they profit from the transition)
        transition_positions = [
            p for p in portfolio.open_positions
            if p.signal_class == SignalClass.TRANSITION
        ]
        if transition_positions:
            log.info("Guardian: maintaining %d Transition position(s) — these hedge Guardian risk",
                     len(transition_positions))

        # Emit alert (monitoring integration point)
        self._emit_alert(
            f"GUARDIAN MODE ACTIVATED: {reason}",
            priority="CRITICAL",
            context={
                "d_eff": state.d_eff,
                "gbp": state.gbp,
                "positions_closed": len(positions_to_close),
            }
        )

    def check_guardian_exit(self, state) -> bool:
        """
        Check if Guardian mode can be deactivated.

        Exit conditions (ALL must be met for 5 consecutive business days):
          - D_eff > 6.0
          - GBP < 0.4
          - Phase is Ordered

        Plus: minimum 48-hour cooloff since activation.
        """
        if not self.guardian_active:
            return False

        # Enforce minimum cooloff
        elapsed = datetime.utcnow() - self.guardian_activated_at
        if elapsed < timedelta(hours=GUARDIAN_COOLOFF_HOURS):
            remaining_hours = GUARDIAN_COOLOFF_HOURS - elapsed.total_seconds() / 3600
            log.debug("Guardian cooloff: %.1f hours remaining", remaining_hours)
            return False

        # Check all clean conditions simultaneously
        conditions_met = (
            state.d_eff > 6.0
            and state.gbp < 0.4
            and state.phase.is_ordered()
        )

        if conditions_met:
            self.consecutive_clean_days += 1
            log.info("Guardian exit check: clean day %d/%d (D_eff=%.1f GBP=%.3f phase=%s)",
                     self.consecutive_clean_days, GUARDIAN_CLEAN_DAYS_REQUIRED,
                     state.d_eff, state.gbp, state.phase.value)
        else:
            if self.consecutive_clean_days > 0:
                log.info("Guardian exit: clean streak broken at %d days — resetting",
                         self.consecutive_clean_days)
            self.consecutive_clean_days = 0

        if self.consecutive_clean_days >= GUARDIAN_CLEAN_DAYS_REQUIRED:
            self.deactivate_guardian(state)
            return True

        return False

    def deactivate_guardian(self, state) -> None:
        """Deactivate Guardian mode — transition back to Scout."""
        log.warning("GUARDIAN MODE DEACTIVATED — conditions clean for %d days",
                    self.consecutive_clean_days)
        log.warning("Returning to Scout mode for manual review before Hunter permitted")

        self.guardian_active = False
        self.guardian_activated_at = None
        self.guardian_reason = ""
        self.consecutive_clean_days = 0
        self.block_new_positions = False
        # Note: system returns to Scout; Hunter requires human review of conditions

        self._emit_alert(
            "Guardian Mode DEACTIVATED — returning to Scout",
            priority="WARNING",
            context={"d_eff": state.d_eff, "gbp": state.gbp},
        )

    def check_structural_stops_all_positions(self, state, portfolio) -> list[dict]:
        """
        AGT-07 runs structural stop check for every open position every pipeline cycle.
        Returns list of exit decisions for execution agent.
        """
        from helpers import check_structural_stops

        exit_decisions = []
        for position in portfolio.open_positions:
            stop_result = check_structural_stops(
                entry_gbp=position.entry_gbp,
                entry_phase=position.entry_phase,
                signal_class=position.signal_class,
                current_gbp=state.gbp,
                current_d_eff=state.d_eff,
                current_phase=state.phase,
                current_psi_shape=state.psi_shape,
            )
            if stop_result.triggered:
                exit_decisions.append({
                    "position_id": position.position_id,
                    "signal_class": position.signal_class,
                    "reasons": stop_result.reasons,
                    "urgency": "normal",
                })
                log.warning("Structural stop triggered [%s]: %s",
                            position.position_id[:8], "; ".join(stop_result.reasons))

        return exit_decisions

    def check_portfolio_d_eff(self, portfolio) -> bool:
        """
        Monitor portfolio's own D_eff — ensure portfolio isn't becoming
        its own holographic saturation event.

        Returns True if portfolio is too concentrated (block new positions).
        """
        from helpers import compute_d_eff

        if portfolio.position_count < 3:
            return False  # not enough positions to compute

        pnl_matrix = portfolio.get_pnl_matrix(days=30)
        if pnl_matrix.size == 0:
            return False

        portfolio_d_eff = compute_d_eff(pnl_matrix)

        if portfolio_d_eff < 4.0:
            log.warning("Portfolio D_eff = %.1f — too concentrated, blocking new positions",
                        portfolio_d_eff)
            self.block_new_positions = True
            return True

        if self.block_new_positions and portfolio_d_eff > 5.0:
            log.info("Portfolio D_eff recovered to %.1f — releasing position block", portfolio_d_eff)
            self.block_new_positions = False

        return False

    def _emit_alert(self, message: str, priority: str, context: dict = None) -> None:
        """Emit monitoring alert (Slack/email/PagerDuty integration point)."""
        log.log(
            logging.CRITICAL if priority == "CRITICAL" else logging.WARNING,
            "ALERT [%s]: %s | context=%s",
            priority, message, context or {}
        )
        # Production integration:
        # monitor_agent.alert(message, priority=priority, context=context)

    @property
    def status(self) -> dict:
        return {
            "guardian_active": self.guardian_active,
            "activated_at": self.guardian_activated_at.isoformat() if self.guardian_activated_at else None,
            "reason": self.guardian_reason,
            "consecutive_clean_days": self.consecutive_clean_days,
            "block_new_positions": self.block_new_positions,
        }
