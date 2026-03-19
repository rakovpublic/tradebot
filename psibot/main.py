"""
main.py — ΨBot Orchestrator (AGT-00)
=====================================
The master coordinator. Runs the CCDR pipeline in sequence,
manages mode state, and routes results between agents.

This is the system entry point. Run with:
  python -m psibot.main --mode full --no-broker --paper-trading
  python -m psibot.main --mode scout --layers 1,3 --no-broker   (Phase 1)
  python -m psibot.backtest --hypotheses all --start 2000-01-01  (Phase 0)

Pipeline execution (every 5 minutes intraday, every EOD):
  L1 → L2 → L3 → L4 → L5 → mode determination → signal evaluation → execution

Critical rules:
  - L1 must complete before L2, L2 before L3, etc. NEVER run in parallel.
  - Guardian (AGT-07) checks are performed FIRST, LAST, and in between.
  - Pipeline checkpoints to disk every 5 minutes.

CCDR Expectation Field Architecture — Version 1.0
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import sys
from datetime import datetime
from typing import Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import structlog

from helpers import (
    BotMode, determine_bot_mode, format_state_summary,
    d_eff_to_label, gbp_to_label,
)
from psibot.state.condensate_state import CondensateState
from psibot.state.portfolio_state import PortfolioState
from psibot.modes.guardian import GuardianModeController
from psibot.modes.hunter import HunterModeController
from psibot.modes.scout import ScoutModeController
from psibot.execution.sizing import compute_size_multiplier
from psibot.execution.stops import evaluate_stops
from psibot.data.options_feed import OptionsFeed
from psibot.data.analyst_feed import AnalystFeed
from psibot.data.dark_pool_feed import DarkPoolFeed
from psibot.data.cross_asset_feed import CrossAssetFeed
from psibot.pipeline import (
    l1_psi_reconstruction,
    l2_phase_detector,
    l3_holo_monitor,
    l4_grain_boundary,
    l5_acoustic_parser,
)

# Try structlog, fall back to stdlib
try:
    log = structlog.get_logger("psibot.orchestrator")
except Exception:
    log = logging.getLogger("psibot.orchestrator")


class PsiBotOrchestrator:
    """
    AGT-00: Master Orchestrator.
    Coordinates the pipeline, manages mode transitions, routes to agents.
    """

    def __init__(self, config: dict = None):
        self.config = config or {}
        self.paper_mode = self.config.get("paper_trading", True)

        # Data feeds
        options_cfg = self.config.get("data", {}).get("options", {})
        analyst_cfg = self.config.get("data", {}).get("analyst", {})
        dark_pool_cfg = self.config.get("data", {}).get("dark_pool", {})
        cross_asset_cfg = self.config.get("data", {}).get("cross_asset", {})

        self.options_feed = OptionsFeed(
            provider=options_cfg.get("provider", "csv"),
            config=options_cfg,
        )
        self.analyst_feed = AnalystFeed(
            provider=analyst_cfg.get("provider", "csv"),
            config=analyst_cfg,
        )
        self.dark_pool_feed = DarkPoolFeed(
            provider=dark_pool_cfg.get("provider", "csv"),
            config=dark_pool_cfg,
        )
        self.cross_asset_feed = CrossAssetFeed(
            provider=cross_asset_cfg.get("provider", "csv"),
            config=cross_asset_cfg,
        )

        # Mode controllers
        self.guardian = GuardianModeController()
        self.hunter = HunterModeController()
        self.scout = ScoutModeController()

        # Portfolio state
        exec_cfg = self.config.get("execution", {})
        self.portfolio = PortfolioState(
            account_equity=self.config.get("initial_equity", 100_000.0),
            max_risk_usd=exec_cfg.get("max_risk_usd", 10_000.0),
        )

        # State tracking
        self._cycle_count = 0
        self._last_l2_run: Optional[datetime] = None
        self._op_rising_days = 0
        self._gbp_low_days = 0

        # Snapshot directory
        self.snapshot_dir = self.config.get("monitoring", {}).get(
            "snapshot_dir", "state/snapshots"
        )
        os.makedirs(self.snapshot_dir, exist_ok=True)

    async def run_cycle(self, trigger: str = "scheduled") -> CondensateState:
        """
        Execute one full pipeline cycle.

        Pipeline order (CRITICAL: sequential, never parallel):
          1. Pre-check: Guardian risk assessment
          2. L1: ψ_exp wavefunction from options surface
          3. L2: Condensate phase from analyst/survey data
          4. L3: D_eff holographic saturation gauge
          5. L4: GBP grain boundary proximity synthesis
          6. L5: Acoustic confirmation (prices/volume)
          7. Mode determination + position sizing
          8. Signal evaluation (Hunter) or observation (Scout/Guardian)
          9. Structural stop checks
          10. Portfolio D_eff monitor
          11. State snapshot

        Returns:
            CondensateState: final state after all layers
        """
        self._cycle_count += 1
        state = CondensateState(timestamp=datetime.utcnow())
        log.info("Pipeline cycle #%d starting (trigger=%s)", self._cycle_count, trigger)

        # -------------------------
        # Step 0: Pre-check risk
        # -------------------------
        self.guardian.check_guardian_triggers(state, self.portfolio)

        # -------------------------
        # Step 1: L1 — ψ_exp (PRIMARY SENSOR)
        # -------------------------
        try:
            symbol = self.config.get("data", {}).get("options", {}).get("symbols", ["SPX"])[0]
            options_surface = await self.options_feed.get_surface(symbol)
            state = await l1_psi_reconstruction.run(state, options_surface)
        except Exception as e:
            log.error("L1 exception (unhandled): %s", e)
            state.l1_failed = True

        # -------------------------
        # Step 2: L2 — Condensate Phase (Daily + on update)
        # -------------------------
        try:
            analyst_data = await self.analyst_feed.get_analyst_data("SPX_INDEX")
            survey_data = await self.analyst_feed.get_survey_data()
            state = await l2_phase_detector.run(state, analyst_data, survey_data)
        except Exception as e:
            log.error("L2 exception: %s", e)
            state.l2_failed = True

        # -------------------------
        # Step 3: L3 — D_eff (CRITICAL RISK GAUGE)
        # -------------------------
        try:
            cross_asset_data = await self.cross_asset_feed.get_returns_matrix(
                window_days=self.config.get("pipeline", {}).get("d_eff_rolling_window_days", 60)
            )
            state = await l3_holo_monitor.run(state, cross_asset_data)
        except Exception as e:
            log.error("L3 exception: %s", e)
            state.l3_failed = True

        # -------------------------
        # Step 4: L4 — GBP (MOST IMPORTANT NUMBER)
        # -------------------------
        try:
            symbol = self.config.get("data", {}).get("options", {}).get("symbols", ["SPX"])[0]
            dp_data = await self.dark_pool_feed.get_dark_pool_data(symbol)
            dark_pool_ratio = dp_data.dark_pool_ratio if dp_data else 1.0
            state = await l4_grain_boundary.run(state, dark_pool_ratio)
        except Exception as e:
            log.error("L4 exception: %s", e)
            state.l4_failed = True

        # -------------------------
        # Step 5: L5 — Acoustic Confirmation (LOWEST PRIORITY)
        # -------------------------
        try:
            state = await l5_acoustic_parser.run(state, cross_asset_data if 'cross_asset_data' in dir() else None)
        except Exception as e:
            log.error("L5 exception: %s", e)
            state.l5_failed = True

        # -------------------------
        # Step 6: Mode determination + sizing
        # -------------------------
        state.active_mode = determine_bot_mode(state.d_eff, state.gbp, state.phase)

        # Guardian override
        if self.guardian.guardian_active:
            state.active_mode = BotMode.GUARDIAN

        state.signal_size_multiplier = compute_size_multiplier(state)

        log.info(format_state_summary(state))
        log.info("D_eff: %s | GBP: %s",
                 d_eff_to_label(state.d_eff), gbp_to_label(state.gbp))

        # -------------------------
        # Step 7: Post-pipeline Guardian check
        # -------------------------
        self.guardian.check_guardian_triggers(state, self.portfolio)
        self.guardian.check_guardian_exit(state)

        # -------------------------
        # Step 8: Signal evaluation
        # -------------------------
        if state.active_mode == BotMode.HUNTER and not self.guardian.block_new_positions:
            signals = self.hunter.evaluate_all_signals(state, self.portfolio)
            exits = self.hunter.evaluate_all_exits(
                state, self.portfolio,
                op_rising_days=self._op_rising_days,
                gbp_below_threshold_days=self._gbp_low_days,
            )
            await self._execute_signals(signals, state)
            await self._execute_exits(exits)
        elif state.active_mode == BotMode.SCOUT:
            report = self.scout.generate_scout_report(state)
            log.info("Scout observation: %s", report["observation"])

        # -------------------------
        # Step 9: Structural stops for all positions
        # -------------------------
        stop_exits = evaluate_stops(state, self.portfolio)
        await self._execute_exits(stop_exits)

        # -------------------------
        # Step 10: Portfolio D_eff monitor
        # -------------------------
        self.guardian.check_portfolio_d_eff(self.portfolio)

        # -------------------------
        # Step 11: State snapshot (every 5 cycles ≈ 25 minutes)
        # -------------------------
        if self._cycle_count % 5 == 0:
            await self._checkpoint_state(state)

        # Update tracking counters
        self._update_tracking_counters(state)

        return state

    async def _execute_signals(self, signals: list[dict], state) -> None:
        """Route signals to execution agent."""
        if not signals:
            return

        from psibot.execution.sizing import size_order
        from psibot.execution.stops import pre_entry_stop_check
        from psibot.execution.broker_api import BrokerAPI, Order, OrderType
        from psibot.state.portfolio_state import Position
        import uuid

        broker = BrokerAPI(
            provider="paper" if self.paper_mode else self.config.get("broker", {}).get("provider", "paper"),
            config=self.config.get("broker", {}),
        )

        for signal in signals:
            # Pre-entry stop check
            if not pre_entry_stop_check(signal, state):
                log.warning("Pre-entry stop check failed for %s — signal rejected",
                            signal["signal_class"].value)
                continue

            notional = size_order(signal, state, self.portfolio, self.config.get("execution", {}))
            if notional <= 0:
                continue

            # Create and submit order
            primary_instrument = signal["instruments"][0] if signal.get("instruments") else "SPX_FUT"
            order = Order(
                instrument=primary_instrument,
                direction=signal["direction"],
                notional_usd=notional,
                order_type=OrderType.LIMIT,
                signal_class=signal["signal_class"],
            )
            filled_order = await broker.submit_order(order)

            if filled_order.status.value == "FILLED":
                pos = Position(
                    position_id=filled_order.order_id,
                    signal_class=signal["signal_class"],
                    direction=signal["direction"],
                    instrument=primary_instrument,
                    entry_gbp=state.gbp,
                    entry_phase=state.phase,
                    entry_psi_shape=state.psi_shape,
                    entry_d_eff=state.d_eff,
                    entry_price=filled_order.fill_price or 100.0,
                    current_price=filled_order.fill_price or 100.0,
                    notional_usd=notional,
                    size_multiplier=signal.get("size_multiplier", 1.0),
                )
                self.portfolio.add_position(pos)
                log.info("Position opened: %s %s %.0f USD GBP=%.3f",
                         signal["signal_class"].value, signal["direction"].value,
                         notional, state.gbp)

    async def _execute_exits(self, exit_decisions: list[dict]) -> None:
        """Execute position exits via broker."""
        from psibot.execution.broker_api import BrokerAPI

        broker = BrokerAPI(
            provider="paper" if self.paper_mode else "paper",
            config=self.config.get("broker", {}),
        )

        for exit_info in exit_decisions:
            for pos in self.portfolio.open_positions:
                if pos.position_id == exit_info["position_id"]:
                    urgency = exit_info.get("urgency", "normal")
                    await broker.close_position(pos, "; ".join(exit_info["reasons"]), urgency)
                    pos.close("; ".join(exit_info["reasons"]))
                    self.portfolio.account_equity += pos.realised_pnl_usd
                    log.info("Position closed: %s reason=%s pnl=%.0f",
                             pos.position_id[:8], pos.close_reason, pos.realised_pnl_usd)
                    break

    async def _checkpoint_state(self, state: CondensateState) -> None:
        """Save full state snapshot to disk."""
        try:
            snapshot = {
                "state": state.summary_dict(),
                "portfolio": self.portfolio.snapshot(),
                "guardian": self.guardian.status,
                "cycle": self._cycle_count,
            }
            filename = os.path.join(
                self.snapshot_dir,
                f"snapshot_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.json"
            )
            with open(filename, "w") as f:
                json.dump(snapshot, f, indent=2, default=str)
            log.debug("State snapshot saved: %s", filename)
        except Exception as e:
            log.warning("Snapshot failed: %s", e)

    def _update_tracking_counters(self, state: CondensateState) -> None:
        """Update OP rising days and GBP low days counters."""
        # OP rising counter (for Transition exit)
        if state.op_trend_5d > 0.02:
            self._op_rising_days += 1
        else:
            self._op_rising_days = 0

        # GBP below 0.2 counter (for Reorder exit)
        if state.gbp < 0.2:
            self._gbp_low_days += 1
        else:
            self._gbp_low_days = 0

    async def run_continuous(self, interval_minutes: int = 5) -> None:
        """
        Run pipeline continuously on configured interval.
        Market hours only (9:30-16:00 ET by default).
        """
        import asyncio
        log.info("ΨBot starting continuous run (interval=%dm, paper=%s)",
                 interval_minutes, self.paper_mode)

        while True:
            try:
                await self.run_cycle(trigger="scheduled")
            except KeyboardInterrupt:
                log.info("ΨBot stopping — keyboard interrupt")
                break
            except Exception as e:
                log.error("Unhandled exception in pipeline cycle: %s", e, exc_info=True)
                # Safety: activate Guardian on ANY unhandled exception
                log.critical("GUARDIAN ACTIVATED: unhandled exception in pipeline")
                self.guardian.guardian_active = True

            await asyncio.sleep(interval_minutes * 60)


def load_config(settings_path: str = None) -> dict:
    """Load settings.yaml configuration."""
    import yaml
    if settings_path is None:
        settings_path = os.path.join(os.path.dirname(__file__), "config", "settings.yaml")
    try:
        with open(settings_path) as f:
            return yaml.safe_load(f) or {}
    except FileNotFoundError:
        log.warning("settings.yaml not found at %s — using defaults", settings_path)
        return {}
    except Exception as e:
        log.error("Failed to load settings: %s", e)
        return {}


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="ΨBot — CCDR Expectation Field Trading Agent")
    parser.add_argument("--mode", choices=["full", "scout", "paper"], default="paper",
                        help="Pipeline mode (default: paper)")
    parser.add_argument("--layers", default="1,2,3,4,5",
                        help="Pipeline layers to run (default: 1,2,3,4,5)")
    parser.add_argument("--no-broker", action="store_true", default=True,
                        help="Paper trading — no real broker connection")
    parser.add_argument("--paper-trading", action="store_true", default=True)
    parser.add_argument("--config", default=None, help="Path to settings.yaml")
    parser.add_argument("--restore-from-checkpoint", action="store_true",
                        help="Restore from most recent state snapshot")
    parser.add_argument("--once", action="store_true",
                        help="Run one pipeline cycle and exit")
    args = parser.parse_args()

    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )

    config = load_config(args.config)
    config["paper_trading"] = True  # always paper until Phase 3

    orchestrator = PsiBotOrchestrator(config=config)

    if args.once:
        # Single pipeline run
        loop = asyncio.get_event_loop()
        state = loop.run_until_complete(orchestrator.run_cycle(trigger="cli_once"))
        print(format_state_summary(state))
    else:
        # Continuous run
        interval = config.get("pipeline", {}).get("recalc_interval_minutes", 5)
        asyncio.run(orchestrator.run_continuous(interval_minutes=interval))


if __name__ == "__main__":
    main()
