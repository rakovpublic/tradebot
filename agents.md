# ΨBot Agent Definitions
## Multi-Agent System Architecture

> **JNEOPALLIUM — CCDR Expectation Field Architecture**
> Each agent is a specialised Claude Code sub-process responsible for one layer or function.
> Agents communicate through the shared `CondensateState` object (see `helpers.py`).

---

## Agent Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                      ORCHESTRATOR AGENT                         │
│              Coordinates pipeline + mode decisions              │
└───────┬──────┬──────┬──────┬──────┬──────────────────────┬──────┘
        │      │      │      │      │                      │
        ▼      ▼      ▼      ▼      ▼                      ▼
      L1     L2     L3     L4     L5                  BACKTEST
    AGENT  AGENT  AGENT  AGENT  AGENT                  AGENT
      │                                                    │
      ▼                                                    ▼
   SIGNAL                                              RISK
   AGENT                                              AGENT
      │
      ▼
  EXECUTION
   AGENT
```

| Agent ID | Name | Primary Responsibility | Runs |
|----------|------|----------------------|------|
| AGT-00 | Orchestrator | Pipeline coordination, mode switching, state management | Always |
| AGT-01 | ψ_exp Agent | Layer 1: options surface → wavefunction reconstruction | Every 5 min + on vol shift |
| AGT-02 | Phase Agent | Layer 2: analyst data → condensate phase (OP, DP) | Daily + on data update |
| AGT-03 | D_eff Agent | Layer 3: cross-asset returns → holographic saturation | Every 5 min intraday |
| AGT-04 | Grain Agent | Layer 4: GBP synthesis from L1–L3 + dark pool | After each L1/L3 update |
| AGT-05 | Acoustic Agent | Layer 5: price/vol confirmation parser | After each L4 update |
| AGT-06 | Signal Agent | All four signal class generation and lifecycle | After each full pipeline run |
| AGT-07 | Risk Agent | Guardian mode, stops, circuit breaker, portfolio D_eff | Real-time, highest priority |
| AGT-08 | Execution Agent | Order sizing, routing, fill monitoring | On signal + on stop trigger |
| AGT-09 | Backtest Agent | T1–T9 hypothesis testing, paper trade simulation | Phase 0–2 only |
| AGT-10 | Monitor Agent | Dashboards, alerts, logging, state snapshots | Always |

---

## AGT-00: Orchestrator Agent

**Role:** The master coordinator. Runs the pipeline in sequence, manages mode state, and routes results between agents. Never generates trade signals itself.

**Responsibilities:**
- Trigger L1–L5 agents in order at correct intervals
- Collect `CondensateState` after each layer and pass to next agent
- Call `determine_bot_mode()` after L4
- Route final state to AGT-06 (Signal) if mode is Hunter
- Route to AGT-07 (Risk) immediately if Guardian triggers are met
- Checkpoint state every 5 minutes

**Trigger cadence:**
```
Every 5 minutes (intraday, market hours):
  → Trigger AGT-03 (D_eff) — fast, uses cached correlation matrix
  → Trigger AGT-04 (Grain) — GBP recompute from new D_eff
  → If D_eff or GBP changed materially → trigger full AGT-01→AGT-05 chain

On options vol surface shift > 0.5 vega (any time):
  → Trigger full AGT-01→AGT-05 chain immediately

Daily at 07:00 UTC (before market open):
  → Trigger AGT-02 (Phase) — analyst data update
  → Trigger full AGT-01→AGT-05 chain
  → AGT-09 (Backtest/Monitor): generate daily state report
```

**Pseudo-code:**
```python
class OrchestratorAgent:
    async def run_cycle(self, trigger: Trigger) -> CondensateState:
        state = CondensateState(timestamp=datetime.utcnow())

        # Always check AGT-07 (Risk) first — Guardian overrides everything
        await self.risk_agent.pre_check(state)

        # Run pipeline in order
        state = await self.psi_agent.run(state)         # L1
        state = await self.phase_agent.run(state)       # L2
        state = await self.deff_agent.run(state)        # L3
        state = await self.grain_agent.run(state)       # L4
        state = await self.acoustic_agent.run(state)    # L5

        # Determine mode
        state.active_mode = determine_bot_mode(
            state.d_eff, state.gbp, state.phase)
        state.signal_size_multiplier = compute_position_size(...)

        # Route
        if state.active_mode == BotMode.GUARDIAN:
            await self.risk_agent.activate_guardian(state)
        elif state.active_mode == BotMode.HUNTER:
            await self.signal_agent.evaluate(state)

        # Always snapshot
        await self.monitor_agent.snapshot(state)
        return state
```

---

## AGT-01: ψ_exp Agent (Layer 1)

**Role:** Reconstruct the expectation field wavefunction from the options implied volatility surface. This is the system's primary sensor.

**Inputs:**
- `OptionsSurface` object from `data/options_feed.py`
- Spot price for each instrument

**Outputs written to CondensateState:**
- `psi_shape: PsiShape`
- `psi_skew: float` (condensate chirality proxy)
- `psi_kurtosis_excess: float` (grain boundary proximity proxy)
- `psi_entropy: float` (condensate disorder)
- `psi_term_structure: TermStructure`

**Algorithm summary:**
1. Validate surface (`helpers.validate_options_surface`)
2. SVI/SABR smoothing for arbitrage-free surface
3. Dupire PDE → local vol → risk-neutral density p(S_T)
4. Classify shape (`helpers.classify_psi_shape`)
5. Compute skew, kurtosis proxy, entropy, term structure

**Failure protocol:**
```python
# On options data unavailable:
state.psi_shape = PsiShape.UNKNOWN
state.psi_skew = 0.0
# GBP will use conservative UNKNOWN score = 0.60
# Log: WARNING "L1 data unavailable — conservative GBP contribution applied"
```

**Target latency:** < 100ms (GPU-accelerated Dupire PDE)
**Required tests:** `tests/test_l1.py` — all must pass

---

## AGT-02: Phase Agent (Layer 2)

**Role:** Measure the order parameter (OP) and disorder parameter (DP) of the expectation condensate from analyst and survey data.

**Inputs:**
- `AnalystData`: IBES forward EPS estimates across analysts
- `SurveyData`: AAII, Investors Intelligence, institutional surveys

**Outputs written to CondensateState:**
- `order_parameter: float` — OP ∈ [-1, +1]
- `disorder_parameter: float` — DP > 0
- `phase: MarketPhase`

**Update frequency:** Daily (analyst data) + weekly (surveys)
**Between updates:** Use last known values; flag staleness > 5 business days

**Core logic:**
```python
op = compute_order_parameter(survey_data)      # helpers.py
dp = compute_disorder_parameter(analyst_data)  # helpers.py
phase = classify_market_phase(
    op, dp,
    op_trend_5d=op - prev_op_5d_ago,
    dp_trend_5d=dp - prev_dp_5d_ago,
)
```

**Phase transition events:** When phase changes, AGT-07 (Risk) is immediately notified regardless of other conditions.

---

## AGT-03: D_eff Agent (Layer 3)

**Role:** Compute the effective dimensionality D_eff of the cross-asset expectation condensate. This is the leading systemic risk gauge.

**Inputs:**
- `pd.DataFrame`: 60-day rolling window of daily log-returns for all 27 universe assets

**Outputs written to CondensateState:**
- `d_eff: float` — current effective dimensionality
- `d_eff_trend_10d: float` — 10-day slope (negative = crisis building)
- `bot_mode_from_deff: BotMode`

**Algorithm:**
```python
d_eff = compute_d_eff(returns_matrix)  # helpers.py
d_eff_history.append(d_eff)
d_eff_trend_10d = compute_d_eff_trend(pd.Series(d_eff_history), window=10)
bot_mode_from_deff = d_eff_to_bot_mode(d_eff)
```

**Hard trigger:** If `d_eff < 3.0`, notify AGT-07 immediately via high-priority event (do not wait for normal pipeline cycle).

**Target latency:** < 50ms (27×27 eigenvalue decomposition)

---

## AGT-04: Grain Boundary Agent (Layer 4)

**Role:** Synthesise all upstream signals (L1, L2, L3) plus dark pool data into the single most important number: GBP ∈ [0, 1].

**Inputs:**
- `state.psi_shape` from AGT-01
- `state.disorder_parameter` trend from AGT-02
- `state.d_eff_trend_10d` from AGT-03
- Dark pool fraction from `data/dark_pool_feed.py`

**Outputs written to CondensateState:**
- `gbp: float` — grain boundary proximity
- `gbp_components: dict` — breakdown for monitoring

**Algorithm:**
```python
dp_trend_10d = dp - dp_10d_ago
gbp, components = compute_gbp(
    psi_shape=state.psi_shape,
    dp_trend_10d=dp_trend_10d,
    d_eff_trend_20d=state.d_eff_trend_10d,
    dark_pool_ratio=dark_pool_fraction / dark_pool_90d_ma,
)
```

**Hard trigger:** If `gbp >= 0.8`, notify AGT-07 immediately.

---

## AGT-05: Acoustic Agent (Layer 5)

**Role:** Parse conventional market data (prices, volumes) for confirmation or contradiction of the upstream expectation field signals. **This agent never generates primary signals.**

**Inputs:**
- Price time series (20d and 60d momentum)
- Volume relative to 20d average
- Bid-ask spread relative to 30d average
- Market breadth (% assets above 50d MA)
- Realised put-call ratio

**Outputs written to CondensateState:**
- `acoustic_signal: AcousticSignal` — CONFIRM / CONTRADICT / NEUTRAL
- `momentum_20d: float`
- `momentum_60d: float`
- `breadth: float`

**Confirmation logic (simplified):**
```python
def classify_acoustic(state, prices) -> AcousticSignal:
    # Confirm if momentum aligns with ψ_exp chirality
    if state.psi_shape == PsiShape.SKEWED_RIGHT and momentum_20d > 0:
        return AcousticSignal.CONFIRM
    if state.psi_shape == PsiShape.SKEWED_LEFT and momentum_20d < 0:
        return AcousticSignal.CONFIRM
    # Contradict if momentum opposes chirality
    if state.psi_shape == PsiShape.SKEWED_RIGHT and momentum_20d < -0.02:
        return AcousticSignal.CONTRADICT
    return AcousticSignal.NEUTRAL
```

---

## AGT-06: Signal Agent

**Role:** Generate trade signal candidates from the four CCDR signal classes (Soliton, Transition, Reorder, Saturation-Hedge) and manage their lifecycle.

**Only runs when:** `state.active_mode == BotMode.HUNTER` or for SaturationHedge when approaching Guardian.

**Signal generation logic per class:**

### Soliton (SK-12)
```
Entry conditions (all required):
  □ active_mode == HUNTER
  □ gbp < 0.3
  □ phase in [ORDERED_BULL, ORDERED_BEAR]
  □ psi_shape in [SKEWED_RIGHT, SKEWED_LEFT]
  □ No existing Soliton in same direction
Exit (any triggers stop):
  □ GBP stop: gbp > entry_gbp + 0.35
  □ D_eff stop: d_eff < 4.0
  □ Phase stop: phase → DISORDERED
  □ ψ shape stop: psi_shape → BIMODAL
```

### Transition (SK-13)
```
Entry conditions (all required):
  □ active_mode in [HUNTER, SCOUT with special override]
  □ gbp > 0.65
  □ dp_trend accelerating (dp_trend_10d > 90th percentile)
  □ d_eff_trend_10d < -0.3
Exit:
  □ L2 re-ordering signal: |op| rising from trough for 3+ days
  □ gbp falls below 0.4
```

### Reorder (SK-14)
```
Entry conditions (all required):
  □ active_mode == HUNTER
  □ phase == REORDERING
  □ psi_shape == GAUSSIAN (new stable condensate forming)
  □ gbp < 0.3
  □ dp declining: dp_trend_5d < 0
Exit:
  □ gbp < 0.2 sustained for 3+ days (trade complete)
  □ Phase changes away from ORDERED
```

### Saturation-Hedge (SK-15)
```
Entry conditions (any triggers entry):
  □ d_eff < 6.0 AND d_eff_trend_10d < -0.5 per day
  □ psi_shape == FAT_TAILED AND d_eff < 8.0
Exit:
  □ d_eff recovers above 8.0 and stabilises
  □ psi_shape returns to GAUSSIAN
```

**Max concurrent positions:** 6 total. AGT-06 checks `portfolio.position_count` before generating any new signal.

---

## AGT-07: Risk Agent

**Role:** Highest-priority agent. Monitors for Guardian triggers, enforces all CCDR structural stops, manages the drawdown circuit breaker, and activates/deactivates Guardian mode.

**Always runs.** Runs in parallel with the pipeline (not in sequence) and can interrupt any other agent.

**Responsibilities:**

### Guardian Mode Activation
```python
def check_guardian_triggers(state: CondensateState, portfolio: PortfolioState) -> bool:
    # Hard triggers — immediate Guardian activation
    if state.d_eff <= 3.0:
        self.activate_guardian("D_eff below 3.0 — holographic saturation")
        return True
    if state.gbp >= 0.8:
        self.activate_guardian("GBP >= 0.8 — grain boundary crossing in progress")
        return True

    # Drawdown circuit breaker
    drawdown_10d = portfolio.rolling_drawdown(days=10)
    if drawdown_10d <= -0.05:  # 5% loss in rolling 10 days
        self.activate_guardian(f"Drawdown circuit breaker: {drawdown_10d:.1%}")
        return True

    return False
```

### Guardian Mode Actions
```python
def activate_guardian(self, reason: str):
    log.warning("GUARDIAN ACTIVATED: %s", reason)

    # 1. Flatten all Soliton and Reorder positions
    for pos in portfolio.risk_positions:
        execution_agent.close_position(pos, reason=f"Guardian: {reason}")

    # 2. Maintain Transition positions (vol positions hedge Guardian risk)
    # 3. Add Saturation-Hedge positions at 50% normal sizing
    signal_agent.generate_saturation_hedge(size_mult=0.5)

    # 4. Set mode state
    self.guardian_active = True
    self.guardian_activated_at = datetime.utcnow()

    # 5. Alert
    monitor_agent.alert(f"GUARDIAN MODE ACTIVATED: {reason}", priority="CRITICAL")
```

### Guardian Mode Deactivation
```python
def check_guardian_exit(self, state: CondensateState) -> bool:
    if not self.guardian_active:
        return False

    # Minimum cooloff period
    if datetime.utcnow() - self.guardian_activated_at < timedelta(hours=48):
        return False

    # All conditions must be met simultaneously for 5 business days
    conditions_met = (
        state.d_eff > 6.0 and
        state.gbp < 0.4 and
        state.phase.is_ordered()
    )
    if conditions_met:
        self.consecutive_clean_days += 1
    else:
        self.consecutive_clean_days = 0

    if self.consecutive_clean_days >= 5:
        self.deactivate_guardian()
        return True
    return False
```

### Structural Stop Monitoring
```python
# AGT-07 checks structural stops for every open position on every pipeline cycle
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
        execution_agent.close_position(position, reason=", ".join(stop_result.reasons))
```

### Portfolio D_eff Monitor
```python
# Ensure portfolio never becomes its own holographic saturation event
def check_portfolio_d_eff(portfolio: PortfolioState):
    # Build correlation matrix of open position P&L series
    if len(portfolio.open_positions) < 3:
        return  # not enough to compute

    pnl_matrix = portfolio.get_pnl_matrix(days=30)
    portfolio_d_eff = compute_d_eff(pnl_matrix)

    if portfolio_d_eff < 4.0:
        log.warning("Portfolio D_eff = %.1f — too concentrated", portfolio_d_eff)
        # Block new positions until portfolio D_eff recovers
        self.block_new_positions = True
```

---

## AGT-08: Execution Agent

**Role:** Convert signal objects into exchange orders, manage fills, and handle order lifecycle.

**Never generates signals.** Only executes what AGT-06 (Signal) or AGT-07 (Risk) instructs.

**Order types:**
- Soliton / Reorder entries: **limit orders** (patient fills)
- Guardian exits: **market orders** (immediacy required)
- Saturation hedges: limit orders with 15-minute expiry → market if unfilled

**Position sizing calculation:**
```python
def size_order(signal: Signal, state: CondensateState, portfolio: PortfolioState) -> float:
    base_usd = compute_position_size(
        d_eff=state.d_eff,
        gbp=state.gbp,
        acoustic=state.acoustic_signal,
        max_risk_usd=config.max_risk_usd,
    )
    # Check portfolio-level constraints
    remaining_capacity = (config.max_concurrent_positions
                          - portfolio.position_count)
    if remaining_capacity <= 0:
        log.warning("Max positions (%d) reached — blocking new signal",
                    config.max_concurrent_positions)
        return 0.0

    # Never exceed 2× leverage on single position
    max_notional = portfolio.account_equity * config.max_leverage
    return min(base_usd, max_notional)
```

---

## AGT-09: Backtest Agent

**Role:** Run T1–T9 CCDR hypothesis tests and paper trade simulation. Active in Phases 0–2 only. Dormant in Phase 3+.

**Capabilities:**
- Historical pipeline replay (all 5 layers on historical data)
- T1–T9 statistical test suite (see `skill.md::SK-21`)
- Paper trade P&L simulation with full CCDR structural stops
- Performance reporting

**Key entry point:**
```bash
python -m psibot.backtest --hypotheses all --start 1990-01-01
python -m psibot.backtest --paper-trade --start 2020-01-01 --end 2024-12-31
```

---

## AGT-10: Monitor Agent

**Role:** All observability — structured logging, dashboards, alerts, state snapshots.

**Outputs:**
- Structured JSON logs (every pipeline cycle, every signal, every mode change)
- Prometheus metrics (D_eff, GBP, active mode, position count, P&L)
- Grafana dashboard (live visual of all CondensateState fields)
- Slack/email alerts on: Guardian activation, D_eff < 5, GBP > 0.6, drawdown > 3%
- State snapshots to `state/snapshots/` every 5 minutes

**Key dashboard panels:**
```
Row 1: D_eff gauge (1–27) | GBP gauge (0–1) | Active Mode | Current Phase
Row 2: ψ_exp shape classification | Skew | Kurtosis proxy | Entropy
Row 3: Order Parameter (OP) | Disorder Parameter (DP) | Term Structure
Row 4: Open positions | P&L today | Rolling drawdown | Portfolio D_eff
Row 5: D_eff 60-day history chart | GBP 30-day history | Pipeline latency
```

---

## Inter-Agent Communication

All agents communicate through:

1. **`CondensateState` object** — passed sequentially through L1→L2→L3→L4→L5
2. **High-priority event bus** — for Guardian triggers (D_eff < 3, GBP >= 0.8, drawdown 5%)
3. **Shared state store** — Redis or in-memory dict for cross-agent reads

```python
# Event bus usage
event_bus.subscribe("guardian_trigger", risk_agent.handle_guardian_trigger)
event_bus.subscribe("stop_triggered", execution_agent.handle_stop)
event_bus.subscribe("signal_generated", execution_agent.handle_signal)

# Publishing
event_bus.publish("guardian_trigger", {"reason": "D_eff < 3.0", "d_eff": 2.8})
```

---

## Agent Failure Protocol

| Agent Fails | Impact | Recovery Action |
|-------------|--------|-----------------|
| AGT-01 (ψ_exp) | L1 data missing | Set psi_shape=UNKNOWN, GBP += 0.2, Scout mode |
| AGT-02 (Phase) | Stale analyst data | Use last known values, flag staleness |
| AGT-03 (D_eff) | Correlation failure | Set d_eff=5.0 (conservative), Scout mode |
| AGT-04 (Grain) | GBP uncomputable | Set gbp=0.6 (conservative), block new Hunter signals |
| AGT-05 (Acoustic) | Price data missing | Set acoustic=NEUTRAL, continue (lowest priority) |
| AGT-06 (Signal) | Signal error | Log, skip signal, continue pipeline |
| AGT-07 (Risk) | **CRITICAL** | Immediate full stop, alert ops, manual review |
| AGT-08 (Execution) | Order fails | Log, retry once, then cancel and alert |
| AGT-10 (Monitor) | Dashboard down | Continue trading, escalate for monitoring restore |
