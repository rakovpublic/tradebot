# ΨBot — JNEOPALLIUM Trading Agent
## Claude Code Project Instructions

> **CCDR Expectation Field Architecture · Version 1.0**
> Repository: github.com/rakovpublic/jneopallium

---

## Project Identity

ΨBot is an autonomous trading agent grounded in the CCDR (Crystallographic Cosmology from Dimensional Reduction) expectation field framework. Its core premise: financial markets are dark-sector dominated systems — the collective expectation field ψ_exp is the primary substance; physical transactions (prices, volume) are a small acoustic residue.

This is not standard quantitative finance. Every implementation decision must reflect the **expectation-first ontology**:

```
Primary data:    options implied vol surface → ψ_exp wavefunction
Secondary data:  analyst dispersion + surveys → condensate phase
Tertiary data:   cross-asset correlations → D_eff holographic gauge
Quaternary data: grain boundary score GBP ∈ [0,1]
Last priority:   prices and volume → acoustic confirmation only
```

---

## Project Structure

```
psibot/
├── claude.md              ← YOU ARE HERE
├── skill.md               ← capability registry
├── agents.md              ← agent definitions
├── deploy.md              ← deployment runbook
├── helpers.py             ← shared utilities
│
├── pipeline/
│   ├── __init__.py
│   ├── l1_psi_reconstruction.py   ← Layer 1: ψ_exp from options surface
│   ├── l2_phase_detector.py       ← Layer 2: condensate phase (OP, DP)
│   ├── l3_holo_monitor.py         ← Layer 3: D_eff holographic saturation
│   ├── l4_grain_boundary.py       ← Layer 4: GBP grain boundary proximity
│   └── l5_acoustic_parser.py      ← Layer 5: price/vol confirmation
│
├── signals/
│   ├── __init__.py
│   ├── soliton.py                 ← Soliton-Long/Short signal class
│   ├── transition.py              ← Transition (long vol) signal class
│   ├── reorder.py                 ← Reorder (new grain) signal class
│   └── saturation_hedge.py        ← Saturation-Hedge signal class
│
├── modes/
│   ├── __init__.py
│   ├── scout.py                   ← Scout mode: observe only
│   ├── hunter.py                  ← Hunter mode: active execution
│   └── guardian.py                ← Guardian mode: risk management
│
├── execution/
│   ├── __init__.py
│   ├── sizing.py                  ← Position sizing: f(D_eff) × f(GBP)
│   ├── stops.py                   ← CCDR structural stops
│   └── broker_api.py              ← FIX API / broker integration
│
├── data/
│   ├── __init__.py
│   ├── options_feed.py            ← Options surface ingestion
│   ├── analyst_feed.py            ← IBES dispersion + survey data
│   ├── dark_pool_feed.py          ← FINRA ATS + dark pool tape
│   └── cross_asset_feed.py        ← 27-asset universe returns
│
├── state/
│   ├── __init__.py
│   ├── condensate_state.py        ← Full condensate state object
│   └── portfolio_state.py         ← Current positions + P&L
│
├── backtesting/
│   ├── __init__.py
│   ├── hypothesis_tests.py        ← T1–T9 validation tests
│   └── backtest_engine.py         ← Historical simulation engine
│
├── config/
│   ├── settings.yaml              ← All configurable parameters
│   └── universe.yaml              ← 27-asset universe definition
│
└── tests/
    ├── test_l1.py
    ├── test_l2.py
    ├── test_l3.py
    ├── test_l4.py
    └── test_pipeline_integration.py
```

---

## Core Data Structures

All pipeline layers communicate via a single `CondensateState` object. **Never pass raw prices as a primary signal between layers.**

```python
# state/condensate_state.py — canonical state object

@dataclass
class CondensateState:
    timestamp: datetime

    # L1 outputs — PRIMARY
    psi_shape: PsiShape          # Gaussian|Bimodal|SkewedLeft|SkewedRight|FatTailed
    psi_skew: float              # negative=bearish chirality, positive=bullish
    psi_kurtosis_excess: float   # proxy for grain boundary proximity
    psi_entropy: float           # high=disordered condensate
    psi_term_structure: TermStructure  # Contango|Backwardation|InvertedHump

    # L2 outputs — ORDER AND DISORDER PARAMETERS
    order_parameter: float       # OP ∈ [-1, +1], net-bullish coherence
    disorder_parameter: float    # DP > 0, analyst forecast dispersion
    phase: MarketPhase           # Ordered_Bull|Ordered_Bear|Disordered|ReOrdering

    # L3 outputs — HOLOGRAPHIC SATURATION
    d_eff: float                 # ∈ [1, N=27], effective dimensionality
    d_eff_trend_10d: float       # rate of change; negative = crisis building
    bot_mode_from_deff: BotMode  # Scout|Hunter|Guardian from D_eff alone

    # L4 outputs — GRAIN BOUNDARY
    gbp: float                   # ∈ [0, 1], grain boundary proximity
    gbp_components: dict         # {'psi':..., 'dp':..., 'deff':..., 'darkpool':...}

    # L5 outputs — ACOUSTIC CONFIRMATION (lowest priority)
    acoustic_signal: AcousticSignal  # Confirm|Contradict|Neutral
    momentum_20d: float
    momentum_60d: float
    breadth: float               # % assets above 50d MA

    # Derived
    active_mode: BotMode         # final mode after all layers
    signal_size_multiplier: float  # f(D_eff) × f(GBP)
```

---

## Pipeline Execution Order

**Critical rule: L1 must complete before L2, L2 before L3, etc. Never run layers in parallel.**

```python
# Main pipeline execution — run every 5 minutes intraday, every EOD

async def run_pipeline(market_data: MarketData) -> CondensateState:
    state = CondensateState(timestamp=datetime.utcnow())

    # L1: Reconstruct ψ_exp from options surface — PRIMARY SENSOR
    state = await l1_psi_reconstruction.run(state, market_data.options_surface)

    # L2: Measure condensate phase from analyst data
    state = await l2_phase_detector.run(state, market_data.analyst_data)

    # L3: Compute D_eff — holographic saturation gauge
    state = await l3_holo_monitor.run(state, market_data.cross_asset_returns)

    # L4: Compute GBP — synthesise all upstream signals
    state = await l4_grain_boundary.run(state, market_data.dark_pool_fraction)

    # L5: Parse acoustic residue — CONFIRMATION ONLY
    state = await l5_acoustic_parser.run(state, market_data.prices_volumes)

    # Determine active mode
    state.active_mode = determine_mode(state)
    state.signal_size_multiplier = compute_size_multiplier(state)

    return state
```

---

## Mode Logic

```python
def determine_mode(state: CondensateState) -> BotMode:
    # Guardian takes priority — always check first
    if state.d_eff <= 3.0 or state.gbp >= 0.8:
        return BotMode.GUARDIAN

    # Scout: post-crisis recovery or contradictory signals
    if state.d_eff <= 5.0:
        return BotMode.SCOUT
    if state.phase == MarketPhase.DISORDERED and state.gbp > 0.6:
        return BotMode.SCOUT

    # Hunter: normal operating conditions
    if state.gbp < 0.5 and state.d_eff > 5.0 and state.phase != MarketPhase.DISORDERED:
        return BotMode.HUNTER

    return BotMode.SCOUT  # default to Scout when uncertain
```

---

## Position Sizing Formula

```python
def compute_size_multiplier(state: CondensateState) -> float:
    # f(D_eff): linear interpolation
    if state.d_eff >= 20:
        f_deff = 1.0
    elif state.d_eff <= 3:
        f_deff = 0.0
    else:
        f_deff = (state.d_eff - 3) / (20 - 3)

    # f(GBP): linear interpolation
    if state.gbp <= 0.1:
        f_gbp = 1.0
    elif state.gbp >= 0.7:
        f_gbp = 0.0
    else:
        f_gbp = (0.7 - state.gbp) / (0.7 - 0.1)

    base = f_deff * f_gbp

    # L5 acoustic modifier
    if state.acoustic_signal == AcousticSignal.CONFIRM:
        return min(base * 1.2, 1.0)
    elif state.acoustic_signal == AcousticSignal.CONTRADICT:
        return base * 0.7
    return base
```

---

## CCDR Structural Stop Logic

**These stops replace all price-level stop-losses. Never use a fixed price stop.**

```python
# stops.py

STOPS = {
    'gbp_delta':    0.35,   # exit when GBP rises above entry_GBP + 0.35
    'd_eff_floor':  4.0,    # exit all risk positions when D_eff < 4
    'phase_change': True,   # exit when phase changes ordered → disordered
    'psi_bimodal':  True,   # exit soliton trades when ψ turns bimodal
}

def check_structural_stops(position: Position, state: CondensateState) -> StopDecision:
    reasons = []

    if state.gbp > position.entry_gbp + STOPS['gbp_delta']:
        reasons.append(f"GBP stop: {state.gbp:.2f} > {position.entry_gbp + STOPS['gbp_delta']:.2f}")

    if state.d_eff < STOPS['d_eff_floor']:
        reasons.append(f"D_eff stop: {state.d_eff:.1f} < {STOPS['d_eff_floor']}")

    if STOPS['phase_change'] and position.entry_phase.is_ordered() and state.phase == MarketPhase.DISORDERED:
        reasons.append("Phase stop: ordered → disordered transition")

    if STOPS['psi_bimodal'] and position.signal_class == SignalClass.SOLITON:
        if state.psi_shape == PsiShape.BIMODAL:
            reasons.append("ψ shape stop: soliton trade, ψ turned bimodal")

    return StopDecision(triggered=bool(reasons), reasons=reasons)
```

---

## Coding Standards

### Language and Libraries
- Python 3.11+
- `numpy`, `scipy` for numerical work
- `pandas` for time series
- `torch` or `cupy` for GPU-accelerated L1 Dupire PDE
- `asyncio` for async pipeline execution
- `pydantic` for all data models / state validation
- `structlog` for structured logging
- `pytest` for all tests (minimum 80% coverage before any layer goes live)

### Naming Conventions
- All CCDR-specific variables use snake_case with CCDR terminology:
  - `psi_exp`, `d_eff`, `gbp`, `order_parameter`, `disorder_parameter`
  - `grain_boundary_proximity`, `condensate_phase`, `holographic_saturation`
- Never abbreviate pipeline layers in code: `l1_psi_reconstruction`, not `l1` or `psi`
- Signal classes: `soliton_signal`, `transition_signal`, `reorder_signal`, `saturation_hedge_signal`
- Operating modes: `scout_mode`, `hunter_mode`, `guardian_mode`

### Error Handling
- Every pipeline layer must return a valid `CondensateState` even on data failure
- On L1 failure (options surface unavailable): set `psi_shape = PsiShape.UNKNOWN`, `gbp += 0.2` (conservative)
- On L3 failure (correlation matrix error): set `d_eff = 5.0` (conservative mid-range)
- Guardian mode activates on ANY unhandled exception during live trading
- Log all state transitions with full `CondensateState` snapshot

### Testing Requirements
Before any component goes live:
1. Unit tests for each pipeline layer in isolation
2. Integration test: full pipeline run on 10 years of historical data
3. Backtesting: all 9 CCDR hypotheses must pass (see `backtesting/hypothesis_tests.py`)
4. Paper trading: minimum 60 trading days with sharpe > 0.5 before live

---

## What Claude Code Must NOT Do

- **Never use price-level stop-losses** — use CCDR structural stops only
- **Never generate a trade signal from L5 (price/volume) alone** — L5 is confirmation only
- **Never skip L1** — the options surface is the primary sensor; all other layers depend on it
- **Never run in Hunter mode when D_eff < 5** — enforced programmatically
- **Never add leverage > 2× on any single position**
- **Never exceed 6 concurrent correlated positions** — portfolio D_eff constraint
- **Never deploy live without passing all 9 backtesting hypotheses**
- **Never claim the system guarantees profits** — CCDR is a theoretical framework

---

## Key Constants

```python
# config/settings.yaml equivalent in Python

class CCDRConfig:
    # L3 D_eff thresholds
    D_EFF_GUARDIAN_TRIGGER  = 3.0
    D_EFF_SCOUT_TRIGGER     = 5.0
    D_EFF_FULL_HUNTER       = 15.0
    D_EFF_FLOOR_STOP        = 4.0

    # L4 GBP thresholds
    GBP_GUARDIAN_TRIGGER    = 0.8
    GBP_HUNTER_MAX          = 0.5
    GBP_STOP_DELTA          = 0.35
    GBP_REENTRY_MAX         = 0.3

    # L4 GBP weights
    GBP_WEIGHT_PSI          = 0.35
    GBP_WEIGHT_DP_TREND     = 0.25
    GBP_WEIGHT_DEFF_TREND   = 0.30
    GBP_WEIGHT_DARK_POOL    = 0.10

    # L2 phase thresholds
    OP_ORDERED_MIN          = 0.4
    OP_DISORDERED_MAX       = 0.15
    DP_BOUNDARY_THRESHOLD   = 0.25  # DP above this = grain boundary approach

    # Execution
    MAX_CONCURRENT_POSITIONS = 6
    MAX_LEVERAGE             = 2.0
    DRAWDOWN_CIRCUIT_BREAKER = 0.05  # 5% in rolling 10 days
    GUARDIAN_COOLOFF_HOURS   = 48

    # Pipeline timing
    INTRADAY_RECALC_MINUTES  = 5
    L1_RECALC_ON_VOL_SHIFT   = 0.5  # vega shift threshold
    D_EFF_ROLLING_WINDOW     = 60   # trading days
```

---

## References

- CCDR Version 6: Rakovskyi (2026), IJSR SR24703042047
- "Markets Trade Expectations, Not Goods": Rakovskyi (2026)
- ΨBot Architecture Description: Rakovskyi (2026)
- Repository: github.com/rakovpublic/jneopallium
