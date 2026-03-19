# ΨBot Deployment Runbook
## From Development to Live Trading

> **CCDR Expectation Field Architecture — Version 1.0**
> Before deploying any phase, re-read the Limitations section of the ΨBot architecture document.

---

## Deployment Phases

```
Phase 0 → Phase 1 → Phase 2 → Phase 3 → Phase 4 (live)
  ↓           ↓          ↓         ↓
Backtest   Paper L1+L3  Full Paper  Live (min)  Live (scale)
(no risk)  (no risk)   (no risk)   (small)     (institutional)
```

**Critical rule: No phase can begin until the preceding phase's exit criteria are fully met.**

---

## Phase 0: Backtesting Validation

**Duration:** 8–12 weeks
**Environment:** Offline — no live data, no broker connection
**Purpose:** Validate all nine CCDR market predictions against historical data before any live component is activated.

### Prerequisites
```bash
# Install dependencies
pip install -r requirements.txt

# Verify data availability (minimum requirements)
python scripts/check_data_availability.py --start 2000-01-01 --assets all

# Run unit tests for all pipeline components
pytest tests/ -v --tb=short
# REQUIRED: 100% of unit tests must pass
```

### Running Hypothesis Tests
```bash
# Run all 9 CCDR hypothesis tests
python -m backtesting.hypothesis_tests --run-all \
    --start-date 1990-01-01 \
    --output reports/hypothesis_report_$(date +%Y%m%d).json

# Check results
python scripts/check_hypothesis_results.py reports/hypothesis_report_latest.json
```

### Phase 0 Exit Criteria
```
□ T1 (vol surface Granger): p < 0.01 at lag ≤ 5 days          PASS / FAIL
□ T2 (analyst dispersion leads regime): p < 0.05, lead 2–8 mo  PASS / FAIL
□ T3 (momentum crashes bimodal): dip test p < 0.05             PASS / FAIL
□ T4 (D_eff leads crashes): median lead ≥ 25 days              PASS / FAIL
□ T5 (dark pool predicts direction): accuracy > 55%            PASS / FAIL
□ T6 (equity premium spectral peak): 3–7yr band dominant       PASS / FAIL
□ T7 (technical levels persist): > 70% persistence             PASS / FAIL
□ T8 (skew predicts regime): accuracy > 60%                    PASS / FAIL
□ T9 (drift ∝ dispersion): R² > 0.20                          PASS / FAIL

DEPLOYMENT GATE: ≥ 7 of 9 must PASS.
If any 3 FAIL: HOLD — do not proceed to Phase 1.
```

---

## Phase 1: Paper Trading — L1 + L3 Only (Scout Mode)

**Duration:** 30–60 trading days minimum
**Environment:** Live market data, no broker connection, Scout Mode only
**Purpose:** Validate ψ_exp reconstruction (L1) and D_eff monitor (L3) on live data.

### Infrastructure Setup
```bash
# 1. Configure data feeds
cp config/settings.yaml.example config/settings.yaml
# Edit settings.yaml:
#   options_feed: provider, API key, symbols
#   cross_asset_feed: 27 assets, data source

# 2. Start data ingestion
docker-compose up -d data_ingest

# 3. Verify options surface quality
python scripts/verify_options_quality.py --tenor 30 --symbol SPX --days 5

# 4. Start pipeline in Scout-only mode
python -m psibot.main --mode scout --layers 1,3 --no-broker
```

### Monitoring
```bash
# Live dashboard
python -m psibot.monitor --dashboard

# Check D_eff health every hour
watch -n 3600 'python scripts/check_d_eff.py'

# Alert if D_eff < 8 (early warning, not Guardian trigger)
python scripts/setup_alerts.py --d-eff-warning 8.0 --d-eff-critical 5.0
```

### Phase 1 Exit Criteria
```
□ Pipeline runs continuously for 30 trading days with < 5% data outages
□ ψ_exp reconstruction completes in < 100ms on target hardware (95th percentile)
□ D_eff values are plausible: range [2, 20], no stuck values, no extreme jumps
□ D_eff successfully detected the last 2 volatility events (VIX spikes > 25)
□ ψ shape classification: Gaussian > 60% of days (expected in non-crisis period)
□ Unit tests still pass after 30 days of live data
□ No unhandled exceptions in production logs
```

---

## Phase 2: Full Paper Trading — All Layers, Hunter Mode

**Duration:** 60 trading days minimum
**Environment:** Live market data, no broker connection, full pipeline, Hunter Mode
**Purpose:** Validate full 5-layer pipeline, all signal classes, and CCDR structural stops.

### Add L2, L4, L5 Data Feeds
```bash
# Add analyst data feed
python scripts/setup_analyst_feed.py \
    --provider ibes \
    --symbols SPX_CONSTITUENTS \
    --metrics forward_eps_dispersion

# Add survey data feed (weekly)
python scripts/setup_survey_feed.py \
    --sources aaii,investors_intelligence

# Add dark pool feed
python scripts/setup_dark_pool_feed.py \
    --provider finra_ats \
    --symbols SPX_CONSTITUENTS

# Start full pipeline
python -m psibot.main --mode full --no-broker --paper-trading
```

### Paper Trade Logging
```bash
# All signals are logged with full CondensateState snapshot
# Review daily
python scripts/review_paper_trades.py --date yesterday

# Weekly performance report
python scripts/paper_trade_report.py --period 1w
```

### Paper Trade Performance Thresholds
```
□ Soliton signals: hit rate > 50%, average R:R > 1.5:1
□ Transition signals: average vol expansion > 20% after entry
□ Reorder signals: new trend established within 10 days > 60% of cases
□ Guardian mode: activated on all VIX spikes > 30, no false activations < VIX 20
□ CCDR structural stops: never fire on D_eff stop when D_eff > 6
□ Position sizing: never exceeds max_risk_usd × 1.1 (rounding check)
□ Portfolio D_eff > 4 maintained > 95% of trading days
□ Sharpe ratio (paper) > 0.5 annualised over 60 days
□ Maximum paper drawdown < 8%
```

### Phase 2 Exit Criteria
```
□ All Phase 1 criteria still met
□ All paper trade performance thresholds met
□ All 9 hypothesis tests re-run on most recent data: still ≥ 7/9 pass
□ Compliance review: risk documentation completed
□ Legal: fund structure / account type established
□ Counterparty: prime broker agreement signed
```

---

## Phase 3: Live Trading — Minimum Scale

**Duration:** 90 trading days minimum before scaling
**Environment:** Live broker, real capital, minimum position sizes
**Max drawdown:** 5% triggers Scout mode for 48 hours
**Max single position:** 2× leverage
**Max concurrent positions:** 6

### Broker Integration
```bash
# 1. Configure broker API
cp config/broker.yaml.example config/broker.yaml
# Edit: FIX session parameters, account number, permissions

# 2. Test broker connection (paper first)
python scripts/test_broker_connection.py --paper

# 3. Verify order routing
python scripts/verify_order_routing.py \
    --instrument SPX_FUT \
    --size 1 \
    --dry-run

# 4. Verify risk limits are enforced before going live
python scripts/verify_risk_limits.py
```

### Pre-Flight Checklist (run before every live session)
```bash
python scripts/pre_flight_check.py
```

This script verifies:
```
□ D_eff current value > 5 (otherwise start in Scout mode)
□ GBP current value < 0.5 (otherwise start in Scout mode)
□ Options surface available and quality-checked
□ Analyst data current (not stale > 5 business days)
□ Cross-asset return data current
□ Broker connection active and order routing tested
□ Risk limits loaded: max_position, max_leverage, drawdown_cb
□ Guardian mode trigger values configured and tested
□ Emergency stop endpoint reachable
□ Monitoring dashboard running
```

### Emergency Procedures
```bash
# IMMEDIATE FULL STOP — use when something is very wrong
python scripts/emergency_stop.py --flatten-all --reason "manual_override"

# FLATTEN SINGLE INSTRUMENT
python scripts/emergency_stop.py --flatten SPX_FUT --reason "data_outage"

# FORCE GUARDIAN MODE
python scripts/force_guardian_mode.py --duration 48h

# CHECK CURRENT RISK EXPOSURE
python scripts/risk_snapshot.py
```

### Phase 3 Exit Criteria (for Phase 4 scale-up)
```
□ 90 trading days completed at minimum scale
□ All CCDR structural stops functioning correctly in live conditions
□ No regulatory issues or broker margin calls
□ Sharpe ratio (live) > 0.6 annualised over 90 days
□ Maximum live drawdown < 5% (if exceeded: investigate before proceeding)
□ Guardian mode correctly activated on all D_eff < 3 events
□ CCDR hypothesis tests re-run: still ≥ 7/9 pass on most recent data
□ Compliance sign-off for scale-up
□ Counterparty limits expanded for larger positions
```

---

## Phase 4: Institutional Scale

Scale-up is gated behind Phase 3 exit criteria. Proceed carefully.

```bash
# Increase max_risk_usd in settings.yaml
# Re-run pre-flight check
# Monitor D_eff and portfolio D_eff more frequently during scale-up

python scripts/scale_up.py \
    --new-max-risk-usd 1000000 \
    --ramp-days 30 \
    --confirm
```

---

## Runtime Configuration

### settings.yaml critical parameters
```yaml
pipeline:
  recalc_interval_minutes: 5          # intraday recalc frequency
  l1_recalc_on_vol_shift_vega: 0.5   # immediate L1 recalc on large vol move
  d_eff_rolling_window_days: 60

thresholds:
  d_eff_guardian: 3.0
  d_eff_scout: 5.0
  d_eff_full_hunter: 15.0
  d_eff_stop_floor: 4.0
  gbp_guardian: 0.8
  gbp_hunter_max: 0.5
  gbp_stop_delta: 0.35
  gbp_reentry_max: 0.3
  op_ordered_min: 0.4
  op_disordered_max: 0.15

execution:
  max_risk_usd: 100000              # per position — edit for each phase
  max_concurrent_positions: 6
  max_leverage: 2.0
  drawdown_circuit_breaker_pct: 5.0
  guardian_cooloff_hours: 48

data:
  options:
    provider: livevol           # livevol | bloomberg | ibkr | csv
    symbols: [SPX, NDX, QQQ, SPY, IWM, GLD, TLT, HYG]
    tenors_days: [7, 14, 30, 91, 182, 365, 730]
  analyst:
    provider: ibes
    update_frequency: daily
  surveys:
    aaii_frequency: weekly
    ii_frequency: weekly
  dark_pool:
    provider: finra_ats
    update_frequency: daily
  cross_asset:
    universe: config/universe.yaml   # 27 assets
    frequency: 1min_intraday
```

---

## Monitoring and Alerting

### Key Metrics to Monitor
| Metric | Warning | Critical | Action |
|--------|---------|----------|--------|
| D_eff | < 8.0 | < 5.0 | Scout → Guardian |
| GBP | > 0.5 | > 0.8 | Reduce → Guardian |
| Pipeline latency | > 200ms | > 500ms | Alert ops |
| Options surface staleness | > 1 min | > 5 min | Suspend L1 |
| Portfolio D_eff | < 6.0 | < 4.0 | Reduce positions |
| Rolling 10d drawdown | > 3% | > 5% | Circuit breaker |

```bash
# Start monitoring stack
docker-compose up -d monitoring

# Grafana dashboard: http://localhost:3000
# Prometheus metrics: http://localhost:9090
# Alert manager: http://localhost:9093
```

---

## Disaster Recovery

### Data Feed Failure
```
L1 options feed fails:
  → set psi_shape = UNKNOWN (GBP contribution = 0.60)
  → switch to Scout mode automatically
  → alert ops within 30 seconds

L3 cross-asset feed fails:
  → set d_eff = 5.0 (conservative Scout trigger)
  → do not open new Hunter positions
  → maintain existing positions with tighter GBP stops

L2 analyst data stale > 5 business days:
  → use last known DP value with uncertainty multiplier 1.5
  → alert ops

All feeds fail:
  → Emergency stop: flatten all positions
  → Guardian mode until feeds restored and validated
```

### System Failure
```bash
# State is checkpointed to state/snapshots/ every 5 minutes
# On restart:
python -m psibot.main --restore-from-checkpoint --mode scout
# Scout mode on restart — Hunter mode only after manual review
```

---

## Risk Disclosure (operational reminder)

This system is built on a theoretical physics-inspired framework (CCDR) that has not been validated as a physical law. All trading involves risk of total loss. The system's structural stops, Guardian mode, and D_eff circuit breakers reduce but do not eliminate risk. Review the Limitations section of the ΨBot architecture document before each phase transition.
