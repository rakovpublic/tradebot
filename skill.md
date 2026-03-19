# ΨBot Skill Registry
## Claude Code Capability Map

---

## Skill Index

| ID | Skill Name | Layer | Priority | Status |
|----|-----------|-------|----------|--------|
| SK-01 | ψ_exp Wavefunction Reconstruction | L1 | CRITICAL | Implement first |
| SK-02 | Wavefunction Shape Classification | L1 | CRITICAL | Implement first |
| SK-03 | Vol Surface Ingestion & Normalisation | L1 | CRITICAL | Implement first |
| SK-04 | Order Parameter Computation | L2 | HIGH | Phase 1 |
| SK-05 | Disorder Parameter Computation | L2 | HIGH | Phase 1 |
| SK-06 | Condensate Phase Classification | L2 | HIGH | Phase 1 |
| SK-07 | D_eff Eigenvalue Computation | L3 | CRITICAL | Phase 1 |
| SK-08 | D_eff Trend Detection | L3 | HIGH | Phase 1 |
| SK-09 | GBP Score Synthesis | L4 | CRITICAL | Phase 1 |
| SK-10 | Dark Pool Signal Processing | L4 | MEDIUM | Phase 2 |
| SK-11 | Acoustic Confirmation Parser | L5 | LOW | Phase 2 |
| SK-12 | Soliton Signal Generation | Signals | HIGH | Phase 2 |
| SK-13 | Transition Signal Generation | Signals | HIGH | Phase 2 |
| SK-14 | Reorder Signal Generation | Signals | HIGH | Phase 2 |
| SK-15 | Saturation Hedge Generation | Signals | HIGH | Phase 2 |
| SK-16 | Position Sizing Engine | Execution | CRITICAL | Phase 2 |
| SK-17 | CCDR Structural Stop Engine | Execution | CRITICAL | Phase 2 |
| SK-18 | Scout Mode Controller | Modes | HIGH | Phase 2 |
| SK-19 | Hunter Mode Controller | Modes | HIGH | Phase 2 |
| SK-20 | Guardian Mode Controller | Modes | CRITICAL | Phase 2 |
| SK-21 | Hypothesis Backtester T1–T9 | Backtest | HIGH | Phase 0 |
| SK-22 | Broker FIX API Integration | Execution | MEDIUM | Phase 3 |
| SK-23 | Portfolio D_eff Monitor | Risk | HIGH | Phase 3 |
| SK-24 | Drawdown Circuit Breaker | Risk | CRITICAL | Phase 3 |

---

## SK-01: ψ_exp Wavefunction Reconstruction

**What it does:** Converts the raw options implied volatility surface into the expectation field wavefunction ψ_exp(K,T). This is the single most important computation in the entire system.

**Algorithm:**
```
1. Input:  IV surface IV(K, T) for strikes K ∈ [0.5S, 2.0S], tenors T ∈ {1w,2w,1m,3m,6m,1y,2y}
2. Step 1: Arbitrage-free smoothing of IV surface (SVI or SABR parameterisation)
3. Step 2: Dupire local vol: σ_loc²(K,T) = (∂C/∂T) / (0.5K²·∂²C/∂K²)
4. Step 3: Solve Dupire PDE forward to extract risk-neutral density p(S_T)
5. Step 4: ψ_exp(K,T) = √p(K,T) · e^{iθ(K,T)}
           where θ(K,T) is recovered from put-call parity phase constraint
6. Output: ψ_exp array per tenor, complex-valued
```

**Implementation notes:**
- Use GPU (CUDA via cupy or torch) for Dupire PDE — target < 100ms per full surface
- Implement SVI parameterisation for arbitrage-free smoothing before Dupire
- Handle missing strikes by cubic spline interpolation before processing
- Validate: total probability must integrate to 1.0 ± 0.001 at each tenor

**Dependencies:** `numpy`, `scipy.interpolate`, `cupy` or `torch`, `QuantLib-Python` (optional for PDE solver)

**Test:** `tests/test_l1.py::test_psi_reconstruction_integrates_to_unity`

---

## SK-02: Wavefunction Shape Classification

**What it does:** Classifies ψ_exp into one of five shapes, each mapping to a CCDR condensate state.

**Classification rules:**
```python
def classify_psi_shape(p: np.ndarray, strikes: np.ndarray) -> PsiShape:
    """
    p: risk-neutral probability density at a given tenor
    strikes: corresponding strike prices
    """
    # Normalise strikes to moneyness
    K = strikes / strikes[len(strikes)//2]  # ATM normalisation

    # Compute moments
    mean = np.sum(K * p) / np.sum(p)
    variance = np.sum((K - mean)**2 * p) / np.sum(p)
    skewness = np.sum((K - mean)**3 * p) / (np.sum(p) * variance**1.5)
    kurtosis_excess = np.sum((K - mean)**4 * p) / (np.sum(p) * variance**2) - 3.0

    # Bimodality: Hartigan's dip test or bimodality coefficient
    bimodality_coeff = (skewness**2 + 1) / (kurtosis_excess + 3 * (n-1)**2 / ((n-2)*(n-3)))

    # Classification
    if bimodality_coeff > 0.555:       # threshold from Pfister et al. 2013
        return PsiShape.BIMODAL        # grain boundary crossing in progress
    elif kurtosis_excess > 2.0:        # fat tails
        return PsiShape.FAT_TAILED     # approaching holographic saturation
    elif skewness < -0.5:
        return PsiShape.SKEWED_LEFT    # bearish condensate chirality
    elif skewness > 0.5:
        return PsiShape.SKEWED_RIGHT   # bullish condensate chirality
    else:
        return PsiShape.GAUSSIAN       # normal grain interior
```

**Signal mapping:**
| Shape | CCDR Meaning | GBP contribution | Trade implication |
|-------|-------------|-----------------|-------------------|
| Gaussian | Deep grain interior | 0.1 | Ride soliton |
| Skewed Right | Bullish chirality | 0.4 | Soliton-Long candidate |
| Skewed Left | Bearish chirality | 0.4 | Soliton-Short candidate |
| Fat-Tailed | Near holographic saturation | 0.7 | Reduce size, add hedges |
| Bimodal | Grain boundary crossing | 1.0 | Exit trends, go Transition |

---

## SK-03: Vol Surface Ingestion & Normalisation

**Data contracts:**
```python
@dataclass
class OptionsSurface:
    underlying: str                    # e.g. "SPX", "NDX", "AAPL"
    timestamp: datetime
    spot: float
    tenors_days: list[int]             # [7, 14, 30, 91, 182, 365, 730]
    strikes: dict[int, np.ndarray]     # tenor_days → array of strike prices
    iv: dict[int, np.ndarray]          # tenor_days → array of implied vols
    bid_iv: dict[int, np.ndarray]      # for spread quality check
    ask_iv: dict[int, np.ndarray]
    open_interest: dict[int, np.ndarray]

    def validate(self) -> bool:
        """Check: no arbitrage, sufficient strikes, spreads reasonable"""
        ...

    def skew_at_tenor(self, tenor_days: int) -> float:
        """IV(0.9S, T) - IV(1.1S, T) — condensate chirality proxy"""
        ...

    def kurtosis_excess_at_tenor(self, tenor_days: int) -> float:
        """IV(0.8S,T) + IV(1.2S,T) - 2*IV(S,T) — grain boundary proximity proxy"""
        ...
```

**Data sources supported:**
- CBOE LiveVol API (primary for US equities)
- Bloomberg OVDV surface (institutional)
- Interactive Brokers TWS API (retail/paper trading)
- CSV flat file (backtesting)

---

## SK-04 & SK-05: Order and Disorder Parameters

**Order Parameter (OP) — measures condensate coherence direction:**
```python
def compute_order_parameter(survey_data: SurveyData) -> float:
    """
    OP = (% bullish - % bearish) / 100
    Sources: AAII, Investors Intelligence, institutional surveys
    OP ∈ [-1, +1]
    """
    # Weighted average across sources
    weights = {'AAII': 0.3, 'II': 0.4, 'institutional': 0.3}
    ops = {
        'AAII': (survey_data.aaii_bull - survey_data.aaii_bear) / 100,
        'II': (survey_data.ii_bull - survey_data.ii_bear) / 100,
        'institutional': (survey_data.inst_bull - survey_data.inst_bear) / 100,
    }
    return sum(weights[k] * ops[k] for k in weights)
```

**Disorder Parameter (DP) — measures condensate disorder:**
```python
def compute_disorder_parameter(analyst_data: AnalystData) -> float:
    """
    DP = σ(EPS forecasts) / |mean(EPS forecast)|
    Normalised cross-sectional dispersion of 12-month forward EPS estimates
    DP > 0, rising DP = grain boundary approach
    """
    eps_forecasts = analyst_data.forward_12m_eps_by_analyst
    mean_eps = np.mean(eps_forecasts)
    if abs(mean_eps) < 1e-6:
        return np.nan
    return np.std(eps_forecasts) / abs(mean_eps)
```

---

## SK-07: D_eff Eigenvalue Computation

**This is the most critical risk management signal in the system.**

```python
def compute_d_eff(returns_matrix: np.ndarray) -> float:
    """
    D_eff = -log(Σ λᵢ²) / log(N)

    Args:
        returns_matrix: shape (window_days, N_assets) — 60-day rolling window
                        N_assets = 27 (see universe.yaml)
    Returns:
        D_eff ∈ [1, N], where:
          D_eff = 1:  crisis (all assets fully correlated)
          D_eff = N:  fully independent (normal diversified markets)
    """
    N = returns_matrix.shape[1]
    corr_matrix = np.corrcoef(returns_matrix.T)  # (N, N) correlation matrix

    # Eigenvalue decomposition
    eigenvalues = np.linalg.eigvalsh(corr_matrix)
    eigenvalues = eigenvalues / eigenvalues.sum()  # normalise to sum=1

    # D_eff formula
    sum_sq = np.sum(eigenvalues**2)
    d_eff = -np.log(sum_sq) / np.log(N)

    return float(np.clip(d_eff, 1.0, float(N)))
```

**27-Asset Universe (see `config/universe.yaml`):**
```
Equities (6):    SPX, NDX, RUT, SXXP, NKY, MSCIEM
Fixed Income (5): US2Y, US10Y, US30Y, BUND10Y, JGB10Y
Credit (4):      CDXIG, CDXHY, ITRAXXMAIN, ITRAXXCO
FX (5):          DXY, EURUSD, USDJPY, GBPUSD, AUDUSD
Commodities (4): GOLD, CRUDE, COPPER, WHEAT
Vol (3):         VIX, VVIX, MOVE
```

---

## SK-09: GBP Score Synthesis

```python
def compute_gbp(
    psi_shape: PsiShape,
    dp_trend: float,           # rate of change of DP over 10 days
    d_eff_trend: float,        # slope of D_eff over 20 days
    dark_pool_fraction: float, # current vs 90-day rolling average
) -> tuple[float, dict]:
    """
    GBP = w1*f(ψ_shape) + w2*f(DP_trend) + w3*f(D_eff_trend) + w4*f(dark_pool)
    GBP ∈ [0, 1]: 0=deep grain, 1=boundary crossing
    """
    WEIGHTS = {'psi': 0.35, 'dp': 0.25, 'deff': 0.30, 'darkpool': 0.10}

    # f(ψ_shape)
    shape_map = {
        PsiShape.GAUSSIAN:      0.1,
        PsiShape.SKEWED_RIGHT:  0.4,
        PsiShape.SKEWED_LEFT:   0.4,
        PsiShape.FAT_TAILED:    0.7,
        PsiShape.BIMODAL:       1.0,
        PsiShape.UNKNOWN:       0.6,  # conservative on data failure
    }
    f_psi = shape_map[psi_shape]

    # f(DP_trend): normalised 0→1 based on 90th percentile of historical trend
    f_dp = float(np.clip(dp_trend / DP_TREND_90TH_PERCENTILE, 0, 1))

    # f(D_eff_trend): negative slope means declining D_eff (crisis building)
    # normalise: -1.0/day slope → f=1.0
    f_deff = float(np.clip(-d_eff_trend / 1.0, 0, 1))

    # f(dark_pool): high relative dark pool = expectations refusing acoustic coupling
    f_darkpool = float(np.clip(dark_pool_fraction - 1.0, 0, 1))

    gbp = (WEIGHTS['psi'] * f_psi +
           WEIGHTS['dp'] * f_dp +
           WEIGHTS['deff'] * f_deff +
           WEIGHTS['darkpool'] * f_darkpool)

    components = {'psi': f_psi, 'dp': f_dp, 'deff': f_deff, 'darkpool': f_darkpool}
    return float(np.clip(gbp, 0, 1)), components
```

---

## SK-12: Soliton Signal Generation

```python
def check_soliton_signal(state: CondensateState, portfolio: PortfolioState) -> Optional[Signal]:
    """
    Soliton = topological momentum in ordered condensate.
    Entry: condensate ordered, chirality defined, GBP low, L5 confirms.
    Exit: GBP stop, D_eff stop, ψ turns bimodal.
    """
    # Entry conditions
    if state.active_mode != BotMode.HUNTER:
        return None
    if state.gbp > 0.3:
        return None
    if state.phase not in [MarketPhase.ORDERED_BULL, MarketPhase.ORDERED_BEAR]:
        return None
    if state.psi_shape not in [PsiShape.SKEWED_RIGHT, PsiShape.SKEWED_LEFT]:
        return None

    direction = (SignalDirection.LONG
                 if state.psi_shape == PsiShape.SKEWED_RIGHT
                 else SignalDirection.SHORT)

    # Require L5 confirmation for full size; proceed at 70% without
    size_mult = state.signal_size_multiplier
    if state.acoustic_signal == AcousticSignal.CONTRADICT:
        size_mult *= 0.7

    return Signal(
        signal_class=SignalClass.SOLITON,
        direction=direction,
        size_multiplier=size_mult,
        entry_gbp=state.gbp,
        entry_phase=state.phase,
        stop_conditions=STOPS,
        instruments=['SPX_FUT', 'NDX_FUT'],  # from universe
        rationale=f"Soliton-{direction.value}: ψ={state.psi_shape.value}, "
                  f"OP={state.order_parameter:.2f}, GBP={state.gbp:.2f}",
    )
```

---

## SK-21: Hypothesis Backtester T1–T9

All nine CCDR predictions must pass before live deployment. Implement as independent pytest fixtures that can be run as a validation suite.

```python
# backtesting/hypothesis_tests.py

class CCDRHypothesisTests:
    """
    Run all 9 CCDR market predictions as statistical tests.
    All must pass (see pass criteria in claude.md) before Phase 3 deployment.
    """

    def test_T1_vol_surface_granger_causes_price(self, data: HistoricalData):
        """T1: Vol surface changes Granger-cause price changes, not vice versa."""
        # Granger causality test at lags 1–10 days
        # Pass: F-stat p < 0.01 for vol→price direction

    def test_T2_analyst_dispersion_leads_regime(self, data: HistoricalData):
        """T2: Analyst dispersion peaks precede regime changes by weeks."""
        # Cross-correlate DP peaks with NBER recession dates
        # Pass: median lead time 2–8 months, p < 0.05

    def test_T3_momentum_crashes_bimodal(self, data: HistoricalData):
        """T3: Momentum crash drawdown rates are bimodally distributed."""
        # Hartigan's dip test on monthly momentum strategy drawdown rates
        # Pass: dip test p < 0.05 (reject unimodal null)

    def test_T4_deff_leads_crashes(self, data: HistoricalData):
        """T4: D_eff declines 30–60 days before market crashes."""
        # Identify all VIX > 40 events; measure D_eff 60 days prior
        # Pass: median lead time 25+ days, direction correct > 70%

    def test_T5_dark_pool_predicts_direction(self, data: HistoricalData):
        """T5: Dark pool fraction at t predicts price direction at t+1d."""
        # FINRA ATS data vs next-day returns
        # Pass: directional accuracy > 55% with p < 0.05

    def test_T6_equity_premium_spectral_peak(self, data: HistoricalData):
        """T6: Equity risk premium has 3–7yr spectral peak."""
        # Spectral analysis of rolling 12m excess equity returns (Shiller data)
        # Pass: dominant spectral frequency in 3–7yr band, p < 0.05

    def test_T7_technical_levels_survive_turnover(self, data: HistoricalData):
        """T7: Technical levels persist despite participant turnover."""
        # Pre-2005 pivot levels; test persistence through 2010 HFT transition
        # Pass: > 70% of identified levels still operative post-transition

    def test_T8_vol_skew_predicts_regime(self, data: HistoricalData):
        """T8: Vol skew sign predicts direction of next regime change."""
        # Skew at month t vs subsequent bull/bear regime transition
        # Pass: directional accuracy > 60%, p < 0.05

    def test_T9_earnings_drift_proportional_dispersion(self, data: HistoricalData):
        """T9: Post-earnings drift ∝ analyst forecast dispersion."""
        # Cross-sectional regression: drift magnitude ~ f(EPS dispersion)
        # Pass: R² > 0.20, coefficient positive and significant

    def run_all(self, data: HistoricalData) -> HypothesisReport:
        """Run all tests. Return pass/fail per test + deployment recommendation."""
        results = {}
        for test_id in ['T1','T2','T3','T4','T5','T6','T7','T8','T9']:
            method = getattr(self, f'test_{test_id}_')
            results[test_id] = method(data)

        passed = sum(1 for r in results.values() if r.passed)
        deploy_recommended = passed >= 7  # at least 7 of 9 must pass

        return HypothesisReport(results=results, deploy_recommended=deploy_recommended,
                                 blocking_failures=[t for t,r in results.items() if not r.passed][:3])
```

---

## Skill Dependencies

```
SK-01 (ψ reconstruction)
  └── SK-02 (shape classification)
        └── SK-09 (GBP — uses f_psi)
              └── SK-16 (position sizing — uses GBP)
                    └── SK-12..15 (signal generation — uses size)
                          └── SK-18..20 (modes — orchestrate signals)

SK-07 (D_eff)
  └── SK-08 (D_eff trend)
        └── SK-09 (GBP — uses f_deff)
              └── SK-20 (Guardian — triggered by D_eff)

SK-04+05 (OP+DP)
  └── SK-06 (phase classification)
        └── SK-09 (GBP — uses f_dp)

SK-21 (backtesting) — runs independently; must pass before SK-22 (broker)
```

---

## Implementation Priority Queue

```
Week 1–2:   SK-07, SK-03 (D_eff + vol surface ingestion — critical infrastructure)
Week 3–4:   SK-01, SK-02 (ψ_exp reconstruction — core sensor)
Week 5–6:   SK-04, SK-05, SK-06 (phase detection)
Week 7–8:   SK-09 (GBP synthesis — pulls all L1–L4 together)
Week 9–10:  SK-21 (backtesting — validate T1–T9 before any signals)
Week 11–12: SK-12..15 (signal classes), SK-16, SK-17 (sizing + stops)
Week 13–14: SK-18..20 (modes), SK-11 (L5 acoustic — last, lowest priority)
Week 15–16: SK-22..24 (broker API, portfolio D_eff, circuit breaker)
```
