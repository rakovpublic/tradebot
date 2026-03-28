"""
state/condensate_state.py — Canonical CondensateState object
=============================================================
The single shared state object passed through all pipeline layers.
All layers read and write to this object in sequence.

CCDR Expectation Field Architecture — Version 1.0
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from helpers import (
    PsiShape, MarketPhase, BotMode, AcousticSignal, TermStructure,
    CCDR_THRESHOLDS,
)


@dataclass
class CondensateState:
    """
    Full CCDR condensate state — the single object shared across all pipeline layers.
    Populated layer by layer: L1 → L2 → L3 → L4 → L5 → mode determination.

    Primary data (from L1 options surface) is the most important.
    Secondary data (L2 analyst/survey) measures condensate phase.
    Tertiary (L3 cross-asset correlations) measures holographic saturation.
    Quaternary (L4 grain boundary synthesis) is the single most important risk number.
    L5 (prices/volume) is acoustic confirmation only — lowest priority.
    """
    timestamp: datetime = field(default_factory=datetime.utcnow)

    # ------------------------------------------------------------------
    # L1 outputs — PRIMARY: ψ_exp wavefunction from options surface
    # ------------------------------------------------------------------
    psi_shape: PsiShape = PsiShape.UNKNOWN
    psi_skew: float = 0.0              # negative=bearish chirality, positive=bullish
    psi_kurtosis_excess: float = 0.0   # grain boundary proximity proxy
    psi_entropy: float = 0.0           # high=disordered condensate
    psi_term_structure: TermStructure = TermStructure.FLAT

    # Raw wavefunction amplitude array (complex, per tenor) — optional for downstream
    psi_amplitude: Optional[object] = None  # np.ndarray when populated

    # ------------------------------------------------------------------
    # L2 outputs — ORDER AND DISORDER PARAMETERS
    # ------------------------------------------------------------------
    order_parameter: float = 0.0       # OP ∈ [-1, +1]: net-bullish coherence
    disorder_parameter: float = 0.0    # DP > 0: analyst forecast dispersion
    phase: MarketPhase = MarketPhase.UNKNOWN

    # Trend signals for GBP computation
    op_trend_5d: float = 0.0           # change in OP over last 5 business days
    dp_trend_5d: float = 0.0           # change in DP over last 5 business days
    dp_trend_10d: float = 0.0          # change in DP over last 10 business days

    # Staleness flag — analyst data has update latency
    analyst_data_stale: bool = False
    analyst_data_age_days: int = 0

    # ------------------------------------------------------------------
    # L3 outputs — HOLOGRAPHIC SATURATION GAUGE
    # ------------------------------------------------------------------
    d_eff: float = 5.0                 # ∈ [1, N=27]: effective dimensionality
    d_eff_trend_10d: float = 0.0       # slope over 10 days; negative = crisis building
    d_eff_trend_20d: float = 0.0       # slope over 20 days for GBP computation
    bot_mode_from_deff: BotMode = BotMode.SCOUT  # mode implied by D_eff alone

    # ------------------------------------------------------------------
    # L4 outputs — GRAIN BOUNDARY PROXIMITY (most important single number)
    # ------------------------------------------------------------------
    gbp: float = 0.5                   # ∈ [0, 1]; 0=deep grain, 1=boundary crossing
    gbp_components: dict = field(default_factory=dict)
    # {'psi': f_psi, 'dp_trend': f_dp, 'deff_trend': f_deff, 'dark_pool': f_dark}

    # ------------------------------------------------------------------
    # L5 outputs — ACOUSTIC CONFIRMATION (lowest priority)
    # ------------------------------------------------------------------
    acoustic_signal: AcousticSignal = AcousticSignal.NEUTRAL
    momentum_20d: float = 0.0          # 20-day price momentum
    momentum_60d: float = 0.0          # 60-day price momentum
    momentum_252d: float = 0.0         # 12-month price momentum (T3 regime detection)
    # T3 regime: <MOM_CRASH_THRESHOLD(-20%) → crash; >MOM_NORMAL_THRESHOLD(+5%) → normal;
    #            in between → gap zone (soliton size halved, fat-tail acoustic=CONTRADICT)
    vrp_regime: str = "normal"         # "high_vol", "low_vol", "normal" — from VRP signal
    vrp_confidence: float = 0.5        # VRP vol-regime confidence ∈ [0, 1]; used for sizing
    vix_term_structure: float = 0.0    # VIX3M − VIX (daily); >0=contango=bullish, <0=backwardation=bearish
    # T8 regime (70.6% 3-month accuracy): VTS<0 (backwardation) → reduce LONG soliton size;
    #   VTS< -5 (panic backwardation) → halve LONG soliton; VTS>0 (contango) → normal
    breadth: float = 0.5               # % of assets above 50d MA
    volume_ratio: float = 1.0          # volume / 20d average volume
    put_call_ratio: float = 1.0        # realised put-call ratio

    # ------------------------------------------------------------------
    # Derived — computed after all layers
    # ------------------------------------------------------------------
    active_mode: BotMode = BotMode.SCOUT
    signal_size_multiplier: float = 0.0

    # Error/failure tracking
    l1_failed: bool = False
    l2_failed: bool = False
    l3_failed: bool = False
    l4_failed: bool = False
    l5_failed: bool = False
    pipeline_errors: list = field(default_factory=list)

    def is_guardian_condition(self) -> bool:
        """Hard Guardian trigger — check before any trading decision."""
        return (self.d_eff <= CCDR_THRESHOLDS["D_EFF_GUARDIAN"]
                or self.gbp >= CCDR_THRESHOLDS["GBP_GUARDIAN"])

    def is_hunter_eligible(self) -> bool:
        """Whether conditions allow Hunter mode (all constraints)."""
        return (self.active_mode == BotMode.HUNTER
                and self.gbp < CCDR_THRESHOLDS["GBP_HUNTER_MAX"]
                and self.d_eff > CCDR_THRESHOLDS["D_EFF_SCOUT"]
                and self.phase != MarketPhase.DISORDERED)

    def summary_dict(self) -> dict:
        """Serialisable summary for logging and monitoring."""
        return {
            "timestamp": self.timestamp.isoformat(),
            "mode": self.active_mode.value,
            "phase": self.phase.value,
            "psi_shape": self.psi_shape.value,
            "psi_skew": round(self.psi_skew, 4),
            "psi_kurtosis_excess": round(self.psi_kurtosis_excess, 4),
            "psi_entropy": round(self.psi_entropy, 4),
            "psi_term_structure": self.psi_term_structure.value,
            "order_parameter": round(self.order_parameter, 4),
            "disorder_parameter": round(self.disorder_parameter, 4),
            "d_eff": round(self.d_eff, 3),
            "d_eff_trend_10d": round(self.d_eff_trend_10d, 4),
            "gbp": round(self.gbp, 4),
            "gbp_components": {k: round(v, 4) for k, v in self.gbp_components.items()},
            "acoustic_signal": self.acoustic_signal.value,
            "momentum_20d": round(self.momentum_20d, 4),
            "momentum_60d": round(self.momentum_60d, 4),
            "momentum_252d": round(self.momentum_252d, 4),
            "vrp_regime": self.vrp_regime,
            "vrp_confidence": round(self.vrp_confidence, 4),
            "vix_term_structure": round(self.vix_term_structure, 4),
            "breadth": round(self.breadth, 4),
            "signal_size_multiplier": round(self.signal_size_multiplier, 4),
            "l1_failed": self.l1_failed,
            "l2_failed": self.l2_failed,
            "l3_failed": self.l3_failed,
            "l4_failed": self.l4_failed,
            "l5_failed": self.l5_failed,
        }
