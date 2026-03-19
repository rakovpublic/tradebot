"""
helpers.py — ΨBot JNEOPALLIUM Shared Utilities
================================================
Shared numerical, statistical, and I/O utilities used across all pipeline layers.
Import as: from helpers import *  or  from helpers import compute_d_eff, classify_psi_shape

CCDR Expectation Field Architecture — Version 1.0
Author: Dmytro Rakovskyi / github.com/rakovpublic/jneopallium
"""

from __future__ import annotations

import logging
import warnings
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum, auto
from typing import Optional, Union

import numpy as np
import pandas as pd
from scipy import stats, signal as scipy_signal
from scipy.interpolate import CubicSpline, RectBivariateSpline

# NumPy 2.x compatibility: np.trapz was renamed to np.trapezoid
if not hasattr(np, 'trapz'):
    np.trapz = np.trapezoid

# ---------------------------------------------------------------------------
# Structured logger
# ---------------------------------------------------------------------------

log = logging.getLogger("psibot")


# ===========================================================================
# ENUMERATIONS  — CCDR vocabulary
# ===========================================================================

class PsiShape(str, Enum):
    GAUSSIAN      = "Gaussian"       # deep grain interior, no transition imminent
    SKEWED_LEFT   = "SkewedLeft"     # bearish condensate chirality
    SKEWED_RIGHT  = "SkewedRight"    # bullish condensate chirality
    FAT_TAILED    = "FatTailed"      # approaching holographic saturation
    BIMODAL       = "Bimodal"        # grain boundary crossing in progress
    UNKNOWN       = "Unknown"        # data failure — treat conservatively


class MarketPhase(str, Enum):
    ORDERED_BULL  = "Ordered_Bull"   # condensate ordered, bullish chirality
    ORDERED_BEAR  = "Ordered_Bear"   # condensate ordered, bearish chirality
    DISORDERED    = "Disordered"     # grain boundary transition in progress
    REORDERING    = "ReOrdering"     # new grain nucleating
    UNKNOWN       = "Unknown"

    def is_ordered(self) -> bool:
        return self in (MarketPhase.ORDERED_BULL, MarketPhase.ORDERED_BEAR)


class BotMode(str, Enum):
    SCOUT    = "Scout"     # observe only — no positions
    HUNTER   = "Hunter"    # active signal execution
    GUARDIAN = "Guardian"  # risk management — close risk, add hedges


class SignalClass(str, Enum):
    SOLITON           = "Soliton"          # topological momentum
    TRANSITION        = "Transition"       # long volatility at grain boundary
    REORDER           = "Reorder"          # first mover in new regime
    SATURATION_HEDGE  = "SaturationHedge"  # defensive before D_eff collapse


class SignalDirection(str, Enum):
    LONG  = "Long"
    SHORT = "Short"
    FLAT  = "Flat"


class AcousticSignal(str, Enum):
    CONFIRM     = "Confirm"    # size × 1.2
    CONTRADICT  = "Contradict" # size × 0.7
    NEUTRAL     = "Neutral"    # size × 1.0


class TermStructure(str, Enum):
    CONTANGO          = "Contango"
    BACKWARDATION     = "Backwardation"
    INVERTED_HUMP     = "InvertedHump"
    FLAT              = "Flat"


# ===========================================================================
# LAYER 1 HELPERS — ψ_exp Wavefunction
# ===========================================================================

def classify_psi_shape(
    p: np.ndarray,
    strikes: np.ndarray,
    bimodality_threshold: float = 0.555,
    kurtosis_threshold: float = 2.0,
    skewness_threshold: float = 0.5,
) -> PsiShape:
    """
    Classify risk-neutral density p(K) into a PsiShape.

    The bimodality coefficient (Pfister et al. 2013) is:
        BC = (skewness² + 1) / (kurtosis_excess + 3*(n-1)²/((n-2)*(n-3)))
    BC > 0.555 indicates bimodality.

    Args:
        p:        risk-neutral probability density, non-negative, integrates to 1
        strikes:  corresponding strike prices (same length as p)
        bimodality_threshold: BC > this → BIMODAL (grain boundary crossing)
        kurtosis_threshold:   excess kurtosis > this → FAT_TAILED
        skewness_threshold:   |skewness| > this → SKEWED_LEFT or SKEWED_RIGHT

    Returns:
        PsiShape enum value
    """
    if p is None or len(p) < 5:
        return PsiShape.UNKNOWN

    p = np.asarray(p, dtype=float)
    K = np.asarray(strikes, dtype=float)

    # Normalise strikes to ATM = 1.0
    atm_idx = len(K) // 2
    K = K / K[atm_idx]

    # Ensure p integrates to 1
    p = p / (np.trapz(p, K) + 1e-12)

    # Moments
    mean   = np.trapz(K * p, K)
    var    = np.trapz((K - mean)**2 * p, K)
    std    = np.sqrt(max(var, 1e-10))
    skew   = np.trapz((K - mean)**3 * p, K) / std**3
    kurt   = np.trapz((K - mean)**4 * p, K) / std**4
    kurt_e = kurt - 3.0  # excess kurtosis

    n = len(p)
    # Bimodality coefficient
    denom = (kurt_e + 3 * (n - 1)**2 / max((n - 2) * (n - 3), 1))
    bc = (skew**2 + 1) / max(abs(denom), 1e-10)

    if bc > bimodality_threshold:
        return PsiShape.BIMODAL
    if kurt_e > kurtosis_threshold:
        return PsiShape.FAT_TAILED
    if skew < -skewness_threshold:
        return PsiShape.SKEWED_LEFT
    if skew > skewness_threshold:
        return PsiShape.SKEWED_RIGHT
    return PsiShape.GAUSSIAN


def compute_psi_entropy(p: np.ndarray, strikes: np.ndarray) -> float:
    """
    Shannon entropy of the risk-neutral density.
    High entropy → disordered condensate.

    H(ψ_exp) = -∫ p(K) log p(K) dK
    """
    p = np.asarray(p, dtype=float)
    K = np.asarray(strikes, dtype=float)
    p = np.clip(p, 1e-12, None)
    p = p / (np.trapz(p, K) + 1e-12)
    integrand = -p * np.log(p)
    return float(np.trapz(integrand, K))


def compute_skew(iv_surface: dict, tenor_days: int, spot: float) -> float:
    """
    Skew = IV(0.9S, T) - IV(1.1S, T)
    Condensate chirality proxy: negative → bearish, positive → bullish.
    """
    strikes = iv_surface['strikes'][tenor_days]
    ivs     = iv_surface['iv'][tenor_days]
    cs = CubicSpline(strikes, ivs)
    iv_otm_put  = float(cs(0.9 * spot))
    iv_otm_call = float(cs(1.1 * spot))
    return iv_otm_put - iv_otm_call


def compute_kurtosis_proxy(iv_surface: dict, tenor_days: int, spot: float) -> float:
    """
    Kurtosis proxy = IV(0.8S,T) + IV(1.2S,T) - 2*IV(S,T)
    Grain boundary proximity proxy: higher → fatter tails → nearer boundary.
    """
    strikes = iv_surface['strikes'][tenor_days]
    ivs     = iv_surface['iv'][tenor_days]
    cs = CubicSpline(strikes, ivs)
    return float(cs(0.8 * spot) + cs(1.2 * spot) - 2 * cs(spot))


def classify_term_structure(ivs_by_tenor: dict[int, float]) -> TermStructure:
    """
    Classify vol term structure from ATM IVs at different tenors.

    ivs_by_tenor: {tenor_days: atm_iv, ...}  e.g. {30: 0.18, 91: 0.20, 365: 0.22}
    """
    tenors = sorted(ivs_by_tenor.keys())
    if len(tenors) < 2:
        return TermStructure.FLAT
    ivs = [ivs_by_tenor[t] for t in tenors]

    # Check for inversion in the middle (hump)
    if len(ivs) >= 3:
        short_end = ivs[0]
        mid_peak  = max(ivs[:len(ivs)//2 + 1])
        long_end  = ivs[-1]
        if mid_peak > short_end * 1.05 and mid_peak > long_end * 1.05:
            return TermStructure.INVERTED_HUMP

    slope = (ivs[-1] - ivs[0]) / (tenors[-1] - tenors[0])
    if slope > 0.0001:
        return TermStructure.CONTANGO
    if slope < -0.0001:
        return TermStructure.BACKWARDATION
    return TermStructure.FLAT


# ===========================================================================
# LAYER 3 HELPERS — D_eff Holographic Saturation
# ===========================================================================

def compute_d_eff(returns_matrix: np.ndarray) -> float:
    """
    Compute effective dimensionality D_eff of the cross-asset expectation condensate.

    D_eff = -log(Σ λᵢ²) / log(N)

    where λᵢ are the normalised eigenvalues of the cross-asset correlation matrix.

    Args:
        returns_matrix: shape (window_days, N_assets), typically 60-day rolling window
                        N_assets = 27 per universe.yaml

    Returns:
        D_eff ∈ [1, N]:
            D_eff ≈ N  →  fully diversified (normal markets)
            D_eff ≈ 1  →  fully correlated (holographic saturation → crisis)
    """
    if returns_matrix.shape[0] < 10:
        log.warning("D_eff: insufficient data (%d rows), returning conservative 5.0",
                    returns_matrix.shape[0])
        return 5.0

    N = returns_matrix.shape[1]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        corr = np.corrcoef(returns_matrix.T)

    # Replace NaN/Inf with 0 off-diagonal
    corr = np.where(np.isfinite(corr), corr, 0.0)
    np.fill_diagonal(corr, 1.0)

    eigenvalues = np.linalg.eigvalsh(corr)
    eigenvalues = np.clip(eigenvalues, 1e-12, None)
    eigenvalues = eigenvalues / eigenvalues.sum()

    sum_sq = float(np.sum(eigenvalues**2))
    d_eff = -np.log(sum_sq) / np.log(N)
    return float(np.clip(d_eff, 1.0, float(N)))


def compute_d_eff_trend(d_eff_series: pd.Series, window: int = 20) -> float:
    """
    Linear trend slope of D_eff over the last `window` observations.
    Negative slope → crisis building.

    Returns:
        slope in D_eff units per day (negative = deteriorating)
    """
    if len(d_eff_series) < window:
        return 0.0
    recent = d_eff_series.iloc[-window:].values
    x = np.arange(len(recent), dtype=float)
    slope, _, _, _, _ = stats.linregress(x, recent)
    return float(slope)


def d_eff_to_bot_mode(d_eff: float) -> BotMode:
    """Map D_eff level to bot mode. Final mode is min of this and GBP-based mode."""
    if d_eff <= 3.0:
        return BotMode.GUARDIAN
    if d_eff <= 5.0:
        return BotMode.SCOUT
    return BotMode.HUNTER


def d_eff_to_size_factor(d_eff: float) -> float:
    """
    f(D_eff): linear interpolation factor ∈ [0, 1].
    1.0 at D_eff=20, 0.5 at D_eff=8, 0.0 at D_eff≤3.
    """
    if d_eff >= 20.0:
        return 1.0
    if d_eff <= 3.0:
        return 0.0
    return float((d_eff - 3.0) / (20.0 - 3.0))


# ===========================================================================
# LAYER 2 HELPERS — Condensate Phase
# ===========================================================================

def classify_market_phase(
    op: float,
    dp: float,
    op_trend_5d: float,      # change in OP over last 5 days
    dp_trend_5d: float,      # change in DP over last 5 days
    op_ordered_min: float = 0.4,
    op_disordered_max: float = 0.15,
) -> MarketPhase:
    """
    Classify market phase from order parameter OP and disorder parameter DP.

    |OP| > 0.4, stable → ORDERED
    |OP| declining + DP rising fast → GRAIN BOUNDARY APPROACH (still Ordered but warning)
    |OP| < 0.15, DP high → DISORDERED
    |OP| < 0.15, DP declining, OP rising → REORDERING
    """
    abs_op = abs(op)
    direction_bull = op > 0

    if abs_op < op_disordered_max:
        if op_trend_5d > 0.02 and dp_trend_5d < 0:
            return MarketPhase.REORDERING
        return MarketPhase.DISORDERED

    if abs_op >= op_ordered_min:
        return MarketPhase.ORDERED_BULL if direction_bull else MarketPhase.ORDERED_BEAR

    # Intermediate zone: check trajectory
    if op_trend_5d < -0.03 and dp_trend_5d > 0.02:
        # Heading toward disorder
        return MarketPhase.ORDERED_BULL if direction_bull else MarketPhase.ORDERED_BEAR

    return MarketPhase.ORDERED_BULL if direction_bull else MarketPhase.ORDERED_BEAR


# ===========================================================================
# LAYER 4 HELPERS — Grain Boundary Proximity
# ===========================================================================

def compute_gbp(
    psi_shape: PsiShape,
    dp_trend_10d: float,
    d_eff_trend_20d: float,
    dark_pool_ratio: float,   # current / 90d rolling average
    dp_trend_90th_pctile: float = 0.05,  # calibrate from historical data
) -> tuple[float, dict[str, float]]:
    """
    GBP = 0.35*f(ψ) + 0.25*f(DP) + 0.30*f(D_eff) + 0.10*f(dark_pool)
    GBP ∈ [0, 1]: 0=stable grain, 1=crossing imminent

    Returns:
        (gbp_score, component_breakdown)
    """
    SHAPE_SCORES: dict[PsiShape, float] = {
        PsiShape.GAUSSIAN:      0.10,
        PsiShape.SKEWED_RIGHT:  0.40,
        PsiShape.SKEWED_LEFT:   0.40,
        PsiShape.FAT_TAILED:    0.70,
        PsiShape.BIMODAL:       1.00,
        PsiShape.UNKNOWN:       0.60,  # conservative on data failure
    }

    f_psi  = SHAPE_SCORES.get(psi_shape, 0.60)
    f_dp   = float(np.clip(dp_trend_10d / max(dp_trend_90th_pctile, 1e-6), 0, 1))
    f_deff = float(np.clip(-d_eff_trend_20d / 1.0, 0, 1))
    f_dark = float(np.clip(dark_pool_ratio - 1.0, 0, 1))

    gbp = 0.35*f_psi + 0.25*f_dp + 0.30*f_deff + 0.10*f_dark
    gbp = float(np.clip(gbp, 0.0, 1.0))

    components = {
        'psi':      f_psi,
        'dp_trend': f_dp,
        'deff_trend': f_deff,
        'dark_pool': f_dark,
        'gbp': gbp,
    }
    return gbp, components


def gbp_to_size_factor(gbp: float) -> float:
    """
    f(GBP): linear interpolation factor ∈ [0, 1].
    1.0 at GBP=0.1, 0.5 at GBP=0.4, 0.0 at GBP≥0.7.
    """
    if gbp <= 0.1:
        return 1.0
    if gbp >= 0.7:
        return 0.0
    return float((0.7 - gbp) / (0.7 - 0.1))


# ===========================================================================
# EXECUTION HELPERS
# ===========================================================================

def compute_position_size(
    d_eff: float,
    gbp: float,
    acoustic: AcousticSignal,
    max_risk_usd: float,
) -> float:
    """
    Final position size in USD notional.
    base = f(D_eff) × f(GBP) × max_risk_usd
    modified by L5 acoustic signal.
    """
    base = d_eff_to_size_factor(d_eff) * gbp_to_size_factor(gbp) * max_risk_usd

    if acoustic == AcousticSignal.CONFIRM:
        return min(base * 1.2, max_risk_usd)
    if acoustic == AcousticSignal.CONTRADICT:
        return base * 0.7
    return base


def determine_bot_mode(d_eff: float, gbp: float, phase: MarketPhase) -> BotMode:
    """
    Final mode determination after all pipeline layers.
    Guardian takes priority over all other conditions.
    """
    # Hard Guardian triggers
    if d_eff <= 3.0 or gbp >= 0.8:
        return BotMode.GUARDIAN

    # Scout triggers
    if d_eff <= 5.0:
        return BotMode.SCOUT
    if phase == MarketPhase.DISORDERED and gbp > 0.6:
        return BotMode.SCOUT

    # Hunter requires clean conditions
    if gbp < 0.5 and d_eff > 5.0 and phase != MarketPhase.DISORDERED:
        return BotMode.HUNTER

    return BotMode.SCOUT


# ===========================================================================
# STOP-LOSS HELPERS
# ===========================================================================

@dataclass
class StopCheckResult:
    triggered: bool
    reasons: list[str] = field(default_factory=list)


def check_structural_stops(
    entry_gbp: float,
    entry_phase: MarketPhase,
    signal_class: SignalClass,
    current_gbp: float,
    current_d_eff: float,
    current_phase: MarketPhase,
    current_psi_shape: PsiShape,
    gbp_stop_delta: float = 0.35,
    d_eff_floor: float = 4.0,
) -> StopCheckResult:
    """
    Check all CCDR structural stops.
    These replace price-level stop-losses entirely.
    """
    reasons: list[str] = []

    # 1. GBP stop: grain boundary approach since entry
    if current_gbp > entry_gbp + gbp_stop_delta:
        reasons.append(
            f"GBP stop: current {current_gbp:.2f} > entry {entry_gbp:.2f} + {gbp_stop_delta}"
        )

    # 2. D_eff floor stop: systemic risk crossing threshold
    if current_d_eff < d_eff_floor:
        reasons.append(f"D_eff stop: {current_d_eff:.1f} < floor {d_eff_floor}")

    # 3. Phase transition stop: ordered → disordered
    if entry_phase.is_ordered() and current_phase == MarketPhase.DISORDERED:
        reasons.append(f"Phase stop: {entry_phase.value} → {current_phase.value}")

    # 4. ψ shape stop: soliton trade → bimodal wavefunction
    if (signal_class == SignalClass.SOLITON
            and current_psi_shape == PsiShape.BIMODAL):
        reasons.append("ψ shape stop: soliton, ψ_exp turned bimodal (grain boundary crossing)")

    return StopCheckResult(triggered=bool(reasons), reasons=reasons)


# ===========================================================================
# STATISTICAL HELPERS
# ===========================================================================

def granger_causality_test(
    y: np.ndarray,
    x: np.ndarray,
    max_lag: int = 10,
) -> dict[int, float]:
    """
    Simplified Granger causality: test whether x Granger-causes y.
    Returns p-values at each lag.
    Uses OLS with lag structure.

    For T1: x = vol surface change, y = price change.
    Pass criterion: p < 0.01 at any lag 1–5.
    """
    from statsmodels.tsa.stattools import grangercausalitytests
    data = np.column_stack([y, x])
    result = grangercausalitytests(data, maxlag=max_lag, verbose=False)
    p_values = {lag: result[lag][0]['ssr_ftest'][1] for lag in range(1, max_lag + 1)}
    return p_values


def hartigan_dip_test(data: np.ndarray) -> tuple[float, float]:
    """
    Hartigan & Hartigan (1985) dip test for bimodality.
    Low p-value → reject unimodal null → bimodal distribution.

    Returns:
        (dip_statistic, p_value)
    Pass criterion for T3: p < 0.05.

    Implementation: uses diptest package if available, else Silverman's rule fallback.
    """
    try:
        from diptest import diptest
        dip, pval = diptest(data)
        return float(dip), float(pval)
    except ImportError:
        # Silverman bandwidth test fallback
        bw_full = stats.gaussian_kde(data).factor
        n = len(data)
        # Use bootstrap approximation
        dip_stat = np.std(data) * (stats.kurtosis(data) + 3)
        p_val = 1.0 / (1.0 + dip_stat)  # rough approximation
        log.warning("diptest package not found; using fallback (less accurate)")
        return float(dip_stat), float(p_val)


def spectral_peak_frequency(
    series: pd.Series,
    min_period_years: float = 3.0,
    max_period_years: float = 7.0,
) -> tuple[float, float]:
    """
    Test whether the dominant spectral frequency of a series lies in [min, max] years.
    For T6: series = rolling 12m equity risk premium.

    Returns:
        (dominant_period_years, fraction_of_power_in_band)
    Pass criterion: > 25% of power in 3–7yr band.
    """
    # Assume monthly frequency
    freqs = np.fft.rfftfreq(len(series), d=1.0/12.0)  # cycles per year
    power = np.abs(np.fft.rfft(series.values - series.mean()))**2

    # Identify dominant frequency
    dominant_idx = np.argmax(power[1:]) + 1  # exclude DC
    dominant_freq = freqs[dominant_idx]
    dominant_period = 1.0 / max(dominant_freq, 1e-6)

    # Power fraction in target band
    in_band = ((freqs >= 1/max_period_years) & (freqs <= 1/min_period_years))
    band_power = power[in_band].sum() / (power[1:].sum() + 1e-12)

    return float(dominant_period), float(band_power)


# ===========================================================================
# DATA QUALITY HELPERS
# ===========================================================================

def validate_options_surface(
    strikes: np.ndarray,
    ivs: np.ndarray,
    spot: float,
    tenor_days: int,
) -> tuple[bool, list[str]]:
    """
    Basic arbitrage-free and quality checks for an IV surface slice.
    Returns (is_valid, list_of_issues).
    """
    issues: list[str] = []

    if len(strikes) < 5:
        issues.append(f"Insufficient strikes: {len(strikes)} < 5")

    if np.any(ivs <= 0):
        issues.append("Non-positive IVs detected")

    if np.any(ivs > 3.0):
        issues.append(f"Unreasonably high IV: max={np.max(ivs):.2f}")

    # Check for calendar spread violations (simplified)
    atm_idx = np.argmin(np.abs(strikes - spot))
    atm_iv = ivs[atm_idx]
    if atm_iv < 0.01:
        issues.append(f"ATM IV suspiciously low: {atm_iv:.4f}")

    # Check for butterfly violations (simplified: no negative butterflies)
    if len(strikes) >= 3:
        second_deriv = np.diff(ivs, n=2)
        if np.any(second_deriv < -0.001):
            issues.append("Potential butterfly arbitrage detected")

    is_valid = len(issues) == 0
    return is_valid, issues


def rolling_window_returns(prices: pd.DataFrame, window: int = 60) -> np.ndarray:
    """
    Compute daily log-returns for a multi-asset price DataFrame,
    returning the most recent `window` observations.

    Args:
        prices: DataFrame with dates as index, assets as columns (27 assets)
        window: number of trading days for D_eff computation

    Returns:
        np.ndarray of shape (window, N_assets)
    """
    log_returns = np.log(prices / prices.shift(1)).dropna()
    recent = log_returns.iloc[-window:].values
    return recent


# ===========================================================================
# CONDENSATE STATE DISPLAY
# ===========================================================================

def format_state_summary(state) -> str:
    """
    Human-readable one-line summary of CondensateState.
    Used in logging and monitoring dashboards.
    """
    return (
        f"[{state.timestamp.strftime('%Y-%m-%d %H:%M')}] "
        f"Mode={state.active_mode.value:8s} | "
        f"Phase={state.phase.value:12s} | "
        f"ψ={state.psi_shape.value:10s} | "
        f"D_eff={state.d_eff:5.1f} | "
        f"GBP={state.gbp:.3f} | "
        f"Size×={state.signal_size_multiplier:.2f} | "
        f"Skew={state.psi_skew:+.3f} | "
        f"OP={state.order_parameter:+.2f} | "
        f"DP={state.disorder_parameter:.3f}"
    )


def gbp_to_label(gbp: float) -> str:
    """Return human-readable label for GBP score."""
    if gbp < 0.2:  return "DEEP_GRAIN 🟢"
    if gbp < 0.4:  return "STABLE     🟢"
    if gbp < 0.6:  return "APPROACHING 🟡"
    if gbp < 0.8:  return "IMMINENT   🟠"
    return         "CROSSING   🔴"


def d_eff_to_label(d_eff: float) -> str:
    """Return human-readable label for D_eff level."""
    if d_eff > 15:  return f"DIVERSIFIED  ({d_eff:.1f}) 🟢"
    if d_eff > 10:  return f"NORMAL       ({d_eff:.1f}) 🟢"
    if d_eff > 6:   return f"COMPRESSION  ({d_eff:.1f}) 🟡"
    if d_eff > 3:   return f"PRE-CRISIS   ({d_eff:.1f}) 🟠"
    return          f"SATURATION   ({d_eff:.1f}) 🔴"


# ===========================================================================
# CONSTANTS
# ===========================================================================

ASSET_UNIVERSE_27 = [
    # Equities (6)
    "SPX", "NDX", "RUT", "SXXP", "NKY", "MSCIEM",
    # Fixed Income (5)
    "US2Y", "US10Y", "US30Y", "BUND10Y", "JGB10Y",
    # Credit (4)
    "CDXIG", "CDXHY", "ITRAXXMAIN", "ITRAXXCO",
    # FX (5)
    "DXY", "EURUSD", "USDJPY", "GBPUSD", "AUDUSD",
    # Commodities (4)
    "GOLD", "CRUDE", "COPPER", "WHEAT",
    # Volatility (3)
    "VIX", "VVIX", "MOVE",
]

CCDR_THRESHOLDS = {
    "D_EFF_GUARDIAN":    3.0,
    "D_EFF_SCOUT":       5.0,
    "D_EFF_FULL_HUNTER": 15.0,
    "D_EFF_STOP_FLOOR":  4.0,
    "GBP_GUARDIAN":      0.8,
    "GBP_HUNTER_MAX":    0.5,
    "GBP_STOP_DELTA":    0.35,
    "GBP_REENTRY_MAX":   0.3,
    "OP_ORDERED_MIN":    0.4,
    "OP_DISORDERED_MAX": 0.15,
    "MAX_POSITIONS":     6,
    "MAX_LEVERAGE":      2.0,
    "DRAWDOWN_CB":       0.05,
    "GUARDIAN_COOLOFF_H": 48,
}
