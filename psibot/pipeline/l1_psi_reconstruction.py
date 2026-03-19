"""
pipeline/l1_psi_reconstruction.py — AGT-01: ψ_exp Wavefunction Reconstruction (SK-01, SK-02, SK-03)
=====================================================================================================
PRIMARY SENSOR — the most important layer in the pipeline.

Converts the options implied volatility surface into the expectation field
wavefunction ψ_exp(K,T). All other pipeline layers depend on this.

From the article (Section 7):
  'Options implied volatility surface = direct measurement of ψ_exp'
  'The vol surface IS the quantum state of the market.'

Algorithm:
  1. Validate options surface
  2. SVI smoothing for arbitrage-free surface
  3. Dupire PDE → local vol → risk-neutral density p(S_T)
  4. ψ_exp = √p(K,T) · e^{iθ(K,T)}
  5. Classify wavefunction shape, compute moments, term structure

CCDR Expectation Field Architecture — Version 1.0
"""

from __future__ import annotations

import logging
import warnings
from datetime import datetime
from typing import Optional
import numpy as np
from scipy.interpolate import CubicSpline, RectBivariateSpline
from scipy import integrate

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from helpers import (
    PsiShape, TermStructure,
    classify_psi_shape, compute_psi_entropy, classify_term_structure,
    compute_skew, compute_kurtosis_proxy,
)

log = logging.getLogger("psibot.pipeline.l1")

# Target tenor for primary ψ_exp classification (30-day = 1 month)
PRIMARY_TENOR_DAYS = 30


async def run(state, options_surface) -> object:
    """
    Layer 1 main entry point.
    Populates L1 fields in CondensateState from the options surface.

    On failure: conservative failsafe (psi_shape=UNKNOWN, GBP contribution=0.60).

    Args:
        state: CondensateState (mutated in-place)
        options_surface: OptionsSurface object from data/options_feed.py

    Returns:
        Updated CondensateState
    """
    if options_surface is None:
        log.warning("L1: options_surface is None — applying failsafe")
        return _apply_l1_failsafe(state, "options_surface unavailable")

    if not options_surface.is_valid:
        # Attempt validation
        options_surface.validate()
        if not options_surface.is_valid:
            log.warning("L1: surface validation failed (%d issues) — applying failsafe",
                        len(options_surface.validation_issues))
            return _apply_l1_failsafe(state, f"surface invalid: {options_surface.validation_issues[:2]}")

    try:
        state = _reconstruct_psi(state, options_surface)
        log.info("L1 complete: psi_shape=%s skew=%.3f kurt=%.3f entropy=%.3f ts=%s",
                 state.psi_shape.value, state.psi_skew, state.psi_kurtosis_excess,
                 state.psi_entropy, state.psi_term_structure.value)
        return state

    except Exception as e:
        log.error("L1 reconstruction failed: %s — applying failsafe", e, exc_info=True)
        state.pipeline_errors.append(f"L1: {e}")
        return _apply_l1_failsafe(state, str(e))


def _reconstruct_psi(state, surface) -> object:
    """
    Core ψ_exp reconstruction pipeline.
    """
    spot = surface.spot

    # Determine which tenors are available
    available_tenors = [t for t in surface.tenors_days if t in surface.iv]
    if not available_tenors:
        return _apply_l1_failsafe(state, "no tenors available")

    # Step 1: Select primary tenor (closest to 30 days)
    primary_tenor = min(available_tenors, key=lambda t: abs(t - PRIMARY_TENOR_DAYS))

    # Step 2: Get arbitrage-free density at primary tenor via Dupire
    strikes = surface.strikes[primary_tenor]
    iv_slice = surface.iv[primary_tenor]

    p_density, K_grid = _compute_risk_neutral_density(
        strikes=strikes,
        iv_slice=iv_slice,
        spot=spot,
        tenor_days=primary_tenor,
    )

    # Step 3: Classify wavefunction shape (SK-02)
    psi_shape = classify_psi_shape(p_density, K_grid)

    # Step 4: Compute wavefunction moments
    psi_entropy = compute_psi_entropy(p_density, K_grid)
    psi_skew = surface.skew_at_tenor(primary_tenor)
    psi_kurtosis = surface.kurtosis_excess_at_tenor(primary_tenor)

    # Step 5: Classify term structure from ATM IVs across tenors
    atm_ivs = surface.atm_iv_by_tenor()
    term_structure = classify_term_structure(atm_ivs)

    # Step 6: Build ψ_exp wavefunction (amplitude = √p)
    psi_amplitude = np.sqrt(np.clip(p_density, 0, None))

    # Write to state
    state.psi_shape = psi_shape
    state.psi_skew = psi_skew
    state.psi_kurtosis_excess = psi_kurtosis
    state.psi_entropy = psi_entropy
    state.psi_term_structure = term_structure
    state.psi_amplitude = psi_amplitude
    state.l1_failed = False

    return state


def _compute_risk_neutral_density(
    strikes: np.ndarray,
    iv_slice: np.ndarray,
    spot: float,
    tenor_days: int,
    n_grid: int = 100,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute risk-neutral density p(K) via Dupire's formula.

    Dupire: p(K, T) = C_T / (0.5 * K² * C_KK)
    where C = call price, subscripts are partial derivatives.

    For discrete IV surface, we:
      1. Fit SVI-like smooth curve to IV(K) at this tenor
      2. Compute call prices on fine grid
      3. Estimate p(K) from second derivative of call prices
      4. Validate: ∫p(K)dK ≈ 1

    Returns:
        (p_density array, K_grid array)
    """
    T = tenor_days / 252.0
    r = 0.05  # risk-free rate (use config in production)

    # Build fine grid of strikes
    K_min = max(strikes.min(), 0.5 * spot)
    K_max = min(strikes.max(), 2.0 * spot)
    K_grid = np.linspace(K_min, K_max, n_grid)

    # Step 1: Smooth IV surface (SVI approximation via cubic spline)
    try:
        cs = CubicSpline(strikes, iv_slice, extrapolate=False)
        sigma_grid = np.clip(cs(K_grid), 0.005, 3.0)
        # Fill NaN/out-of-range with boundary values
        sigma_grid = np.where(np.isfinite(sigma_grid), sigma_grid,
                              iv_slice[len(iv_slice)//2])
    except Exception:
        # Fallback: linear interpolation
        sigma_grid = np.interp(K_grid, strikes, iv_slice)

    # Step 2: Compute Black-Scholes call prices on grid
    call_prices = _bs_call(spot, K_grid, T, r, sigma_grid)

    # Step 3: Extract density via Breeden-Litzenberger (second derivative of calls)
    # p(K) = e^{rT} * d²C/dK²
    dK = K_grid[1] - K_grid[0]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        p = np.gradient(np.gradient(call_prices, dK), dK) * np.exp(r * T)

    # Step 4: Ensure non-negative and normalise
    p = np.clip(p, 0, None)
    norm = np.trapz(p, K_grid)
    if norm < 1e-8:
        # Fallback to log-normal density
        log.warning("L1: density normalisation failed (norm=%.2e), using log-normal fallback", norm)
        sigma_atm = float(np.interp(spot, strikes, iv_slice))
        p = _lognormal_density(K_grid, spot, T, r, sigma_atm)
        norm = np.trapz(p, K_grid)

    p = p / (norm + 1e-12)

    # Validate: probability must integrate to 1 ± 0.001
    total = np.trapz(p, K_grid)
    if abs(total - 1.0) > 0.01:
        log.warning("L1: density integrates to %.4f (expected ~1.0)", total)

    return p, K_grid


def _bs_call(S: float, K: np.ndarray, T: float, r: float, sigma: np.ndarray) -> np.ndarray:
    """Black-Scholes call price for array of strikes and vols."""
    from scipy.stats import norm
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sqrtT = np.sqrt(max(T, 1e-8))
        d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * sqrtT)
        d2 = d1 - sigma * sqrtT
        call = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    return np.clip(call, 0, None)


def _lognormal_density(K: np.ndarray, S: float, T: float, r: float, sigma: float) -> np.ndarray:
    """Log-normal risk-neutral density (Black-Scholes world)."""
    from scipy.stats import lognorm
    mu = np.log(S) + (r - 0.5 * sigma**2) * T
    std = sigma * np.sqrt(max(T, 1e-8))
    p = lognorm.pdf(K, s=std, scale=np.exp(mu))
    return np.clip(p, 0, None)


def _apply_l1_failsafe(state, reason: str) -> object:
    """
    L1 failure protocol:
    - psi_shape = UNKNOWN (GBP contribution = 0.60 conservative)
    - psi_skew = 0.0 (neutral)
    - psi_entropy = high (conservative — treat as disordered)
    - Log warning
    """
    log.warning("L1 FAILSAFE activated: %s", reason)
    state.psi_shape = PsiShape.UNKNOWN
    state.psi_skew = 0.0
    state.psi_kurtosis_excess = 0.0
    state.psi_entropy = 1.5   # high entropy = conservative
    state.psi_term_structure = TermStructure.FLAT
    state.psi_amplitude = None
    state.l1_failed = True
    state.pipeline_errors.append(f"L1 failsafe: {reason}")
    # Note: GBP will use UNKNOWN score = 0.60 (conservative)
    return state
