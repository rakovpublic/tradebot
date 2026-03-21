"""
pipeline/l5_acoustic_parser.py — AGT-05: Acoustic Residue Parser (SK-11)
=========================================================================
Parses conventional market data (prices, volume) for CONFIRMATION or
CONTRADICTION of upstream expectation field signals.

THIS AGENT NEVER GENERATES PRIMARY SIGNALS.
L5 is the acoustic residue — the visible downstream consequence of the
optical primary expectation field.

From the article (Section 8.2):
  'Price time series: transactions = expectation collapses. Downstream, lagging signal.'
  'Volume: coupling rate between expectation and price fields.'
  'Bid-ask spreads: the acoustic-optical phonon gap width.'

L5 only modifies position sizing (×1.2 confirm, ×0.7 contradict, ×1.0 neutral).

CCDR Expectation Field Architecture — Version 1.0
"""

from __future__ import annotations

import logging
from typing import Optional
import numpy as np

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from helpers import AcousticSignal, PsiShape, CCDR_THRESHOLDS

_MOM_CRASH  = CCDR_THRESHOLDS["MOM_CRASH_THRESHOLD"]   # -0.20
_MOM_NORMAL = CCDR_THRESHOLDS["MOM_NORMAL_THRESHOLD"]  #  0.05

log = logging.getLogger("psibot.pipeline.l5")


async def run(state, cross_asset_data) -> object:
    """
    Layer 5 main entry point.
    Populates acoustic_signal, momentum_20d, momentum_60d, breadth.

    Args:
        state: CondensateState (with L1-L4 fields populated)
        cross_asset_data: CrossAssetData from cross_asset_feed.py

    Returns:
        Updated CondensateState
    """
    try:
        if cross_asset_data is None:
            log.warning("L5: no cross-asset data — setting acoustic=NEUTRAL")
            return _apply_l5_neutral(state, "no cross-asset data")

        # Extract acoustic measurements from price data
        momentum_20d  = cross_asset_data.momentum_20d
        momentum_60d  = cross_asset_data.momentum_60d
        momentum_252d = cross_asset_data.momentum_252d
        breadth       = cross_asset_data.breadth
        volume_ratio  = cross_asset_data.volume_ratio

        # Classify acoustic signal (CONFIRM / CONTRADICT / NEUTRAL)
        acoustic_signal = _classify_acoustic(
            psi_shape=state.psi_shape,
            momentum_20d=momentum_20d,
            momentum_252d=momentum_252d,
            breadth=breadth,
            volume_ratio=volume_ratio,
        )

        # Write to state
        state.acoustic_signal = acoustic_signal
        state.momentum_20d  = momentum_20d
        state.momentum_60d  = momentum_60d
        state.momentum_252d = momentum_252d
        state.breadth       = breadth
        state.volume_ratio  = volume_ratio
        state.l5_failed = False

        log.info(
            "L5 complete: acoustic=%s mom20d=%.3f mom60d=%.3f mom252d=%.3f breadth=%.2f",
            acoustic_signal.value, momentum_20d, momentum_60d, momentum_252d, breadth,
        )
        return state

    except Exception as e:
        log.error("L5 acoustic parsing failed: %s — setting NEUTRAL", e)
        state.pipeline_errors.append(f"L5: {e}")
        return _apply_l5_neutral(state, str(e))


def _classify_acoustic(
    psi_shape: PsiShape,
    momentum_20d: float,
    momentum_252d: float,
    breadth: float,
    volume_ratio: float,
) -> AcousticSignal:
    """
    Classify acoustic signal by comparing price momentum with ψ_exp chirality.

    Note: psi_skew (scalar CBOE-style skew) is NOT used here — T8 showed it
    has no directional predictive power (49.8% accuracy). Direction comes only
    from psi_shape (full risk-neutral density, validated by T1 Granger test).

    T3 gap-zone logic (12m momentum thresholds):
      mom_252d <  -20%  → crash regime: CONTRADICT on all soliton-adjacent shapes
      mom_252d ∈ (-20%, +5%) → gap zone: CONTRADICT when fat tails (boundary imminent)
      mom_252d >  +5%   → normal regime: standard chirality-alignment logic

    CONFIRM conditions:
      - ψ skewed right (bullish chirality) AND momentum_20d > 0
      - ψ skewed left (bearish chirality) AND momentum_20d < 0
      - Breadth confirms direction AND volume above average

    CONTRADICT conditions:
      - ψ skewed right BUT momentum_20d < -0.02
      - ψ skewed left BUT momentum_20d > 0.02
      - FAT_TAILED AND mom_252d in gap zone or crash regime (T3: soliton collapse risk)
    """
    in_crash_regime = momentum_252d < _MOM_CRASH
    in_gap_zone     = _MOM_CRASH <= momentum_252d < _MOM_NORMAL

    # CONFIRM: acoustic aligns with expectation field chirality
    if psi_shape == PsiShape.SKEWED_RIGHT:
        if momentum_20d > 0.01:
            if breadth > 0.55 and volume_ratio > 1.0:
                return AcousticSignal.CONFIRM
            return AcousticSignal.CONFIRM  # basic chirality alignment
        if momentum_20d < -0.02:
            return AcousticSignal.CONTRADICT

    elif psi_shape == PsiShape.SKEWED_LEFT:
        if momentum_20d < -0.01:
            if breadth < 0.45 and volume_ratio > 1.0:
                return AcousticSignal.CONFIRM
            return AcousticSignal.CONFIRM
        if momentum_20d > 0.02:
            return AcousticSignal.CONTRADICT

    elif psi_shape == PsiShape.GAUSSIAN:
        # Deep grain interior — check if momentum is quiet (confirming stability)
        if abs(momentum_20d) < 0.01:
            return AcousticSignal.CONFIRM  # quiet momentum = stable grain
        if abs(momentum_20d) > 0.05:
            return AcousticSignal.CONTRADICT  # strong momentum in stable grain = warning

    elif psi_shape == PsiShape.BIMODAL:
        # Grain boundary crossing — acoustic signal is highly uncertain
        # Strong volume confirms crossing in progress
        if volume_ratio > 1.5:
            return AcousticSignal.CONFIRM  # high volume confirms transition
        return AcousticSignal.NEUTRAL

    elif psi_shape == PsiShape.FAT_TAILED:
        # T3: fat tails in crash regime or gap zone = CONTRADICT (soliton collapse risk).
        # Previously returned CONFIRM on negative short-term momentum alone, but that
        # ignored the bimodal regime structure: the gap zone has abrupt crash risk with
        # no gradual warning — acoustic must signal danger, not false confirmation.
        if in_crash_regime or in_gap_zone:
            log.warning(
                "L5 FAT_TAILED + T3 %s (mom252d=%.1f%%) → CONTRADICT",
                "crash regime" if in_crash_regime else "gap zone",
                momentum_252d * 100,
            )
            return AcousticSignal.CONTRADICT
        # Normal regime + fat tails: approaching saturation but not crash-regime
        if momentum_20d < -0.01:
            return AcousticSignal.CONFIRM
        return AcousticSignal.NEUTRAL

    # Default: NEUTRAL (L5 is lowest priority — don't force signal)
    return AcousticSignal.NEUTRAL


def _apply_l5_neutral(state, reason: str) -> object:
    """L5 failure protocol: set acoustic=NEUTRAL. Continue — L5 is lowest priority."""
    state.acoustic_signal = AcousticSignal.NEUTRAL
    state.l5_failed = True
    return state
