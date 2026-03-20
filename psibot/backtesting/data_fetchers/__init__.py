"""
backtesting/data_fetchers/__init__.py

All data fetchers return clean pandas DataFrames with DatetimeIndex.
Convention:
  - All dates in UTC
  - All returns as log-returns (not percent)
  - NaN rows dropped before returning
  - Columns named consistently (symbol or 'value')

CCDR Expectation Field Architecture — Version 1.0
"""

import logging
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd

log = logging.getLogger("psibot.backtest")

# Simple disk cache to avoid re-fetching during development
CACHE_DIR = Path(".cache/backtest_data")
CACHE_DIR.mkdir(parents=True, exist_ok=True)
CACHE_TTL_HOURS = 24  # re-fetch after 24 hours


def cached_fetch(key: str, fetch_fn, ttl_hours: int = CACHE_TTL_HOURS):
    """Cache wrapper for expensive data fetches."""
    cache_file = CACHE_DIR / f"{key}.parquet"
    if cache_file.exists():
        age_hours = (time.time() - cache_file.stat().st_mtime) / 3600
        if age_hours < ttl_hours:
            log.debug("Cache hit: %s (age=%.1fh)", key, age_hours)
            return pd.read_parquet(cache_file)
    log.info("Fetching: %s", key)
    df = fetch_fn()
    if df is not None and len(df) > 0:
        df.to_parquet(cache_file)
    return df


FRED_API_KEY = os.environ.get("FRED_API_KEY", "")
