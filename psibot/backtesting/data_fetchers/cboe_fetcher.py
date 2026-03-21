"""
backtesting/data_fetchers/cboe_fetcher.py

CBOE index fetchers for VIX, SKEW, and VIX9D.
Primary source: Yahoo Finance (^VIX, ^SKEW, ^VIX9D) — no auth required.
Fallback: CBOE CDN CSV (cdn.cboe.com) with browser headers in case Yahoo gaps.

CCDR Expectation Field Architecture — Version 1.0
"""

import io
import logging

import pandas as pd
import requests
import yfinance as yf

log = logging.getLogger("psibot.backtest")

# CBOE CDN requires browser-like headers; kept as fallback only.
_CBOE_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Referer": "https://www.cboe.com/",
}


def _cboe_csv(path: str, col_rename: str, start: str) -> pd.DataFrame:
    """Download a single CBOE CDN CSV and return a single-column DataFrame."""
    url = f"https://cdn.cboe.com/api/global/us_indices/daily_prices/{path}"
    resp = requests.get(url, headers=_CBOE_HEADERS, timeout=30)
    resp.raise_for_status()
    df = pd.read_csv(io.StringIO(resp.text))
    df.columns = df.columns.str.strip().str.upper()
    df["DATE"] = pd.to_datetime(df["DATE"])
    df = df.set_index("DATE").sort_index()
    df = df.rename(columns={"CLOSE": col_rename})[[col_rename]]
    return df[df.index >= start].dropna()


def _yf_index(ticker: str, col_name: str, start: str) -> pd.DataFrame:
    """Download a Yahoo Finance index ticker and return a single-column DataFrame."""
    raw = yf.download(ticker, start=start, auto_adjust=True, progress=False)
    if raw is None or raw.empty:
        raise ValueError(f"Yahoo Finance returned no data for {ticker}")
    # yfinance may return MultiIndex columns for single ticker in newer versions
    close = raw["Close"].squeeze() if "Close" in raw.columns else raw.squeeze()
    if isinstance(close, pd.DataFrame):
        close = close.iloc[:, 0]
    df = close.to_frame(name=col_name)
    df.index = pd.to_datetime(df.index)
    return df[df.index >= start].dropna()


def fetch_vix_history(start: str = "1990-01-02") -> pd.DataFrame:
    """
    Fetch daily VIX closing values.
    Primary: Yahoo Finance (^VIX). Fallback: CBOE CDN CSV.

    Returns DataFrame: index=Date, columns=['VIX']
    """
    try:
        return _yf_index("^VIX", "VIX", start)
    except Exception as exc:
        log.warning("VIX Yahoo fetch failed (%s); trying CBOE CDN", exc)
        return _cboe_csv("VIX_History.csv", "VIX", start)


def fetch_skew_history(start: str = "1990-01-02") -> pd.DataFrame:
    """
    Fetch CBOE SKEW Index history.
    Primary: Yahoo Finance (^SKEW). Fallback: CBOE CDN CSV.

    Returns DataFrame: index=Date, columns=['SKEW']
    """
    try:
        return _yf_index("^SKEW", "SKEW", start)
    except Exception as exc:
        log.warning("SKEW Yahoo fetch failed (%s); trying CBOE CDN", exc)
        return _cboe_csv("SKEW_History.csv", "SKEW", start)


def fetch_vix9d_history(start: str = "2011-01-01") -> pd.DataFrame:
    """
    Fetch VIX9D (9-day VIX).
    Primary: Yahoo Finance (^VIX9D). Fallback: CBOE CDN CSV.

    Returns DataFrame: index=Date, columns=['VIX9D']
    """
    try:
        return _yf_index("^VIX9D", "VIX9D", start)
    except Exception as exc:
        log.warning("VIX9D Yahoo fetch failed (%s); trying CBOE CDN", exc)
        return _cboe_csv("VIX9D_History.csv", "VIX9D", start)


def fetch_vix3m_history(start: str = "2011-01-01") -> pd.DataFrame:
    """
    Fetch VIX3M (3-month / 93-day VIX).  Available from Yahoo Finance since 2011.
    Primary: Yahoo Finance (^VIX3M). Fallback: CBOE CDN CSV.

    VIX3M > VIX  → contango (calm, normal market)   → bullish 3-month signal
    VIX3M < VIX  → backwardation (stressed market)  → bearish 3-month signal

    Returns DataFrame: index=Date, columns=['VIX3M']
    """
    try:
        return _yf_index("^VIX3M", "VIX3M", start)
    except Exception as exc:
        log.warning("VIX3M Yahoo fetch failed (%s); trying CBOE CDN", exc)
        return _cboe_csv("VIX3M_History.csv", "VIX3M", start)
