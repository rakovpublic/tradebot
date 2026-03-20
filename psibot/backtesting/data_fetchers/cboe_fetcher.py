"""
backtesting/data_fetchers/cboe_fetcher.py

CBOE free public data fetchers for VIX, SKEW, and VIX9D indices.
No API key required. Data updated daily at cdn.cboe.com.

CCDR Expectation Field Architecture — Version 1.0
"""

import io
import requests
import pandas as pd

# CBOE CDN blocks requests that lack a browser-like User-Agent (returns 403).
_CBOE_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Referer": "https://www.cboe.com/",
}


def fetch_vix_history(start: str = "1990-01-02") -> pd.DataFrame:
    """
    Fetch daily VIX closing values from CBOE.
    VIX = 30-day ATM implied vol of SPX options — proxy for ψ_exp L1 signal.

    Source: https://cdn.cboe.com/api/global/us_indices/daily_prices/VIX_History.csv
    No API key. Updated daily.

    Returns DataFrame: index=Date, columns=['VIX']
    """
    url = "https://cdn.cboe.com/api/global/us_indices/daily_prices/VIX_History.csv"
    resp = requests.get(url, headers=_CBOE_HEADERS, timeout=30)
    resp.raise_for_status()

    df = pd.read_csv(io.StringIO(resp.text))
    df.columns = df.columns.str.strip().str.upper()

    # CBOE format: DATE,OPEN,HIGH,LOW,CLOSE
    df["DATE"] = pd.to_datetime(df["DATE"])
    df = df.set_index("DATE").sort_index()
    df = df.rename(columns={"CLOSE": "VIX"})[["VIX"]]
    df = df[df.index >= start].dropna()
    return df


def fetch_skew_history(start: str = "1990-01-02") -> pd.DataFrame:
    """
    Fetch CBOE SKEW Index history (measures tail risk / ψ_exp skewness).
    Negative relationship: high SKEW → left-skewed ψ_exp → bearish condensate chirality.

    Source: https://cdn.cboe.com/api/global/us_indices/daily_prices/SKEW_History.csv
    """
    url = "https://cdn.cboe.com/api/global/us_indices/daily_prices/SKEW_History.csv"
    resp = requests.get(url, headers=_CBOE_HEADERS, timeout=30)
    resp.raise_for_status()
    df = pd.read_csv(io.StringIO(resp.text))
    df.columns = df.columns.str.strip().str.upper()
    df["DATE"] = pd.to_datetime(df["DATE"])
    df = df.set_index("DATE").sort_index()
    df = df.rename(columns={"CLOSE": "SKEW"})[["SKEW"]]
    return df[df.index >= start].dropna()


def fetch_vix9d_history(start: str = "2011-01-01") -> pd.DataFrame:
    """
    Fetch VIX9D (9-day VIX) — short-term expectation field proxy.
    Source: https://cdn.cboe.com/api/global/us_indices/daily_prices/VIX9D_History.csv
    """
    url = "https://cdn.cboe.com/api/global/us_indices/daily_prices/VIX9D_History.csv"
    resp = requests.get(url, headers=_CBOE_HEADERS, timeout=30)
    resp.raise_for_status()
    df = pd.read_csv(io.StringIO(resp.text))
    df.columns = df.columns.str.strip().str.upper()
    df["DATE"] = pd.to_datetime(df["DATE"])
    df = df.set_index("DATE").sort_index()
    df = df.rename(columns={"CLOSE": "VIX9D"})[["VIX9D"]]
    return df[df.index >= start].dropna()
