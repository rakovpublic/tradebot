"""
backtesting/data_fetchers/fred_fetcher.py

FRED (Federal Reserve Economic Data) fetchers.
Requires a free FRED API key: https://fred.stlouisfed.org/docs/api/api_key.html
Set via environment variable: export FRED_API_KEY=your_key_here

CCDR Expectation Field Architecture — Version 1.0
"""

import os
import pandas as pd


def get_fred():
    """Return a Fred client. Raises EnvironmentError if API key not set."""
    from fredapi import Fred
    key = os.environ.get("FRED_API_KEY", "")
    if not key:
        raise EnvironmentError(
            "FRED_API_KEY not set. Get a free key at: "
            "https://fred.stlouisfed.org/docs/api/api_key.html"
        )
    return Fred(api_key=key)


def fetch_spx_prices(start: str = "1990-01-01") -> pd.DataFrame:
    """
    S&P 500 daily close from FRED (series: SP500).
    Returns DataFrame: index=Date, columns=['SPX']
    """
    fred = get_fred()
    series = fred.get_series("SP500", observation_start=start)
    df = series.to_frame(name="SPX").dropna()
    df.index = pd.to_datetime(df.index)
    return df


def fetch_10y_treasury(start: str = "1990-01-01") -> pd.DataFrame:
    """10-year Treasury yield (DGS10) from FRED."""
    fred = get_fred()
    s = fred.get_series("DGS10", observation_start=start)
    df = s.to_frame(name="US10Y").dropna()
    df.index = pd.to_datetime(df.index)
    return df


def fetch_fed_funds_rate(start: str = "1990-01-01") -> pd.DataFrame:
    """Effective federal funds rate (EFFR) from FRED."""
    fred = get_fred()
    s = fred.get_series("EFFR", observation_start=start)
    df = s.to_frame(name="EFFR").dropna()
    df.index = pd.to_datetime(df.index)
    return df


def fetch_nber_recession_dates(start: str = "1990-01-01") -> pd.DataFrame:
    """
    NBER US recession indicator from FRED (USREC series).
    Returns DataFrame: index=date, columns=['recession'] (1=recession, 0=expansion)
    """
    fred = get_fred()
    s = fred.get_series("USREC", observation_start=start)
    df = s.to_frame(name="recession")
    df.index = pd.to_datetime(df.index)
    return df
