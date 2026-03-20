"""
backtesting/data_fetchers/french_fetcher.py

Kenneth French Data Library fetchers via pandas_datareader.
No API key required.

CCDR Expectation Field Architecture — Version 1.0
"""

import pandas as pd
import pandas_datareader.data as web
from datetime import datetime


def fetch_momentum_factor(start: str = "1990-01-01") -> pd.DataFrame:
    """
    Fetch Fama-French momentum factor (MOM / UMD) from Kenneth French's Data Library.
    Standard 12-1 month momentum factor return series, monthly.

    Source: Kenneth French Data Library (via pandas_datareader, no API key)

    Returns DataFrame: index=Date (monthly), columns=['MOM'] (decimal returns)
    """
    ff_data = web.DataReader(
        "F-F_Momentum_Factor",
        "famafrench",
        start=start,
        end=datetime.today().strftime("%Y-%m-%d"),
    )
    # ff_data[0] = monthly, ff_data[1] = annual
    monthly = ff_data[0].copy()
    monthly.index = pd.to_datetime(monthly.index.to_timestamp())
    monthly.columns = monthly.columns.str.strip()

    # Find momentum column (MOM or UMD)
    mom_col = [c for c in monthly.columns if "mom" in c.lower() or "umd" in c.lower()]
    if not mom_col:
        mom_col = monthly.columns.tolist()[:1]

    df = monthly[mom_col[0]].to_frame(name="MOM") / 100.0
    return df.dropna()


def fetch_ff5_factors(start: str = "1990-01-01") -> pd.DataFrame:
    """
    Fetch Fama-French 5-factor model data (Mkt-RF, SMB, HML, RMW, CMA + RF).
    Used as control variables in T9 cross-sectional regression.
    Monthly, decimal returns.
    """
    ff_data = web.DataReader(
        "F-F_Research_Data_5_Factors_2x3",
        "famafrench",
        start=start,
        end=datetime.today().strftime("%Y-%m-%d"),
    )
    monthly = ff_data[0].copy()
    monthly.index = pd.to_datetime(monthly.index.to_timestamp())
    monthly.columns = monthly.columns.str.strip()
    return (monthly / 100.0).dropna()
