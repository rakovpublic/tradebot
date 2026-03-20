"""
backtesting/data_fetchers/shiller_fetcher.py

Robert Shiller Yale dataset fetcher — monthly S&P 500 data back to 1871.
No API key required.

Source: http://www.econ.yale.edu/~shiller/data/ie_data.xls

CCDR Expectation Field Architecture — Version 1.0
"""

import io
import pandas as pd
import requests


def fetch_shiller_data() -> pd.DataFrame:
    """
    Fetch Robert Shiller's monthly S&P 500 dataset directly from Yale.

    Contains: Date, S&P 500 Price, Dividend, Earnings, CPI, Long Rate,
              Real Price, Real Dividend, Real Total Return Price,
              Real Earnings, CAPE, and more.

    Monthly data from January 1871 to present.

    Returns DataFrame: index=Date (monthly), columns include
        price, dividend, earnings, cpi, long_rate, real_price,
        real_dividend, real_tr_price, real_earnings, cape
    """
    url = "http://www.econ.yale.edu/~shiller/data/ie_data.xls"
    resp = requests.get(url, timeout=60)
    resp.raise_for_status()

    df = pd.read_excel(
        io.BytesIO(resp.content),
        sheet_name="Data",
        header=7,           # Row 8 (0-indexed = 7) is the header
        usecols="A:P",
    )

    df.columns = df.columns.str.strip()
    df = df.rename(columns={
        "Date": "date_raw",
        "P": "price",
        "D": "dividend",
        "E": "earnings",
        "CPI": "cpi",
        "Rate GS10": "long_rate",
        "Real Price": "real_price",
        "Real Dividend": "real_dividend",
        "Real TR Price": "real_tr_price",
        "Real Earnings": "real_earnings",
        "P/E10 or CAPE": "cape",
    })

    def decimal_to_date(d):
        try:
            year = int(d)
            month_frac = d - year
            month = max(1, min(12, round(month_frac * 12) + 1))
            return pd.Timestamp(year=year, month=month, day=1)
        except (ValueError, TypeError):
            return pd.NaT

    df["date"] = df["date_raw"].apply(
        lambda x: decimal_to_date(float(str(x).replace(" ", "")))
        if pd.notna(x) else pd.NaT
    )
    df = df.dropna(subset=["date", "price"])
    df = df.set_index("date").sort_index()

    numeric_cols = [
        "price", "dividend", "earnings", "cpi", "long_rate",
        "real_price", "real_dividend", "real_tr_price", "real_earnings", "cape",
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    return df.dropna(how="all")
