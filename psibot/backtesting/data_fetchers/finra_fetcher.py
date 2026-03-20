"""
backtesting/data_fetchers/finra_fetcher.py

FINRA OTC Transparency (dark pool / ATS) weekly data fetcher.
Free public data — no registration required.

CCDR Expectation Field Architecture — Version 1.0
"""

import io
import json
import logging
from datetime import datetime, timedelta

import pandas as pd
import requests

log = logging.getLogger("psibot.backtest")


def fetch_finra_ats_weekly(
    start_year: int = 2014,
    end_year: int = None,
    symbol: str = "SPY",
) -> pd.DataFrame:
    """
    Fetch FINRA ATS (dark pool) weekly trading volume data and compute
    dark_pool_fraction = ATS_volume / total_volume.

    FINRA publishes two-week delayed ATS data via REST API and direct CSV downloads.
    Combines API + CSV fallback for maximum coverage.

    Returns DataFrame: index=week_start_date, columns=['dark_pool_fraction', 'total_shares']
    """
    import yfinance as yf

    if end_year is None:
        end_year = datetime.today().year

    all_records = []

    base_url = "https://api.finra.org/data/group/otcMarket/name/weeklySummary"
    # FINRA REST API requires filters as a JSON array (not the legacy compareFilters string)
    _FINRA_HEADERS = {"Accept": "application/json"}

    for year in range(start_year, end_year + 1):
        try:
            params = {
                "limit": 52,
                "offset": 0,
                "fields": "weekStartDate,totalShares,totalTrades,issueSymbolIdentifier",
                "filters": json.dumps([
                    {"fieldName": "issueSymbolIdentifier",
                     "fieldValue": symbol, "compareType": "EQUAL"},
                    {"fieldName": "weekStartDate",
                     "fieldValue": f"{year}-01-01", "compareType": "GREATER_THAN_OR_EQUAL"},
                    {"fieldName": "weekStartDate",
                     "fieldValue": f"{year}-12-31", "compareType": "LESS_THAN_OR_EQUAL"},
                ]),
            }
            resp = requests.get(base_url, params=params, headers=_FINRA_HEADERS,
                                timeout=30, allow_redirects=True)

            if resp.status_code == 200:
                data = resp.json()
                for record in data:
                    all_records.append({
                        "date": pd.to_datetime(record.get("weekStartDate")),
                        "ats_shares": float(record.get("totalShares", 0)),
                    })
            else:
                log.warning("FINRA API returned %d for year %d", resp.status_code, year)
                _fetch_finra_csv_fallback(year, symbol, all_records)

        except Exception as exc:
            log.warning("FINRA fetch error for %d: %s — trying fallback", year, exc)
            _fetch_finra_csv_fallback(year, symbol, all_records)

    if not all_records:
        log.error("No FINRA data fetched — returning empty DataFrame")
        return pd.DataFrame()

    df = pd.DataFrame(all_records).dropna()
    df = df.set_index("date").sort_index()

    # Fetch total volume from Yahoo Finance for dark pool fraction
    yf_data = yf.download(
        symbol,
        start=f"{start_year}-01-01",
        progress=False,
        auto_adjust=True,
    )["Volume"]
    weekly_total = yf_data.resample("W-MON").sum()

    combined = pd.concat([df["ats_shares"], weekly_total.rename("total_shares")], axis=1)
    combined = combined.dropna()
    combined["dark_pool_fraction"] = (
        combined["ats_shares"] / combined["total_shares"].clip(lower=1)
    ).clip(0, 1)

    return combined[["dark_pool_fraction", "total_shares"]].dropna()


def _fetch_finra_csv_fallback(year: int, symbol: str, records: list) -> None:
    """
    Fallback: fetch FINRA weekly ATS data via direct CSV download.
    Tries known FINRA CSV URL pattern.
    """
    start = datetime(year, 1, 1)
    start += timedelta(days=(7 - start.weekday()) % 7)

    current = start
    while current.year == year:
        date_str = current.strftime("%Y%m%d")
        url = f"https://cdn.finra.org/equity/otcmarket/weekly/finra_ats_{date_str}.csv"
        try:
            resp = requests.get(url, timeout=10)
            if resp.status_code == 200:
                df = pd.read_csv(io.StringIO(resp.text))
                df.columns = df.columns.str.strip().str.upper()
                sym_col = next((c for c in df.columns if "SYMBOL" in c or "TICKER" in c), None)
                vol_col = next((c for c in df.columns if "SHARE" in c or "VOLUME" in c), None)
                if sym_col and vol_col:
                    row = df[df[sym_col].str.strip() == symbol.upper()]
                    if len(row) > 0:
                        records.append({
                            "date": pd.to_datetime(current),
                            "ats_shares": float(row[vol_col].values[0]),
                        })
        except Exception:
            pass
        current += timedelta(weeks=1)
