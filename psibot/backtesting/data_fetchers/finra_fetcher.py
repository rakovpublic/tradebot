"""
backtesting/data_fetchers/finra_fetcher.py

FINRA OTC Transparency (dark pool / ATS) weekly data fetcher.
Free public data — no registration required.

CCDR Expectation Field Architecture — Version 1.0
"""

import logging
from datetime import datetime

import pandas as pd
import requests

log = logging.getLogger("psibot.backtest")

_FINRA_URL = "https://api.finra.org/data/group/otcMarket/name/weeklySummary"
_FINRA_HEADERS = {
    "Accept": "application/json",
    "Content-Type": "application/json",
}


def fetch_finra_ats_weekly(
    start_year: int = 2014,
    end_year: int = None,
    symbol: str = "SPY",
) -> pd.DataFrame:
    """
    Fetch FINRA ATS (dark pool) weekly trading volume data and compute
    dark_pool_fraction = ATS_volume / total_volume.

    Uses a single POST request to the FINRA REST API — the GET endpoint
    triggers an infinite redirect loop regardless of filter format.

    Returns DataFrame: index=week_start_date, columns=['dark_pool_fraction', 'total_shares']
    Returns empty DataFrame if FINRA is unreachable (T5 fails gracefully).
    """
    import yfinance as yf

    if end_year is None:
        end_year = datetime.today().year

    # Single bulk POST request — avoids the GET redirect loop.
    # FINRA POST body uses compareFilters (array) + dateRangeFilters (array).
    payload = {
        "compareFilters": [
            {
                "fieldName": "issueSymbolIdentifier",
                "fieldValue": symbol,
                "compareType": "equal",
            },
        ],
        "dateRangeFilters": [
            {
                "fieldName": "weekStartDate",
                "startDate": f"{start_year}-01-01",
                "endDate": f"{end_year}-12-31",
            }
        ],
        "fields": ["weekStartDate", "totalShares", "issueSymbolIdentifier"],
        "limit": 5000,
        "offset": 0,
        "sortFields": [{"fieldName": "weekStartDate", "sortType": "ASC"}],
    }

    all_records = []
    try:
        resp = requests.post(
            _FINRA_URL, json=payload, headers=_FINRA_HEADERS, timeout=60
        )
        if resp.status_code == 200:
            for record in resp.json():
                all_records.append({
                    "date": pd.to_datetime(record.get("weekStartDate")),
                    "ats_shares": float(record.get("totalShares", 0) or 0),
                })
        else:
            log.warning("FINRA POST API returned HTTP %d — T5 will fail gracefully",
                        resp.status_code)
    except Exception as exc:
        log.warning("FINRA POST API failed: %s — T5 will fail gracefully", exc)

    if not all_records:
        log.warning("No FINRA ATS data fetched — returning empty DataFrame")
        return pd.DataFrame()

    df = pd.DataFrame(all_records).dropna()
    df = df.set_index("date").sort_index()

    # Fetch total volume from Yahoo Finance to compute dark pool fraction
    raw_vol = yf.download(
        symbol,
        start=f"{start_year}-01-01",
        progress=False,
        auto_adjust=True,
    )["Volume"]
    # squeeze in case yfinance returns single-column DataFrame
    vol_series = raw_vol.squeeze() if isinstance(raw_vol, pd.DataFrame) else raw_vol
    weekly_total = vol_series.resample("W-MON").sum()

    combined = pd.concat(
        [df["ats_shares"], weekly_total.rename("total_shares")], axis=1
    ).dropna()
    combined["dark_pool_fraction"] = (
        combined["ats_shares"] / combined["total_shares"].clip(lower=1)
    ).clip(0, 1)

    return combined[["dark_pool_fraction", "total_shares"]].dropna()
