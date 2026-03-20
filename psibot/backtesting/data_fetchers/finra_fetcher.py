"""
backtesting/data_fetchers/finra_fetcher.py

Dark pool / ATS fraction data with three-tier fallback:

  1. FINRA POST API  (api.finra.org/data/group/otcMarket/name/weeklySummary)
  2. FINRA quarterly ATS transparency ZIP files (direct website download)
  3. Amihud illiquidity proxy from Yahoo Finance (always available)
     dark_pool_proxy ≈ 1 / Amihud_illiquidity, normalised to [0,1].
     Academic basis: Amihud (2002); low illiquidity ↔ institutional flow.

CCDR Expectation Field Architecture — Version 1.0
"""

import io
import logging
import zipfile
from datetime import datetime

import numpy as np
import pandas as pd
import requests

log = logging.getLogger("psibot.backtest")

_FINRA_API_URL = "https://api.finra.org/data/group/otcMarket/name/weeklySummary"
_FINRA_HEADERS = {"Accept": "application/json", "Content-Type": "application/json"}


# ---------------------------------------------------------------------------
# Tier 1 — FINRA REST API (POST)
# ---------------------------------------------------------------------------

def _try_finra_api(symbol: str, start_year: int, end_year: int) -> list:
    """Return list of {date, ats_shares} dicts from the FINRA API, or []."""
    payload = {
        "compareFilters": [
            {"fieldName": "issueSymbolIdentifier",
             "fieldValue": symbol, "compareType": "equal"},
        ],
        "dateRangeFilters": [
            {"fieldName": "weekStartDate",
             "startDate": f"{start_year}-01-01",
             "endDate": f"{end_year}-12-31"},
        ],
        "fields": ["weekStartDate", "totalShares", "issueSymbolIdentifier"],
        "limit": 5000,
        "offset": 0,
        "sortFields": [{"fieldName": "weekStartDate", "sortType": "ASC"}],
    }
    try:
        resp = requests.post(
            _FINRA_API_URL, json=payload, headers=_FINRA_HEADERS,
            timeout=30, allow_redirects=False,   # don't follow redirects
        )
        if resp.status_code == 200:
            records = resp.json()
            return [
                {"date": pd.to_datetime(r.get("weekStartDate")),
                 "ats_shares": float(r.get("totalShares", 0) or 0)}
                for r in records
            ]
        log.warning("FINRA API returned HTTP %d", resp.status_code)
    except Exception as exc:
        log.warning("FINRA API unavailable: %s", exc)
    return []


# ---------------------------------------------------------------------------
# Tier 2 — FINRA quarterly ATS transparency ZIP files (direct download)
# ---------------------------------------------------------------------------

def _try_finra_quarterly_zips(symbol: str, start_year: int, end_year: int) -> list:
    """
    Download FINRA quarterly ATS transparency ZIPs from the FINRA website
    (not the blocked API).  File structure per FINRA's OTC Transparency page:

    URL pattern:
      https://www.finra.org/sites/default/files/OTC-Transparency-Data/
      ATSTransparencyData_{YYYY}Q{Q}.zip

    Inside each ZIP: one CSV with columns including
      Issue Symbol, Week Start Date, Total Shares
    """
    records = []
    for year in range(start_year, end_year + 1):
        for quarter in range(1, 5):
            url = (
                "https://www.finra.org/sites/default/files/"
                f"OTC-Transparency-Data/ATSTransparencyData_{year}Q{quarter}.zip"
            )
            try:
                resp = requests.get(url, timeout=30, allow_redirects=True)
                if resp.status_code != 200:
                    continue
                with zipfile.ZipFile(io.BytesIO(resp.content)) as z:
                    csv_names = [n for n in z.namelist()
                                 if n.lower().endswith(".csv")]
                    if not csv_names:
                        continue
                    df = pd.read_csv(z.open(csv_names[0]), low_memory=False)

                df.columns = df.columns.str.strip()
                # Normalise column names
                sym_col = next(
                    (c for c in df.columns
                     if "symbol" in c.lower() or "ticker" in c.lower()), None)
                date_col = next(
                    (c for c in df.columns if "week" in c.lower() and "date" in c.lower()), None)
                vol_col = next(
                    (c for c in df.columns
                     if "total" in c.lower() and "share" in c.lower()), None)

                if not (sym_col and date_col and vol_col):
                    log.warning("FINRA ZIP %dQ%d: unexpected columns %s",
                                year, quarter, df.columns.tolist())
                    continue

                sym_rows = df[df[sym_col].str.strip().str.upper() == symbol.upper()]
                for _, row in sym_rows.iterrows():
                    try:
                        records.append({
                            "date": pd.to_datetime(row[date_col]),
                            "ats_shares": float(row[vol_col]),
                        })
                    except (ValueError, TypeError):
                        continue

                log.info("FINRA ZIP %dQ%d: %d rows for %s",
                         year, quarter, len(sym_rows), symbol)

            except Exception as exc:
                log.debug("FINRA ZIP %dQ%d failed: %s", year, quarter, exc)
                continue

    return records


# ---------------------------------------------------------------------------
# Tier 3 — Amihud illiquidity proxy (Yahoo Finance, always available)
# ---------------------------------------------------------------------------

def _amihud_dark_pool_proxy(symbol: str, start_year: int) -> pd.DataFrame:
    """
    Compute a weekly dark-pool-activity proxy from Yahoo Finance price/volume.

    Method (Amihud 2002, Journal of Financial Markets):
        ILLIQ_t = |r_t| / DollarVolume_t   (daily, × 10^9 for scale)

    Rationale: when institutions route orders through dark pools, price impact
    per unit of volume drops — i.e. ILLIQ falls.  Therefore:
        dark_pool_proxy ∝ 1 / ILLIQ (normalised to [0,1])

    The proxy is NOT a direct measure of ATS volume, but it carries the same
    information content for T5 (directional prediction of next-week return).
    Returns DataFrame: index=week_start (Monday), columns=['dark_pool_fraction','total_shares']
    """
    import yfinance as yf

    raw = yf.download(
        symbol,
        start=f"{start_year}-01-01",
        auto_adjust=True,
        progress=False,
    )
    close = raw["Close"].squeeze()
    volume = raw["Volume"].squeeze()

    daily_ret = close.pct_change().abs()
    dollar_vol = close * volume
    illiq = (daily_ret / dollar_vol.clip(lower=1) * 1e9).replace(
        [np.inf, -np.inf], np.nan
    )

    # Weekly aggregation (Monday week start)
    weekly_illiq = illiq.resample("W-MON").mean()
    weekly_vol = volume.resample("W-MON").sum()

    # Invert: low ILLIQ ↔ high institutional/dark-pool activity
    raw_proxy = 1.0 / (weekly_illiq + 1e-12)

    # Rolling normalisation over 52-week window (keeps proxy stationary)
    roll_min = raw_proxy.rolling(52, min_periods=4).min()
    roll_max = raw_proxy.rolling(52, min_periods=4).max()
    dark_pool_fraction = (
        (raw_proxy - roll_min) / (roll_max - roll_min + 1e-12)
    ).clip(0, 1)

    result = pd.DataFrame({
        "dark_pool_fraction": dark_pool_fraction,
        "total_shares": weekly_vol,
    }).dropna()

    log.info("Amihud dark-pool proxy: %d weekly observations for %s", len(result), symbol)
    return result


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def fetch_finra_ats_weekly(
    start_year: int = 2014,
    end_year: int = None,
    symbol: str = "SPY",
) -> pd.DataFrame:
    """
    Fetch dark pool / ATS fraction for `symbol`, weekly frequency.

    Priority:
      1. FINRA POST API
      2. FINRA quarterly ATS transparency ZIP files
      3. Amihud illiquidity proxy from Yahoo Finance

    Returns DataFrame: index=week_start_date (Monday),
                       columns=['dark_pool_fraction', 'total_shares']
    """
    import yfinance as yf

    if end_year is None:
        end_year = datetime.today().year

    # --- Tier 1: FINRA API ---
    records = _try_finra_api(symbol, start_year, end_year)
    source = "FINRA API"

    # --- Tier 2: FINRA quarterly ZIPs ---
    if not records:
        log.info("Trying FINRA quarterly ATS transparency ZIPs...")
        records = _try_finra_quarterly_zips(symbol, start_year, end_year)
        source = "FINRA quarterly ZIPs"

    if records:
        df = pd.DataFrame(records).dropna()
        df = df.set_index("date").sort_index()

        raw_vol = yf.download(
            symbol, start=f"{start_year}-01-01",
            progress=False, auto_adjust=True,
        )["Volume"]
        vol_series = raw_vol.squeeze() if isinstance(raw_vol, pd.DataFrame) else raw_vol
        weekly_total = vol_series.resample("W-MON").sum()

        combined = pd.concat(
            [df["ats_shares"], weekly_total.rename("total_shares")], axis=1
        ).dropna()
        combined["dark_pool_fraction"] = (
            combined["ats_shares"] / combined["total_shares"].clip(lower=1)
        ).clip(0, 1)

        result = combined[["dark_pool_fraction", "total_shares"]].dropna()
        if len(result) >= 50:
            log.info("Dark pool data from %s: %d weekly observations", source, len(result))
            return result
        log.warning("%s returned only %d rows — falling through to proxy", source, len(result))

    # --- Tier 3: Amihud proxy ---
    log.info("Using Amihud illiquidity proxy as dark-pool-fraction substitute")
    return _amihud_dark_pool_proxy(symbol, start_year)
