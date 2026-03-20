"""
backtesting/data_fetchers/french_fetcher.py

Kenneth French Data Library fetchers via direct HTTP download.
No API key required. pandas_datareader is not used (broken with pandas 2.x).

CCDR Expectation Field Architecture — Version 1.0
"""

import io
import zipfile

import pandas as pd
import requests

# Base URL for Kenneth French's data library
_FRENCH_BASE = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp"


def _fetch_french_csv(dataset_name: str) -> pd.DataFrame:
    """
    Download a Kenneth French Data Library zip, extract the CSV, and return
    the monthly factor returns as a DataFrame (decimal, DatetimeIndex).

    French CSV files are comma-delimited:
      - Several header/description lines (no leading digit in first comma-field)
      - One column-header line: " ,Col1 ,Col2 ..."
      - Monthly data rows: "YYYYMM , val1 , val2 ..."
      - Blank line separating monthly from annual section
    """
    url = f"{_FRENCH_BASE}/{dataset_name}_CSV.zip"
    resp = requests.get(url, timeout=60)
    resp.raise_for_status()

    with zipfile.ZipFile(io.BytesIO(resp.content)) as z:
        csv_name = next(n for n in z.namelist() if n.upper().endswith(".CSV"))
        raw = z.read(csv_name).decode("utf-8", errors="replace")

    lines = raw.splitlines()

    # Locate the first data row: first comma-field is a 6-digit YYYYMM integer.
    data_start = None
    header_idx = None
    for i, line in enumerate(lines):
        first_field = line.split(",")[0].strip()
        if first_field.isdigit() and len(first_field) == 6:
            data_start = i
            # Column header is the last non-blank line before data_start
            for j in range(i - 1, -1, -1):
                candidate = lines[j].strip()
                if candidate:
                    header_idx = j
                    break
            break

    if data_start is None:
        raise ValueError(f"No data rows found in {dataset_name}")

    # Parse column names from header line
    if header_idx is not None:
        raw_cols = [c.strip() for c in lines[header_idx].split(",")]
        # First element is empty (date column placeholder); skip it
        col_names = [c for c in raw_cols if c]
    else:
        col_names = []

    # Parse monthly data rows until blank line (= start of annual section)
    rows = []
    for line in lines[data_start:]:
        parts = [p.strip() for p in line.split(",")]
        if not parts or not parts[0]:
            break  # blank line → end of monthly section
        try:
            date_int = int(parts[0])
            year, month = date_int // 100, date_int % 100
            if not (1 <= month <= 12):
                continue
            values = [float(v) for v in parts[1:] if v]
            rows.append((pd.Timestamp(year=year, month=month, day=1), values))
        except (ValueError, TypeError):
            continue

    if not rows:
        raise ValueError(f"Failed to parse any rows from {dataset_name}")

    n_cols = len(rows[0][1])
    if len(col_names) < n_cols:
        col_names = col_names + [f"F{i}" for i in range(len(col_names), n_cols)]

    dates = [r[0] for r in rows]
    values = [r[1] for r in rows]
    df = pd.DataFrame(values, index=dates, columns=col_names[:n_cols])
    df.columns = df.columns.str.strip()
    return df.replace(-99.99, float("nan")).replace(-999, float("nan"))


def fetch_momentum_factor(start: str = "1990-01-01") -> pd.DataFrame:
    """
    Fetch Fama-French momentum factor (MOM / UMD) from Kenneth French's Data Library.
    Standard 12-1 month momentum factor return series, monthly.

    Source: https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/data_library.html
    No API key. Direct CSV zip download.

    Returns DataFrame: index=Date (monthly), columns=['MOM'] (decimal returns)
    """
    df = _fetch_french_csv("F-F_Momentum_Factor")
    df = df / 100.0  # percent → decimal

    # Find momentum column (MOM or UMD)
    mom_col = next(
        (c for c in df.columns if "mom" in c.lower() or "umd" in c.lower()),
        df.columns[0],
    )
    result = df[[mom_col]].rename(columns={mom_col: "MOM"})
    result = result[result.index >= start]
    return result.dropna()


def fetch_ff5_factors(start: str = "1990-01-01") -> pd.DataFrame:
    """
    Fetch Fama-French 5-factor model data (Mkt-RF, SMB, HML, RMW, CMA + RF).
    Used as control variables in T9 cross-sectional regression.
    Monthly, decimal returns.
    """
    df = _fetch_french_csv("F-F_Research_Data_5_Factors_2x3")
    df = df / 100.0
    df = df[df.index >= start]
    return df.dropna()
