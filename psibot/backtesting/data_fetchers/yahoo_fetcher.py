"""
backtesting/data_fetchers/yahoo_fetcher.py

Yahoo Finance data fetchers for equity prices, earnings, and asset universe.
No API key required.

CCDR Expectation Field Architecture — Version 1.0
"""

import logging
import numpy as np
import pandas as pd
import yfinance as yf

log = logging.getLogger("psibot.backtest")

# Yahoo Finance ticker mapping for the 27-asset ΨBot universe
YAHOO_MAP = {
    # Equities
    "SPX":      "^GSPC",
    "NDX":      "^NDX",
    "RUT":      "^RUT",
    "SXXP":     "^STOXX",
    "NKY":      "^N225",
    "MSCIEM":   "EEM",
    # Fixed Income
    "US2Y":     "SHY",
    "US10Y":    "IEF",
    "US30Y":    "TLT",
    "BUND10Y":  "IBGL.L",
    "JGB10Y":   "2621.T",
    # Credit
    "CDXIG":    "LQD",
    "CDXHY":    "HYG",
    "ITRAXXMAIN": "IEAA.L",
    "ITRAXXCO": "IHYG.L",
    # FX
    "DXY":      "DX-Y.NYB",
    "EURUSD":   "EURUSD=X",
    "USDJPY":   "JPY=X",
    "GBPUSD":   "GBPUSD=X",
    "AUDUSD":   "AUDUSD=X",
    # Commodities
    "GOLD":     "GC=F",
    "CRUDE":    "CL=F",
    "COPPER":   "HG=F",
    "WHEAT":    "ZW=F",
    # Volatility
    "VIX":      "^VIX",
    "VVIX":     "^VVIX",
    "MOVE":     "^MOVE",
}


def fetch_asset_universe_prices(start: str = "2000-01-01") -> pd.DataFrame:
    """
    Fetch daily closing prices for the 27-asset ΨBot universe from Yahoo Finance.

    Returns DataFrame: index=Date, columns=ΨBot asset names (as many as available)
    """
    tickers = list(YAHOO_MAP.values())
    raw = yf.download(
        tickers,
        start=start,
        end=None,
        auto_adjust=True,
        progress=False,
        threads=True,
    )

    # Handle single-ticker vs multi-ticker download shape
    if isinstance(raw.columns, pd.MultiIndex):
        close = raw["Close"]
    else:
        close = raw[["Close"]] if "Close" in raw.columns else raw

    reverse_map = {v: k for k, v in YAHOO_MAP.items()}
    close = close.rename(columns=reverse_map)
    close = close.dropna(how="all")

    missing = [k for k in YAHOO_MAP if k not in close.columns]
    if missing:
        log.warning("Missing assets from Yahoo Finance (reduced universe): %s", missing)

    return close.dropna(how="all")


def fetch_earnings_surprise_dispersion(
    tickers: list = None,
    start: str = "2000-01-01",
) -> pd.DataFrame:
    """
    Compute cross-sectional std of earnings surprises across S&P 500 basket.
    Free proxy for IBES analyst forecast dispersion (DP).

    Returns DataFrame: index=quarter_date, columns=['dp_proxy']
    """
    if tickers is None:
        tickers = [
            "AAPL", "MSFT", "AMZN", "GOOGL", "META", "NVDA", "TSLA", "BRK-B",
            "JPM", "JNJ", "V", "PG", "UNH", "HD", "MA", "BAC", "XOM", "CVX",
            "LLY", "ABBV", "KO", "PEP", "MRK", "TMO", "ABT", "COST", "WMT",
            "DIS", "CSCO", "VZ", "INTC", "CMCSA", "ADBE", "CRM", "PYPL",
            "NFLX", "AMD", "QCOM", "TXN", "HON", "UNP", "CAT", "BA", "MMM",
            "GS", "MS", "WFC", "C", "AXP", "USB",
        ]

    records = []
    for ticker in tickers:
        try:
            stock = yf.Ticker(ticker)
            earnings = stock.earnings_dates
            if earnings is None or len(earnings) == 0:
                continue
            earnings = earnings[["EPS Estimate", "Reported EPS"]].dropna()
            earnings["surprise_pct"] = (
                (earnings["Reported EPS"] - earnings["EPS Estimate"])
                / earnings["EPS Estimate"].abs().clip(lower=0.01)
            )
            earnings["ticker"] = ticker
            records.append(earnings[["surprise_pct", "ticker"]])
        except Exception:
            continue

    if not records:
        return pd.DataFrame()

    all_surprises = pd.concat(records)
    all_surprises.index = pd.to_datetime(all_surprises.index).tz_localize(None)

    monthly_dp = (
        all_surprises.groupby(pd.Grouper(freq="ME"))["surprise_pct"]
        .std()
        .to_frame(name="dp_proxy")
        .dropna()
    )
    return monthly_dp[monthly_dp.index >= start]
