"""
ibkr_config.py — All IBKR connection parameters.
Edit this file for paper vs live trading.
NEVER commit API credentials to git.

CCDR Expectation Field Architecture — Version 1.0
"""

import os
from dataclasses import dataclass, field


@dataclass
class IBKRConfig:
    # IB Gateway connection
    host: str = "127.0.0.1"
    port: int = 4002          # 4002 = paper trading; 4001 = live trading
    client_id: int = 1        # unique per connection; use 1=main, 2=monitor, 3=backtest
    readonly: bool = False    # True for data-only connections (AGT-01 can use readonly=True)
    timeout: int = 20         # seconds to wait for connection

    # Account
    account_id: str = field(default_factory=lambda: os.environ.get("IBKR_ACCOUNT_ID", ""))

    # Risk limits (enforced at broker level, separate from CCDR structural stops)
    max_order_value_usd: float = field(
        default_factory=lambda: float(os.environ.get("IBKR_MAX_ORDER_USD", "50000"))
    )
    max_daily_loss_usd: float = field(
        default_factory=lambda: float(os.environ.get("IBKR_MAX_DAILY_LOSS", "5000"))
    )

    # Options surface streaming
    # Instruments to stream IV surfaces for L1 ψ_exp reconstruction
    options_universe: list = field(default_factory=list)

    def __post_init__(self):
        if not self.options_universe:
            self.options_universe = [
                "SPX",   # S&P 500 index options — primary condensate sensor
                "NDX",   # Nasdaq 100
                "SPY",   # SPX ETF (more liquid for some strikes)
                "QQQ",   # NDX ETF
                "IWM",   # Russell 2000
                "GLD",   # Gold
                "TLT",   # 20yr Treasury
                "HYG",   # High yield credit
                "VIX",   # VIX options
            ]

    @property
    def is_paper(self) -> bool:
        return self.port == 4002

    @classmethod
    def paper(cls) -> "IBKRConfig":
        return cls(port=4002)

    @classmethod
    def live(cls) -> "IBKRConfig":
        if not os.environ.get("IBKR_ACCOUNT_ID"):
            raise ValueError("IBKR_ACCOUNT_ID environment variable required for live trading")
        return cls(port=4001)
