"""Load frozen OHLC and build the daily variance proxy (Garman-Klass, open-to-close)."""
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "raw"
TICKERS = ["SPY", "XLK", "XLF", "XLE", "XLV", "XLY", "XLP", "XLI", "XLU", "XLB"]


def load(ticker: str) -> pd.DataFrame:
    df = pd.read_csv(RAW / f"{ticker}.csv", parse_dates=["date"], index_col="date")
    return df.sort_index()


def garman_klass(df: pd.DataFrame) -> pd.Series:
    """Daily open-to-close variance, in percent^2. Ratios of same-day prices, so split/dividend adjustment cancels."""
    hl = np.log(df["High"] / df["Low"])
    co = np.log(df["Close"] / df["Open"])
    v = 0.5 * hl**2 - (2 * np.log(2) - 1) * co**2
    return (1e4 * v).rename("rv")


def frame(ticker: str) -> pd.DataFrame:
    """Per-day series used by every model. Row t only contains information known at the close of day t."""
    df = load(ticker)
    out = pd.DataFrame(index=df.index)
    out["rv"] = garman_klass(df)
    out["r_oc"] = 100 * np.log(df["Close"] / df["Open"])
    out["r_on"] = 100 * np.log(df["Open"] / df["Close"].shift(1))
    out = out.dropna()
    # floor zero-range days (rare) at a tiny positive value so logs/QLIKE are defined
    out["rv"] = out["rv"].clip(lower=1e-4)
    return out
