"""Download daily OHLC from Yahoo Finance. Not used by `make all`."""
import datetime as dt
import json
from pathlib import Path

import yfinance as yf

TICKERS = ["SPY", "XLK", "XLF", "XLE", "XLV", "XLY", "XLP", "XLI", "XLU", "XLB"]
START, END = "2005-01-01", "2026-10-23"  # END exclusive
RAW = Path(__file__).resolve().parents[1] / "data" / "raw"
RAW.mkdir(parents=True, exist_ok=True)
for t in TICKERS:
    df = yf.Ticker(t).history(start=START, end=END, auto_adjust=False, actions=False)
    df.index = df.index.tz_localize(None).date
    df.index.name = "date"
    df[["Open", "High", "Low", "Close", "Adj Close", "Volume"]].round(6).to_csv(RAW / f"{t}.csv")
    print(t, len(df), df.index[0], df.index[-1])
meta = {"tickers": TICKERS, "start": START, "end_exclusive": END,
        "downloaded_utc": dt.datetime.now(dt.UTC).isoformat(timespec="seconds"),
        "source": "yfinance (Yahoo Finance) daily OHLC, auto_adjust=False"}
(RAW / "meta.json").write_text(json.dumps(meta, indent=2) + "\n")
