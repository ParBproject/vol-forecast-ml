"""The web app's exported data must be exactly the published results."""
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
D = ROOT / "web" / "public" / "data"
S = json.loads((ROOT / "results" / "summary.json").read_text())


def test_summary_identical():
    assert json.loads((D / "summary.json").read_text()) == S


def test_forecast_files_match():
    fc = pd.read_csv(ROOT / "results" / "forecasts.csv.gz")
    for t in S["tickers"]:
        w = json.loads((D / f"{t}.json").read_text())
        g = fc[fc["ticker"] == t]
        assert w["date"] == g["date"].tolist()
        for c in ["rv", *S["models"]]:
            assert max(abs(a - b) for a, b in zip(w[c], g[c], strict=True)) < 1e-6
