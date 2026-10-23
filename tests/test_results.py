"""Pin the published numbers and check they are reproducible from results/forecasts.csv.gz."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from volml.evaluate import diebold_mariano, mse, qlike

ROOT = Path(__file__).resolve().parents[1]
S = json.loads((ROOT / "results" / "summary.json").read_text())
FC = pd.read_csv(ROOT / "results" / "forecasts.csv.gz", parse_dates=["date"])

PINNED_MEAN_QLIKE = {"HAR": 0.3262, "GARCH": 0.3473, "GBM": 0.3251, "LSTM": 0.2968}
PINNED_MEAN_MSE = {"HAR": 10.212, "GARCH": 10.350, "GBM": 9.327, "LSTM": 9.803}


def test_pinned_headline():
    for k, v in PINNED_MEAN_QLIKE.items():
        assert S["aggregate"][k]["mean_qlike"] == pytest.approx(v, abs=5e-4)
    for k, v in PINNED_MEAN_MSE.items():
        assert S["aggregate"][k]["mean_mse"] == pytest.approx(v, abs=5e-3)
    a = S["aggregate"]
    assert a["LSTM"]["wins_qlike"] == 10 and a["LSTM"]["sig_better_qlike"] == 8 and a["LSTM"]["sig_better_mse"] == 0
    assert a["GARCH"]["sig_worse_qlike"] == 7 and a["HAR"]["wins_mse"] == 5


def test_summary_recomputes_from_forecasts():
    for t, row in S["per_ticker"].items():
        g = FC[FC["ticker"] == t]
        assert len(g) == row["n_oos"] == 3473
        assert str(g["date"].min().date()) == row["oos_first"] == "2013-01-02"
        for k, m in row["models"].items():
            assert np.mean(qlike(g["rv"], g[k])) == pytest.approx(m["qlike"], rel=1e-4)
            assert np.mean(mse(g["rv"], g[k])) == pytest.approx(m["mse"], rel=1e-4)
            if k != "HAR":
                s, _ = diebold_mariano(qlike(g["rv"], g[k]), qlike(g["rv"], g["HAR"]))
                assert s == pytest.approx(m["dm_qlike_vs_har"]["stat"], abs=2e-2)


def test_mse_concentration_recorded():
    assert S["per_ticker"]["XLK"]["har_mse_top5_days"]["dates"][0] == "2015-08-24"
    assert all(0 < r["har_mse_top5_days"]["share"] <= 1 for r in S["per_ticker"].values())


def test_oos_starts_after_training_window():
    assert (FC["date"] >= "2013-01-01").all()
