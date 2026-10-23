"""Run the full walk-forward study and write results/summary.json + results/forecasts.csv.gz."""
import json

import numpy as np
import pandas as pd

from volml.data import ROOT, TICKERS, frame
from volml.evaluate import OOS_START, diebold_mariano, mse, qlike, walk_forward
from volml.models import MODELS

RES = ROOT / "results"


def run(tickers=TICKERS) -> dict:
    RES.mkdir(exist_ok=True)
    allf, per = [], {}
    for t in tickers:
        f = frame(t)
        fc = {M.name: walk_forward(f, M) for M in MODELS}
        base = fc["HAR"]
        common = base.index
        for k in fc:
            common = common.intersection(fc[k].dropna().index)
        rv = base.loc[common, "rv"]
        L = {k: {"qlike": qlike(rv, fc[k].loc[common, "forecast"]), "mse": mse(rv, fc[k].loc[common, "forecast"])} for k in fc}
        row = {"n_oos": int(len(common)), "oos_first": str(common[0].date()), "oos_last": str(common[-1].date()), "models": {}}
        for k in fc:
            r = {"qlike": float(L[k]["qlike"].mean()), "mse": float(L[k]["mse"].mean())}
            if k != "HAR":
                for loss in ("qlike", "mse"):
                    s, p = diebold_mariano(L[k][loss], L["HAR"][loss])
                    r[f"dm_{loss}_vs_har"] = {"stat": round(s, 3), "p": round(p, 4)}
            row["models"][k] = r
        e = pd.Series(L["HAR"]["mse"], index=common)
        top = e.nlargest(5)
        row["har_mse_top5_days"] = {"share": round(float(top.sum() / e.sum()), 4), "dates": [str(d.date()) for d in top.index]}
        row["best_qlike"] = min(fc, key=lambda k: row["models"][k]["qlike"])
        row["best_mse"] = min(fc, key=lambda k: row["models"][k]["mse"])
        per[t] = row
        df = pd.DataFrame({k: fc[k].loc[common, "forecast"] for k in fc})
        df.insert(0, "rv", rv)
        df.insert(0, "ticker", t)
        allf.append(df)
        print(t, {k: round(v["qlike"], 4) for k, v in row["models"].items()})
    names = [M.name for M in MODELS]
    agg = {}
    for k in names:
        q = [per[t]["models"][k]["qlike"] for t in tickers]
        m = [per[t]["models"][k]["mse"] for t in tickers]
        a = {"mean_qlike": float(np.mean(q)), "mean_mse": float(np.mean(m)),
             "wins_qlike": sum(per[t]["best_qlike"] == k for t in tickers),
             "wins_mse": sum(per[t]["best_mse"] == k for t in tickers)}
        if k != "HAR":
            for loss in ("qlike", "mse"):
                dm = [per[t]["models"][k][f"dm_{loss}_vs_har"] for t in tickers]
                a[f"sig_better_{loss}"] = sum(d["stat"] < 0 and d["p"] < 0.05 for d in dm)
                a[f"sig_worse_{loss}"] = sum(d["stat"] > 0 and d["p"] < 0.05 for d in dm)
        agg[k] = a
    summary = {
        "target": "next-day Garman-Klass open-to-close variance (percent^2), from daily OHLC",
        "tickers": list(tickers), "models": names, "oos_start": OOS_START,
        "refit": "expanding window, refit once per calendar year", "dm_hac_lags": 5,
        "per_ticker": per, "aggregate": agg,
    }
    (RES / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    out = pd.concat(allf)
    out.index.name = "date"
    out.round(6).to_csv(RES / "forecasts.csv.gz", compression={"method": "gzip", "mtime": 0})
    return summary


if __name__ == "__main__":
    run()
