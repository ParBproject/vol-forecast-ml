"""Walk-forward engine, losses and Diebold-Mariano test."""
import numpy as np
import pandas as pd
from scipy import stats

OOS_START = "2013-01-01"  # first forecast target year; 2005-2012 is the initial training window


def walk_forward(f: pd.DataFrame, model_cls, oos_start: str = OOS_START) -> pd.DataFrame:
    """Expanding window, refit every calendar year. For each refit year Y the model is fit on rows with
    date < first trading day of Y minus one row (so the last training target rv[t+1] is still before Y),
    then forecasts rv[t+1] for every target t+1 in Y using rows <= t only."""
    dates = f.index
    rows = []
    years = sorted({d.year for d in dates if d >= pd.Timestamp(oos_start)})
    for y in years:
        tgt_pos = np.where(dates.year == y)[0]
        first = tgt_pos[0]
        train = f.iloc[:first]  # targets inside train are rv[1..first-1]; all before year Y
        m = model_cls().fit(train)
        hist = f.iloc[: tgt_pos[-1]]  # info up to the day before the last target
        pred = m.predict(hist)
        for p in tgt_pos:
            rows.append((dates[p], dates[p - 1], train.index[-1], float(pred.iloc[p - 1]), float(f["rv"].iloc[p])))
    out = pd.DataFrame(rows, columns=["target_date", "info_date", "train_end", "forecast", "rv"]).set_index("target_date")
    return out


def qlike(rv, f):
    r = np.asarray(rv) / np.asarray(f)
    return r - np.log(r) - 1


def mse(rv, f):
    return (np.asarray(rv) - np.asarray(f)) ** 2


def diebold_mariano(l_model, l_bench, lags: int = 5):
    """d = l_model - l_bench. Negative stat => model beats benchmark. Newey-West HAC variance, two-sided p."""
    d = np.asarray(l_model) - np.asarray(l_bench)
    n = len(d)
    dc = d - d.mean()
    v = dc @ dc / n
    for k in range(1, lags + 1):
        v += 2 * (1 - k / (lags + 1)) * (dc[k:] @ dc[:-k]) / n
    stat = d.mean() / np.sqrt(v / n)
    return float(stat), float(2 * stats.norm.sf(abs(stat)))
