"""Write README.md from results/summary.json and the frozen raw files."""
import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
S = json.loads((ROOT / "results" / "summary.json").read_text())
META = json.loads((ROOT / "data" / "raw" / "meta.json").read_text())
MANIFEST = json.loads((ROOT / "data" / "MANIFEST.json").read_text())
P, A, M = S["per_ticker"], S["aggregate"], S["models"]
T = S["tickers"]
first = P[T[0]]


def dm(t, k, loss):
    d = P[t]["models"][k][f"dm_{loss}_vs_har"]
    star = "*" if d["p"] < 0.05 else ""
    return f"{d['stat']:+.2f}{star}"


def rel(k, loss):
    return (A[k][f"mean_{loss}"] / A["HAR"][f"mean_{loss}"] - 1) * 100


def mse_concentration():
    shares = [P[t]["har_mse_top5_days"]["share"] for t in T]
    lo, hi = min(shares), max(shares)
    march_n = 0
    counts = {}
    for t in T:
        dates = P[t]["har_mse_top5_days"]["dates"]
        if any(d.startswith("2020-03") for d in dates):
            march_n += 1
        for d in dates:
            if not d.startswith("2020-03"):
                counts[d] = counts.get(d, 0) + 1
    bits = []
    if march_n:
        bits.append((march_n, f"March 2020 ({march_n} of {len(T)})"))
    bits.extend((n, f"{d} ({n} of {len(T)})") for d, n in counts.items() if n >= 5)
    bits.sort(key=lambda x: (-x[0], x[1]))
    where = ", ".join(text for _, text in bits)
    sentence = f"The five largest HAR squared errors are {lo:.0%} to {hi:.0%} of each ETF's total MSE"
    if where:
        sentence += f", mostly {where}"
    return sentence + "."


def xli_print_sentence(day="2019-02-26"):
    high = close = None
    with (ROOT / "data" / "raw" / "XLI.csv").open() as f:
        for row in csv.DictReader(f):
            if row["date"] == day:
                high, close = float(row["High"]), float(row["Close"])
                break
    if high is None or close <= 0 or high < close * 1.05:
        return "The Yahoo bars are used as stored, with no edits."
    dates = P["XLI"]["har_mse_top5_days"]["dates"]
    rank = ""
    if day in dates:
        rank = f" That day ranks {dates.index(day) + 1} among XLI's five largest HAR squared errors."
    return (
        f"The Yahoo bars are used as stored, with no edits. On {day}, XLI's high in `data/raw/XLI.csv` "
        f"is {high:.2f} and the close is {close:.2f}.{rank} The row is left as downloaded."
    )


q_rows = "\n".join(
    f"| {t} | " + " | ".join(
        f"{'**' if P[t]['best_qlike'] == k else ''}{P[t]['models'][k]['qlike']:.4f}"
        f"{'**' if P[t]['best_qlike'] == k else ''}" for k in M)
    + " | " + " | ".join(dm(t, k, "qlike") for k in M[1:]) + " |" for t in T)
m_rows = "\n".join(
    f"| {t} | " + " | ".join(
        f"{'**' if P[t]['best_mse'] == k else ''}{P[t]['models'][k]['mse']:.3f}"
        f"{'**' if P[t]['best_mse'] == k else ''}" for k in M)
    + " | " + " | ".join(dm(t, k, "mse") for k in M[1:]) + " |" for t in T)
agg_rows = "\n".join(
    f"| {k} | {A[k]['mean_qlike']:.4f} | {rel(k, 'qlike'):+.1f}% | {A[k]['wins_qlike']} | "
    f"{A[k].get('sig_better_qlike', '-')} / {A[k].get('sig_worse_qlike', '-')} | {A[k]['mean_mse']:.3f} | "
    f"{rel(k, 'mse'):+.1f}% | {A[k]['wins_mse']} | "
    f"{A[k].get('sig_better_mse', '-')} / {A[k].get('sig_worse_mse', '-')} |"
    for k in M)
L = A["LSTM"]
n = len(T)
text = f"""# vol-forecast-ml

[![CI](https://github.com/ParBproject/vol-forecast-ml/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/ParBproject/vol-forecast-ml/actions/workflows/ci.yml)

One-day-ahead realized-volatility forecasts for SPY and the nine SPDR sector ETFs ({", ".join(T[1:])}).
HAR-RV and GARCH(1,1) are compared with gradient boosting and a small LSTM.
Evaluation is walk-forward, with QLIKE, MSE, and Diebold-Mariano tests against HAR.
`scripts/build_readme.py` writes the numbers below from `results/summary.json`.

**Live app: [parbproject.github.io/vol-forecast-ml](https://parbproject.github.io/vol-forecast-ml/)**

| Desktop | Mobile |
|---|---|
| ![Overview](docs/img/overview.png) | ![Mobile](docs/img/mobile.png) |

## Results

- Out of sample: {first['n_oos']:,} trading days per ETF, {first['oos_first']} to {first['oos_last']}, {n} ETFs.
- QLIKE: the LSTM is lowest on {L['wins_qlike']} of {n} ETFs. Mean QLIKE is {L['mean_qlike']:.4f} versus {A['HAR']['mean_qlike']:.4f} for HAR ({rel('LSTM', 'qlike'):+.1f}%). The Diebold-Mariano improvement over HAR is significant at 5% on {L['sig_better_qlike']} of {n}. Gradient boosting's mean QLIKE is {rel('GBM', 'qlike'):+.1f}% versus HAR (significantly better on {A['GBM']['sig_better_qlike']}, worse on {A['GBM']['sig_worse_qlike']}). GARCH's mean QLIKE is {rel('GARCH', 'qlike'):+.1f}% versus HAR and significantly worse on {A['GARCH']['sig_worse_qlike']}.
- MSE: HAR is lowest on {A['HAR']['wins_mse']} of {n} ETFs, gradient boosting on {A['GBM']['wins_mse']}, and the LSTM on {L['wins_mse']}. No model's MSE difference versus HAR is significant at 5% on any ETF. {mse_concentration()} Squared error does not separate the models. The QLIKE gap does not show that the LSTM forecasts the large variance jumps more closely than HAR.

## Mean loss across the {n} ETFs

| Model | QLIKE | vs HAR | QLIKE wins | DM sig. better / worse | MSE | vs HAR | MSE wins | DM sig. better / worse |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
{agg_rows}

## Per-ETF QLIKE (best in bold; DM stat vs HAR, negative = better than HAR, * = p < 0.05)

| ETF | {" | ".join(M)} | {" | ".join(f"DM {k}" for k in M[1:])} |
|---|{"---:|" * (2 * len(M) - 1)}
{q_rows}

## Per-ETF MSE (daily variance in %², best in bold)

| ETF | {" | ".join(M)} | {" | ".join(f"DM {k}" for k in M[1:])} |
|---|{"---:|" * (2 * len(M) - 1)}
{m_rows}

## Data

- Prices: {META['source']}, {len(META['tickers'])} ETFs ({", ".join(META['tickers'])}). Requested {META['start']} to {META['end_exclusive']} (end exclusive). Downloaded {META['downloaded_utc']}.
- `data/raw/` has {len(MANIFEST)} files (one CSV per ETF and `meta.json`). SHA-256 hashes are in `data/MANIFEST.json`. `make verify` fails if a file has changed.
- `make fetch` downloads the files again. That changes the hashes and the results.
- The CSVs are in the repo so this study can be rerun. Yahoo's terms still apply if the files are reused.

## Method

- The target is next-day Garman-Klass variance from open, high, low, and close, in %². It is an open-to-close range proxy, not intraday realized variance.
- Walk-forward, expanding window. The first training sample is 2005-2012. Each model is refit once per calendar year on rows dated before that year, and the forecast for day t+1 uses rows through day t only. `tests/test_methods.py` checks the timestamps and checks that scrambling later rows leaves earlier forecasts unchanged.
- HAR-RV: OLS of rv[t+1] on the daily, 5-day, and 22-day mean rv, in levels.
- GARCH(1,1): constant mean, normal errors, fit with `arch` on open-to-close percent returns.
- Gradient boosting: scikit-learn `HistGradientBoostingRegressor` with library defaults, on log rv. Features are log rv (daily, weekly, monthly), the open-to-close return, the 5-day return sum, and the absolute overnight gap. The forecast is exp(prediction + half the training residual variance).
- LSTM: PyTorch, one layer of 16 units, 22-day windows of the same features, 15 epochs, Adam at 1e-3, seed 0, log-rv target.
- Those settings are constants in `src/volml/models.py`. There is no search over them.
- Losses are QLIKE = rv/f - log(rv/f) - 1 and MSE on the variance. Diebold-Mariano tests compare each model with HAR using Newey-West variance ({S['dm_hac_lags']} lags) and two-sided normal p-values. All models are scored on the same dates.

## Caveats

- QLIKE fits a log target, and MSE fits OLS in levels. Both losses are reported.
- {xli_print_sentence()}
- The {n} ETFs move together, so {n} tests are not {n} separate studies. There is no multiple-testing correction.
- LSTM training uses a fixed seed. Another PyTorch build can move the forecasts slightly.

## Reproduce

```bash
python -m venv .venv && .venv/bin/pip install -e ".[dev]"
make all      # verify hashes, walk-forward study (~4 min CPU), README, web data, ruff, pytest
make web      # web app tests + build (Node 20+)
make deploy   # publish web/dist to gh-pages
```

## License

MIT
"""
(ROOT / "README.md").write_text(text)
print("wrote README.md")
