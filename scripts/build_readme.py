"""Generate README.md from results/summary.json (no hand-typed numbers)."""
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


q_rows = "\n".join(
    f"| {t} | " + " | ".join(f"{'**' if P[t]['best_qlike'] == k else ''}{P[t]['models'][k]['qlike']:.4f}"
                             f"{'**' if P[t]['best_qlike'] == k else ''}" for k in M)
    + " | " + " | ".join(dm(t, k, "qlike") for k in M[1:]) + " |" for t in T)
m_rows = "\n".join(
    f"| {t} | " + " | ".join(f"{'**' if P[t]['best_mse'] == k else ''}{P[t]['models'][k]['mse']:.3f}"
                             f"{'**' if P[t]['best_mse'] == k else ''}" for k in M)
    + " | " + " | ".join(dm(t, k, "mse") for k in M[1:]) + " |" for t in T)
agg_rows = "\n".join(
    f"| {k} | {A[k]['mean_qlike']:.4f} | {rel(k, 'qlike'):+.1f}% | {A[k]['wins_qlike']} | "
    f"{A[k].get('sig_better_qlike', '-')} / {A[k].get('sig_worse_qlike', '-')} | {A[k]['mean_mse']:.3f} | {rel(k, 'mse'):+.1f}% | "
    f"{A[k]['wins_mse']} | {A[k].get('sig_better_mse', '-')} / {A[k].get('sig_worse_mse', '-')} |" for k in M)
L = A["LSTM"]
PROVENANCE = "\n".join([
    "## Data provenance",
    "",
    f"- **Prices:** {META['source']}, for {len(META['tickers'])} ETFs ({', '.join(META['tickers'])}).",
    f"  Requested {META['start']} to {META['end_exclusive']} (end exclusive); downloaded {META['downloaded_utc']} (UTC).",
    f"- **Frozen snapshot:** {len(MANIFEST)} files in `data/raw/` (one CSV per ETF plus `meta.json`), each SHA-256",
    "  hashed in `data/MANIFEST.json`. `make verify` (part of `make all`) fails if any file has changed.",
    "- **Refresh:** `make fetch` re-downloads everything, which changes the hashes and every result.",
    "- **Terms:** the snapshot is committed only so the study reproduces. Check Yahoo's terms before reusing it.",
    "",
    "",
])
text = f"""# vol-forecast-ml

[![CI](https://github.com/ParBproject/vol-forecast-ml/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/ParBproject/vol-forecast-ml/actions/workflows/ci.yml)

One-day-ahead **realized-volatility forecasting** for SPY and the nine SPDR sector ETFs ({", ".join(T[1:])}).
Two econometric baselines, **HAR-RV** and **GARCH(1,1)**, are compared against **gradient boosting** and a **small LSTM**
in a strictly walk-forward test, scored with QLIKE and MSE, with Diebold-Mariano tests against HAR. Every number below
is generated from `results/summary.json` by `scripts/build_readme.py` and pinned by tests.

**Live app: [parbproject.github.io/vol-forecast-ml](https://parbproject.github.io/vol-forecast-ml/)**

| Desktop | Mobile |
|---|---|
| ![Overview](docs/img/overview.png) | ![Mobile](docs/img/mobile.png) |

## Headline (honest result)

- Out-of-sample: {first['n_oos']:,} trading days per ETF, {first['oos_first']} to {first['oos_last']}, {len(T)} ETFs.
- **QLIKE:** the LSTM has the lowest loss on {L['wins_qlike']} of {len(T)} ETFs; mean QLIKE {L['mean_qlike']:.4f} vs HAR
  {A['HAR']['mean_qlike']:.4f} ({rel('LSTM', 'qlike'):+.1f}%). Its Diebold-Mariano improvement over HAR is significant at 5% on
  {L['sig_better_qlike']} of {len(T)}. Gradient boosting is roughly level with HAR ({rel('GBM', 'qlike'):+.1f}%; significantly better
  on {A['GBM']['sig_better_qlike']}, worse on {A['GBM']['sig_worse_qlike']}). GARCH is the worst on QLIKE (significantly worse than HAR on
  {A['GARCH']['sig_worse_qlike']}).
- **MSE:** HAR has the lowest MSE on {A['HAR']['wins_mse']} of {len(T)} ETFs and gradient boosting on {A['GBM']['wins_mse']}. No model's MSE difference vs HAR
  is significant on any ETF. Squared error is dominated by a few crash days: the worst 5 days make up
  {min(P[t]['har_mse_top5_days']['share'] for t in T):.0%} to {max(P[t]['har_mse_top5_days']['share'] for t in T):.0%} of HAR's total MSE per ETF (mostly 2015-08-24 and March 2020).
  So MSE cannot separate the models.
- Takeaway: the LSTM's edge is real but only in the *proportional* (QLIKE) sense. On squared error it is no better than
  HAR, so this is not evidence that it forecasts the large volatility spikes better.

## Mean loss across the {len(T)} ETFs

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

{PROVENANCE}## Method

- **Data:** daily OHLC from Yahoo Finance via yfinance, 2005-01-03 to 2026-10-22, frozen in `data/raw` with SHA-256 hashes in
  `data/MANIFEST.json` (`make verify`). `make all` runs fully offline.
- **Target:** next-day Garman-Klass variance from the open/high/low/close, in %². This is a range-based proxy computed from daily
  bars, not intraday 5-minute realized variance (not freely available), and it covers the open-to-close session only.
- **Walk-forward:** expanding window. 2005-2012 is the first training window. Every model is refit once per calendar year
  on data dated strictly before that year, and each forecast of day t+1 uses rows up to day t only. Tests assert this with
  timestamps and by scrambling all data after a cut date and checking earlier forecasts are bit-identical.
- **HAR-RV:** OLS of rv[t+1] on daily, weekly (5d) and monthly (22d) mean rv, in levels.
- **GARCH(1,1):** constant mean, normal errors, fit with `arch` on open-to-close % returns (the same session as the target).
- **Gradient boosting:** scikit-learn `HistGradientBoostingRegressor` with library defaults on log rv. Features: log rv
  daily/weekly/monthly, open-to-close return, 5-day return sum and absolute overnight gap. Prediction exp(f + s²/2).
- **LSTM:** PyTorch, 1 layer of 16 units on 22-day windows of the same features, 15 epochs, Adam 1e-3, seed 0, log-rv target.
- **No tuning:** every setting above was fixed before the first out-of-sample run and has not been changed since. No
  hyperparameter search.
- **Losses and tests:** QLIKE = rv/f - log(rv/f) - 1 and MSE on variance. Diebold-Mariano test vs HAR with Newey-West
  variance ({S['dm_hac_lags']} lags) and two-sided normal p-values. All models are scored on the same dates.

## Caveats

- Log-target models are naturally better suited to QLIKE, and OLS in levels to MSE. Both losses are reported for this reason.
- Yahoo daily bars are used as delivered, with no cleaning. At least one suspect print is in the sample: XLI's 2019-02-26 high
  of 82.66 against a 76.30 close. Cleaning it after seeing results would be data snooping, so it stays.
- 10 tests on 10 correlated ETFs are not 10 independent confirmations. No multiple-testing correction is applied.
- The LSTM is trained on CPU with a fixed seed. Results are reproducible on the same library versions (torch 2.14 CPU) but
  can move slightly on other builds.

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
