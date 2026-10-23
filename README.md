# vol-forecast-ml

One-day-ahead **realized-volatility forecasting** for SPY and the nine SPDR sector ETFs (XLK, XLF, XLE, XLV, XLY, XLP, XLI, XLU, XLB).
Two econometric baselines, **HAR-RV** and **GARCH(1,1)**, are compared against **gradient boosting** and a **small LSTM**
in a strictly walk-forward test, scored with QLIKE and MSE, with Diebold-Mariano tests against HAR. Every number below
is generated from `results/summary.json` by `scripts/build_readme.py` and pinned by tests.

**Live app: [parbproject.github.io/vol-forecast-ml](https://parbproject.github.io/vol-forecast-ml/)**

| Desktop | Mobile |
|---|---|
| ![Overview](docs/img/overview.png) | ![Mobile](docs/img/mobile.png) |

## Headline (honest result)

- Out-of-sample: 3,473 trading days per ETF, 2013-01-02 to 2026-10-22, 10 ETFs.
- **QLIKE:** the LSTM has the lowest loss on 10 of 10 ETFs; mean QLIKE 0.2968 vs HAR
  0.3262 (-9.0%). Its Diebold-Mariano improvement over HAR is significant at 5% on
  8 of 10. Gradient boosting is roughly level with HAR (-0.3%; significantly better
  on 2, worse on 2). GARCH is the worst on QLIKE (significantly worse than HAR on
  7).
- **MSE:** HAR has the lowest MSE on 5 of 10 ETFs and gradient boosting on 4. No model's MSE difference vs HAR
  is significant on any ETF. Squared error is dominated by a few crash days: the worst 5 days make up
  40% to 94% of HAR's total MSE per ETF (mostly 2015-08-24 and March 2020).
  So MSE cannot separate the models.
- Takeaway: the LSTM's edge is real but only in the *proportional* (QLIKE) sense. On squared error it is no better than
  HAR, so this is not evidence that it forecasts the large volatility spikes better.

## Mean loss across the 10 ETFs

| Model | QLIKE | vs HAR | QLIKE wins | DM sig. better / worse | MSE | vs HAR | MSE wins | DM sig. better / worse |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| HAR | 0.3262 | +0.0% | 0 | - / - | 10.212 | +0.0% | 5 | - / - |
| GARCH | 0.3473 | +6.5% | 0 | 0 / 7 | 10.350 | +1.3% | 0 | 0 / 0 |
| GBM | 0.3251 | -0.3% | 0 | 2 / 2 | 9.327 | -8.7% | 4 | 0 / 0 |
| LSTM | 0.2968 | -9.0% | 10 | 8 / 0 | 9.803 | -4.0% | 1 | 0 / 0 |

## Per-ETF QLIKE (best in bold; DM stat vs HAR, negative = better than HAR, * = p < 0.05)

| ETF | HAR | GARCH | GBM | LSTM | DM GARCH | DM GBM | DM LSTM |
|---|---:|---:|---:|---:|---:|---:|---:|
| SPY | 0.3978 | 0.4319 | 0.4073 | **0.3701** | +2.95* | +0.93 | -2.81* |
| XLK | 0.4250 | 0.4127 | 0.3975 | **0.3698** | -1.35 | -2.55* | -4.13* |
| XLF | 0.3656 | 0.3779 | 0.3613 | **0.3383** | +0.63 | -0.19 | -1.00 |
| XLE | 0.2482 | 0.2738 | 0.2440 | **0.2295** | +2.79* | -0.86 | -5.08* |
| XLV | 0.3769 | 0.3632 | 0.3391 | **0.3022** | -1.52 | -2.99* | -5.16* |
| XLY | 0.3297 | 0.3595 | 0.3323 | **0.3009** | +2.41* | +0.29 | -3.43* |
| XLP | 0.2842 | 0.3166 | 0.3072 | **0.2740** | +3.97* | +3.08* | -1.78 |
| XLI | 0.3211 | 0.3545 | 0.3314 | **0.2997** | +3.20* | +1.26 | -4.82* |
| XLU | 0.2414 | 0.2696 | 0.2567 | **0.2298** | +3.92* | +3.22* | -4.20* |
| XLB | 0.2716 | 0.3129 | 0.2747 | **0.2537** | +4.12* | +0.41 | -2.90* |

## Per-ETF MSE (daily variance in %², best in bold)

| ETF | HAR | GARCH | GBM | LSTM | DM GARCH | DM GBM | DM LSTM |
|---|---:|---:|---:|---:|---:|---:|---:|
| SPY | **1.839** | 2.713 | 2.139 | 2.500 | +1.38 | +1.02 | +1.01 |
| XLK | 28.041 | 25.827 | **25.024** | 25.577 | -0.82 | -1.12 | -0.90 |
| XLF | 28.989 | 25.578 | **24.292** | 25.149 | -0.80 | -1.12 | -0.91 |
| XLE | 7.165 | 10.570 | **7.071** | 8.585 | +1.40 | -0.09 | +1.29 |
| XLV | 21.762 | 20.179 | 20.009 | **19.963** | -1.04 | -1.13 | -1.15 |
| XLY | 3.548 | 4.664 | **3.085** | 3.986 | +1.25 | -1.59 | +0.74 |
| XLP | **1.081** | 1.293 | 1.185 | 1.277 | +1.12 | +1.09 | +1.00 |
| XLI | **5.142** | 6.154 | 5.366 | 5.437 | +1.23 | +0.68 | +0.61 |
| XLU | **1.830** | 2.603 | 1.872 | 2.003 | +1.65 | +0.25 | +0.62 |
| XLB | **2.725** | 3.915 | 3.228 | 3.554 | +1.33 | +0.91 | +0.95 |

## Method

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
  variance (5 lags) and two-sided normal p-values. All models are scored on the same dates.

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
