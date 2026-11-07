"""Math of the forecasters: feature windows, HAR OLS, GARCH recursion, walk-forward dates.

Feature row t is known at the close of day t and is the regressor for rv[t+1].
HAR and the ML features use trailing windows of 5 and 22 trading days, including day t.
"""
import numpy as np
import pandas as pd
import pytest

from volml.evaluate import walk_forward
from volml.models import GARCH, GBM, HAR, LSTM, har_features, ml_features

# intercept, daily, weekly, monthly. Persistence 0.86 so the process is stationary.
HAR_BETA = np.array([0.04, 0.36, 0.28, 0.22])


def _har_path(n, beta, noise, seed):
    """rv[t] = c + b_d * rv[t-1] + b_w * mean(rv[t-5:t]) + b_m * mean(rv[t-22:t]) + shock."""
    c, b_d, b_w, b_m = beta
    rng = np.random.default_rng(seed)
    rv = np.empty(n)
    rv[:22] = rng.uniform(0.4, 2.0, 22)
    for t in range(22, n):
        shock = 0.0 if noise == 0 else rng.normal(0, noise)
        rv[t] = c + b_d * rv[t - 1] + b_w * rv[t - 5:t].mean() + b_m * rv[t - 22:t].mean() + shock
    return rv


def _garch_returns(n, mu, omega, alpha, beta, seed):
    """Constant-mean GARCH(1,1): sigma2[t] = omega + alpha*eps[t-1]^2 + beta*sigma2[t-1]."""
    rng = np.random.default_rng(seed)
    z = rng.standard_normal(n)
    eps = np.empty(n)
    sigma2 = np.empty(n)
    sigma2[0] = omega / (1 - alpha - beta)
    eps[0] = np.sqrt(sigma2[0]) * z[0]
    for t in range(1, n):
        sigma2[t] = omega + alpha * eps[t - 1] ** 2 + beta * sigma2[t - 1]
        eps[t] = np.sqrt(sigma2[t]) * z[t]
    return mu + eps


def test_har_features_are_trailing_and_causal():
    rv = np.arange(1, 31, dtype=float)  # values 1..30 at positions 0..29
    f = pd.DataFrame({"rv": rv}, index=pd.bdate_range("2021-01-04", periods=len(rv)))
    X = har_features(f)

    assert X.index.equals(f.index)
    assert list(X.columns) == ["d", "w", "m"]
    assert X["w"].iloc[:4].isna().all()
    assert X["m"].iloc[:21].isna().all()
    # position 21 holds rv 22. Weekly window is 18..22 (mean 20); monthly is 1..22 (mean 11.5).
    assert X["d"].iloc[21] == 22
    assert X["w"].iloc[21] == 20
    assert X["m"].iloc[21] == pytest.approx(11.5)

    for i in range(len(rv)):
        assert X["d"].iloc[i] == rv[i]
        if i >= 4:
            assert X["w"].iloc[i] == pytest.approx(rv[i - 4:i + 1].mean())
        else:
            assert np.isnan(X["w"].iloc[i])
        if i >= 21:
            assert X["m"].iloc[i] == pytest.approx(rv[i - 21:i + 1].mean())
        else:
            assert np.isnan(X["m"].iloc[i])

    # A shock at t must not change any regressor dated before t, and it must leave
    # the window once it rolls out (5 days for weekly, 22 for monthly).
    t = 6
    shocked = f.copy()
    shocked.iloc[t, shocked.columns.get_loc("rv")] = 1000.0
    Y = har_features(shocked)
    np.testing.assert_allclose(X.iloc[:t].to_numpy(), Y.iloc[:t].to_numpy(), equal_nan=True)
    assert Y["d"].iloc[t] == 1000.0
    assert Y["d"].iloc[t + 1] == X["d"].iloc[t + 1]
    assert Y["w"].iloc[t + 4] != X["w"].iloc[t + 4]
    assert Y["w"].iloc[t + 5] == pytest.approx(X["w"].iloc[t + 5])
    assert Y["m"].iloc[t + 21] != X["m"].iloc[t + 21]
    assert Y["m"].iloc[t + 22] == pytest.approx(X["m"].iloc[t + 22])


def test_ml_features_are_trailing_and_causal():
    rng = np.random.default_rng(0)
    n = 30
    f = pd.DataFrame({
        "rv": np.arange(1, n + 1, dtype=float),
        "r_oc": rng.normal(size=n),
        "r_on": rng.normal(size=n),
    })
    f.loc[10, "r_on"] = -1.5
    X = ml_features(f)
    lrv = np.log(f["rv"].to_numpy())

    assert list(X.columns) == ["lrv_d", "lrv_w", "lrv_m", "r_oc", "abs_r_on", "r_oc_w"]
    assert X.index.equals(f.index)
    assert X["lrv_w"].iloc[:4].isna().all() and X["r_oc_w"].iloc[:4].isna().all()
    assert X["lrv_m"].iloc[:21].isna().all()
    assert X["lrv_d"].iloc[7] == pytest.approx(np.log(8))
    assert X["lrv_w"].iloc[7] == pytest.approx(lrv[3:8].mean())
    assert X["lrv_m"].iloc[21] == pytest.approx(lrv[:22].mean())
    assert X["r_oc"].iloc[7] == pytest.approx(f["r_oc"].iloc[7])
    assert X["abs_r_on"].iloc[10] == 1.5
    assert X["r_oc_w"].iloc[7] == pytest.approx(f["r_oc"].iloc[3:8].sum())

    for i in range(n):
        assert X["lrv_d"].iloc[i] == pytest.approx(lrv[i])
        if i >= 4:
            assert X["lrv_w"].iloc[i] == pytest.approx(lrv[i - 4:i + 1].mean())
            assert X["r_oc_w"].iloc[i] == pytest.approx(f["r_oc"].iloc[i - 4:i + 1].sum())
        if i >= 21:
            assert X["lrv_m"].iloc[i] == pytest.approx(lrv[i - 21:i + 1].mean())

    t = 8
    shocked = f.copy()
    shocked.iloc[t, shocked.columns.get_loc("r_oc")] += 7
    Y = ml_features(shocked)
    np.testing.assert_allclose(X["lrv_d"], Y["lrv_d"])
    np.testing.assert_allclose(X["r_oc_w"].iloc[:t].to_numpy(), Y["r_oc_w"].iloc[:t].to_numpy(), equal_nan=True)
    assert Y["r_oc_w"].iloc[t + 4] != X["r_oc_w"].iloc[t + 4]
    assert Y["r_oc_w"].iloc[t + 5] == pytest.approx(X["r_oc_w"].iloc[t + 5])


def test_har_fit_recovers_known_process():
    exact = _har_path(600, HAR_BETA, noise=0.0, seed=1)
    fit = HAR().fit(pd.DataFrame({"rv": exact}))
    np.testing.assert_allclose(fit.beta, HAR_BETA, atol=1e-8)
    # forecast stored at t is rv[t+1], the day the regressors were built to predict
    pred = fit.predict(pd.DataFrame({"rv": exact}))
    np.testing.assert_allclose(pred.iloc[21:-1].to_numpy(), exact[22:], atol=1e-8)
    assert pred.iloc[:21].isna().all()

    noisy = _har_path(4000, HAR_BETA, noise=0.05, seed=2)
    recovered = HAR().fit(pd.DataFrame({"rv": noisy})).beta
    np.testing.assert_allclose(recovered, HAR_BETA, atol=0.03)


def test_har_fit_matches_least_squares():
    rng = np.random.default_rng(4)
    rv = rng.uniform(0.2, 3.0, 500)
    rows = np.arange(21, len(rv) - 1)  # full 22-day window, and a next-day target exists
    design = np.column_stack([
        np.ones(len(rows)),
        rv[rows],
        [rv[i - 4:i + 1].mean() for i in rows],
        [rv[i - 21:i + 1].mean() for i in rows],
    ])
    target = rv[rows + 1]
    expected, *_ = np.linalg.lstsq(design, target, rcond=None)

    fit = HAR().fit(pd.DataFrame({"rv": rv}))
    np.testing.assert_allclose(fit.beta, expected, atol=1e-8)

    pred = fit.predict(pd.DataFrame({"rv": rv}))
    linear = design @ expected
    np.testing.assert_allclose(pred.iloc[rows].to_numpy(), np.maximum(linear, fit.floor), atol=1e-8)
    assert fit.floor == pytest.approx(rv.min())


def test_garch_recovers_params_and_forecast_recursion():
    mu, omega, alpha, beta = 0.0, 0.05, 0.08, 0.90
    returns = _garch_returns(8000, mu, omega, alpha, beta, seed=1)
    f = pd.DataFrame({"r_oc": returns})
    g = GARCH().fit(f)

    assert g.w == pytest.approx(omega, abs=0.015)
    assert g.a == pytest.approx(alpha, abs=0.02)
    assert g.b == pytest.approx(beta, abs=0.02)
    assert g.w > 0 and g.a >= 0 and g.b >= 0
    assert g.a + g.b < 1

    # One-step variance from the fitted params. sigma2 starts at the training
    # sample variance, then sigma2 <- omega + alpha*eps^2 + beta*sigma2 after each return.
    # predict()[t] is that variance for t+1.
    assert g.s0 == pytest.approx(float(f["r_oc"].var(ddof=1)))
    eps2 = (returns - g.mu) ** 2
    sigma2 = g.s0
    hand = np.empty(len(returns))
    for t in range(len(returns)):
        sigma2 = g.w + g.a * eps2[t] + g.b * sigma2
        hand[t] = sigma2
    assert hand[0] == pytest.approx(g.w + g.a * eps2[0] + g.b * g.s0)
    assert hand[1] == pytest.approx(g.w + g.a * eps2[1] + g.b * hand[0])
    np.testing.assert_allclose(g.predict(f).to_numpy(), hand, rtol=1e-12, atol=0)

    later = f.copy()
    later.iloc[-10:, later.columns.get_loc("r_oc")] += 5
    original = g.predict(f)
    perturbed = g.predict(later)
    np.testing.assert_allclose(original.iloc[:-10].to_numpy(), perturbed.iloc[:-10].to_numpy())
    assert not np.isclose(original.iloc[-10], perturbed.iloc[-10])


def test_gbm_forecast_uses_half_residual_variance():
    rng = np.random.default_rng(0)
    n = 280
    f = pd.DataFrame({
        "rv": rng.uniform(0.3, 3.0, n),
        "r_oc": rng.normal(size=n),
        "r_on": rng.normal(scale=0.4, size=n),
    }, index=pd.bdate_range("2014-01-02", periods=n))
    model = GBM().fit(f)
    X = ml_features(f)
    complete = X.notna().to_numpy().all(axis=1)
    train_idx = np.flatnonzero(complete[:-1])  # last row has features but no next-day rv
    log_next = np.log(f["rv"].to_numpy()[train_idx + 1])
    fitted = model.m.predict(X.iloc[train_idx])
    # pandas Series.var (ddof=1), which is what GBM.fit uses for the lognormal correction
    assert model.adj == pytest.approx(0.5 * np.var(log_next - fitted, ddof=1))

    pred = model.predict(f)
    assert pred.iloc[:21].isna().all()
    ok = np.flatnonzero(complete)
    expected = np.exp(model.m.predict(X.iloc[ok]) + model.adj)
    np.testing.assert_allclose(pred.iloc[ok].to_numpy(), expected)
    assert np.isfinite(pred.iloc[-1])  # last row forecasts the day after the sample


def test_lstm_windows_are_trailing_and_causal():
    """Window alignment only. Fitting the network is covered by the causality tests, and is slow."""
    rng = np.random.default_rng(1)
    n = 80
    f = pd.DataFrame({
        "rv": rng.uniform(0.2, 2.0, n),
        "r_oc": rng.normal(size=n),
        "r_on": rng.normal(size=n),
    })
    model = LSTM()
    X = ml_features(f)
    model.mu = X.mean()
    model.sd = X.std()
    seq, idx = model._seq(f)

    complete = X.notna().to_numpy().all(axis=1)
    expected_idx = [i for i in range(model.SEQ - 1, n) if complete[i - model.SEQ + 1:i + 1].all()]
    assert idx == expected_idx
    # 22 days to fill the monthly feature, then 22 complete rows: first window ends at index 42.
    assert expected_idx[0] == 42
    assert seq.shape == (len(expected_idx), 22, X.shape[1])

    end = expected_idx[0]
    scaled = ((X - model.mu) / model.sd).to_numpy(dtype=np.float32)
    np.testing.assert_array_equal(seq[0], scaled[end - 21:end + 1])

    shocked = f.copy()
    shocked.iloc[end + 1, shocked.columns.get_loc("rv")] *= 4
    other = LSTM()
    other.mu, other.sd = model.mu, model.sd
    seq2, idx2 = other._seq(shocked)
    assert idx2 == idx
    np.testing.assert_array_equal(seq[0], seq2[0])  # window ending at `end` does not see end+1
    k = idx.index(end + 1)
    assert not np.allclose(seq[k], seq2[k])


def test_walk_forward_lines_up_with_target_dates():
    class ScaleByTrain:
        """Forecast at the info date is 10 * rv[info] + number of training rows."""

        def fit(self, train):
            self.n = len(train)
            return self

        def predict(self, hist):
            return hist["rv"] * 10 + self.n

    dates = pd.to_datetime([
        "2018-12-27", "2018-12-28", "2018-12-31",
        "2019-01-02", "2019-01-03", "2019-01-04", "2019-06-03", "2019-12-31",
        "2020-01-02", "2020-01-03", "2020-07-01",
    ])
    rv = np.arange(1, len(dates) + 1, dtype=float)
    f = pd.DataFrame({"rv": rv, "r_oc": 0.0, "r_on": 0.0}, index=dates)
    oos_start = "2019-01-02"  # first session of the first OOS year

    out = walk_forward(f, ScaleByTrain, oos_start)
    oos = dates[dates >= pd.Timestamp(oos_start)]

    assert out.index.name == "target_date"
    assert out.index.equals(oos)
    assert out.index[0] == pd.Timestamp(oos_start)
    assert out.index.is_unique and len(out) == len(oos)
    assert (out.index.year >= 2019).all()
    assert "2018-12-31" not in out.index

    for target, row in out.iterrows():
        pos = dates.get_loc(target)
        year_first = dates[dates.year == target.year][0]
        n_train = int(dates.get_loc(year_first))
        assert row["info_date"] == dates[pos - 1]
        assert row["train_end"] == dates[n_train - 1]
        assert row["rv"] == rv[pos]
        assert row["forecast"] == pytest.approx(10 * rv[pos - 1] + n_train)
        assert row["forecast"] != row["rv"]
