"""Walk-forward integrity (no lookahead), loss functions and the DM test."""
import numpy as np
import pandas as pd
import pytest

from volml.data import frame
from volml.evaluate import diebold_mariano, mse, qlike, walk_forward
from volml.models import GARCH, GBM, HAR, LSTM

F = frame("SPY").loc[:"2016-12-31"]
OOS = "2015-01-01"


@pytest.mark.parametrize("M", [HAR, GARCH, GBM, LSTM])
def test_timestamps_strictly_causal(M):
    fc = walk_forward(F, M, OOS)
    assert (fc["info_date"] < fc.index).all()
    assert (fc["train_end"] < fc.index).all()
    # the model used for year Y was trained only on rows dated before Y
    assert all(pd.Timestamp(te).year < d.year for d, te in zip(fc.index, fc["train_end"], strict=True))
    # info_date is exactly the previous trading day
    pos = F.index.get_indexer(fc.index)
    assert (F.index[pos - 1] == fc["info_date"]).all()
    assert fc["forecast"].notna().all() and (fc["forecast"] > 0).all()


@pytest.mark.parametrize("M", [HAR, GARCH, GBM, LSTM])
def test_future_data_cannot_change_past_forecasts(M):
    """Scramble everything after a cut date; forecasts for targets up to the cut must be bit-identical."""
    cut = pd.Timestamp("2016-06-30")
    base = walk_forward(F, M, OOS)
    G = F.copy()
    late = G.index > cut
    rng = np.random.default_rng(1)
    G.loc[late, "rv"] = rng.uniform(1e-3, 50, late.sum())
    G.loc[late, ["r_oc", "r_on"]] = rng.normal(0, 5, (late.sum(), 2))
    pert = walk_forward(G, M, OOS)
    keep = base.index <= cut
    np.testing.assert_array_equal(base.loc[keep, "forecast"].to_numpy(), pert.loc[keep, "forecast"].to_numpy())


def test_losses():
    assert qlike([2.0], [2.0])[0] == 0 and qlike([1.0], [2.0])[0] > 0 and qlike([4.0], [2.0])[0] > 0
    assert mse([3.0], [1.0])[0] == 4


def test_dm_sign_and_null():
    rng = np.random.default_rng(0)
    a = rng.normal(1, 1, 2000)
    s, p = diebold_mariano(a - 0.5, a + rng.normal(0, 0.1, 2000))
    assert s < 0 and p < 1e-6
    s, p = diebold_mariano(a, a + rng.normal(0, 1, 2000))
    assert p > 0.01
