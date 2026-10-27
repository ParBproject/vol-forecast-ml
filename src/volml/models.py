"""Forecasters. Each `fit(train)` sees only rows <= the training cut; `predict(hist)` uses rows <= t to forecast rv[t+1]."""
import numpy as np
import pandas as pd
from arch import arch_model
from sklearn.ensemble import HistGradientBoostingRegressor


def har_features(f: pd.DataFrame) -> pd.DataFrame:
    rv = f["rv"]
    return pd.DataFrame({"d": rv, "w": rv.rolling(5).mean(), "m": rv.rolling(22).mean()})


def ml_features(f: pd.DataFrame) -> pd.DataFrame:
    lrv = np.log(f["rv"])
    return pd.DataFrame({
        "lrv_d": lrv, "lrv_w": lrv.rolling(5).mean(), "lrv_m": lrv.rolling(22).mean(),
        "r_oc": f["r_oc"], "abs_r_on": f["r_on"].abs(),
        "r_oc_w": f["r_oc"].rolling(5).sum(),
    })


class HAR:
    name = "HAR"

    def fit(self, f):
        X = har_features(f)
        y = f["rv"].shift(-1)
        ok = X.notna().all(1) & y.notna()
        A = np.column_stack([np.ones(ok.sum()), X[ok].to_numpy()])
        self.beta, *_ = np.linalg.lstsq(A, y[ok].to_numpy(), rcond=None)
        self.floor = float(f["rv"].min())
        return self

    def predict(self, f):
        X = har_features(f).to_numpy()
        p = self.beta[0] + X @ self.beta[1:]
        return pd.Series(np.maximum(p, self.floor), index=f.index)


class GARCH:
    """GARCH(1,1), constant mean, normal errors, on open-to-close % returns (same window as the GK proxy)."""
    name = "GARCH"

    def fit(self, f):
        res = arch_model(f["r_oc"], mean="Constant", vol="GARCH", p=1, q=1, dist="normal", rescale=False).fit(disp="off")
        self.mu, self.w, self.a, self.b = (float(res.params[k]) for k in ("mu", "omega", "alpha[1]", "beta[1]"))
        self.s0 = float(f["r_oc"].var())
        return self

    def predict(self, f):
        e2 = (f["r_oc"].to_numpy() - self.mu) ** 2
        s = np.empty(len(e2))
        prev = self.s0
        for i in range(len(e2)):  # sigma2[t+1] = w + a*e2[t] + b*sigma2[t]
            prev = self.w + self.a * e2[i] + self.b * prev
            s[i] = prev
        return pd.Series(s, index=f.index)


class GBM:
    """sklearn HistGradientBoostingRegressor, library defaults, on log rv.

    Forecast is exp(pred + var(train resid)/2), the lognormal bias correction."""
    name = "GBM"

    def fit(self, f):
        X = ml_features(f)
        y = np.log(f["rv"]).shift(-1)
        ok = X.notna().all(1) & y.notna()
        self.m = HistGradientBoostingRegressor(random_state=0).fit(X[ok], y[ok])
        resid = y[ok] - self.m.predict(X[ok])
        self.adj = 0.5 * float(resid.var())
        return self

    def predict(self, f):
        X = ml_features(f)
        p = pd.Series(np.nan, index=f.index)
        ok = X.notna().all(1)
        p[ok] = np.exp(self.m.predict(X[ok]) + self.adj)
        return p


class LSTM:
    """One LSTM layer, 16 units, on 22-day windows of the same features as GBM. Target is log rv[t+1]. Seed 0."""
    name = "LSTM"
    SEQ, HID, EPOCHS, LR, BATCH = 22, 16, 15, 1e-3, 64

    def _seq(self, f):
        X = ml_features(f)
        ok = X.notna().all(1).to_numpy()
        Z = ((X - self.mu) / self.sd).to_numpy(dtype=np.float32)
        idx = [i for i in range(self.SEQ - 1, len(Z)) if ok[i - self.SEQ + 1: i + 1].all()]
        return np.stack([Z[i - self.SEQ + 1: i + 1] for i in idx]) if idx else np.empty((0, self.SEQ, Z.shape[1])), idx

    def fit(self, f):
        import torch
        torch.manual_seed(0)
        np.random.seed(0)
        X = ml_features(f)
        self.mu, self.sd = X.mean(), X.std()
        S, idx = self._seq(f)
        y = np.log(f["rv"]).shift(-1).to_numpy()[idx]
        keep = ~np.isnan(y)
        S, y = S[keep], y[keep]
        self.ym, self.ys = float(y.mean()), float(y.std())
        xt = torch.tensor(S)
        yt = torch.tensor(((y - self.ym) / self.ys).astype(np.float32))

        class Net(torch.nn.Module):
            def __init__(s, k, h):
                super().__init__()
                s.l = torch.nn.LSTM(k, h, batch_first=True)
                s.o = torch.nn.Linear(h, 1)

            def forward(s, x):
                return s.o(s.l(x)[0][:, -1]).squeeze(-1)

        self.net = Net(S.shape[2], self.HID)
        opt = torch.optim.Adam(self.net.parameters(), lr=self.LR)
        g = torch.Generator().manual_seed(0)
        for _ in range(self.EPOCHS):
            perm = torch.randperm(len(xt), generator=g)
            for j in range(0, len(xt), self.BATCH):
                b = perm[j: j + self.BATCH]
                opt.zero_grad()
                loss = torch.mean((self.net(xt[b]) - yt[b]) ** 2)
                loss.backward()
                opt.step()
        with torch.no_grad():
            fit = self.net(xt).numpy() * self.ys + self.ym
        self.adj = 0.5 * float(np.var(y - fit))
        return self

    def predict(self, f):
        import torch
        S, idx = self._seq(f)
        p = pd.Series(np.nan, index=f.index)
        if len(idx):
            with torch.no_grad():
                out = self.net(torch.tensor(S)).numpy() * self.ys + self.ym
            p.iloc[idx] = np.exp(out + self.adj)
        return p


MODELS = [HAR, GARCH, GBM, LSTM]
