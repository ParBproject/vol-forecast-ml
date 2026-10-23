import json
import subprocess
import sys
from pathlib import Path

import numpy as np

from volml.data import TICKERS, frame, garman_klass, load

ROOT = Path(__file__).resolve().parents[1]


def test_manifest_matches():
    r = subprocess.run([sys.executable, str(ROOT / "scripts" / "manifest.py"), "--check"], capture_output=True, text=True)
    assert r.returncode == 0, r.stdout + r.stderr
    assert len(json.loads((ROOT / "data" / "MANIFEST.json").read_text())) == len(TICKERS) + 1


def test_panel_shape():
    idx = None
    for t in TICKERS:
        df = load(t)
        assert len(df) == 5486 and str(df.index[0].date()) == "2005-01-03" and str(df.index[-1].date()) == "2026-10-22"
        assert (df["High"] >= df[["Open", "Close", "Low"]].max(1) - 1e-6).all()
        idx = df.index if idx is None else idx
        assert df.index.equals(idx)


def test_gk_nonnegative_and_finite():
    for t in TICKERS:
        assert (garman_klass(load(t)) >= -1e-12).all()
        f = frame(t)
        assert np.isfinite(f.to_numpy()).all() and (f["rv"] > 0).all()
