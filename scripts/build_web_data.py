"""Export results for the web app: web/public/data/summary.json and one forecasts file per ETF (daily variance, %^2)."""
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "web" / "public" / "data"


def main() -> None:
    S = json.loads((ROOT / "results" / "summary.json").read_text())
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "summary.json").write_text(json.dumps(S, separators=(",", ":")))
    fc = pd.read_csv(ROOT / "results" / "forecasts.csv.gz")
    for t, g in fc.groupby("ticker"):
        d = {"date": g["date"].tolist(), **{c: g[c].round(6).tolist() for c in ["rv", *S["models"]]}}
        (OUT / f"{t}.json").write_text(json.dumps(d, separators=(",", ":")))
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
