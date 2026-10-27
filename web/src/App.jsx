import { useEffect, useMemo, useState } from "react";
import { XYChart } from "./charts.jsx";
import { annVol, bucket, losses, rollingRatio } from "./lossmath.js";

const BASE = import.meta.env.BASE_URL;
const COLORS = { rv: "#5b6673", HAR: "#10b981", GARCH: "#fbbf24", GBM: "#60a5fa", LSTM: "#f472b6" };
const LABEL = { HAR: "HAR-RV", GARCH: "GARCH(1,1)", GBM: "Gradient boosting", LSTM: "LSTM" };
const yr = (ms) => new Date(ms).getUTCFullYear();
const fmtDate = (ms) => new Date(ms).toISOString().slice(0, 7);
const pct = (x, d = 1) => `${x > 0 ? "+" : ""}${(x * 100).toFixed(d)}%`;

function topMseDates(S) {
  const counts = new Map();
  let march = 0;
  for (const t of S.tickers) {
    const dates = S.per_ticker[t].har_mse_top5_days.dates;
    if (dates.some((d) => d.startsWith("2020-03"))) march += 1;
    for (const d of dates) {
      if (!d.startsWith("2020-03")) counts.set(d, (counts.get(d) || 0) + 1);
    }
  }
  const parts = [];
  if (march) parts.push({ n: march, label: `March 2020 (${march} of ${S.tickers.length})` });
  for (const [d, n] of counts) {
    if (n >= 5) parts.push({ n, label: `${d} (${n} of ${S.tickers.length})` });
  }
  parts.sort((a, b) => b.n - a.n || a.label.localeCompare(b.label));
  return parts.map((p) => p.label).join(", ");
}

export default function App() {
  const [S, setS] = useState(null);
  const [tk, setTk] = useState("SPY");
  const [data, setData] = useState({});
  const [on, setOn] = useState({ HAR: true, GARCH: false, GBM: false, LSTM: true });
  const [period, setPeriod] = useState("2020");

  useEffect(() => { fetch(`${BASE}data/summary.json`).then((r) => r.json()).then(setS); }, []);
  useEffect(() => {
    if (data[tk]) return;
    fetch(`${BASE}data/${tk}.json`).then((r) => r.json()).then((d) => {
      d.t = d.date.map((s) => Date.parse(s));
      setData((o) => ({ ...o, [tk]: d }));
    });
  }, [tk, data]);

  const d = data[tk];
  const years = useMemo(() => (d ? [...new Set(d.t.map(yr))] : []), [d]);
  const [lo, hi] = useMemo(() => {
    if (!d) return [0, 0];
    if (period === "all") return [0, d.t.length];
    const y = Number(period), idx = d.t.map(yr);
    return [idx.indexOf(y), idx.lastIndexOf(y) + 1];
  }, [d, period]);

  if (!S) return <div className="wrap"><p className="note">Loading results…</p></div>;
  const A = S.aggregate, P = S.per_ticker, M = S.models;
  const rel = (k, l) => A[k][`mean_${l}`] / A.HAR[`mean_${l}`] - 1;
  const shown = M.filter((m) => on[m]);
  const mseSig = M.reduce((s, m) => s + (A[m].sig_better_mse || 0) + (A[m].sig_worse_mse || 0), 0);

  let vol = null, roll = null, win = null;
  if (d && hi > lo) {
    const x = d.t.slice(lo, hi);
    const ys = [d.rv, ...shown.map((m) => d[m])].map((s) => s.slice(lo, hi).map(annVol));
    const [bx, by] = bucket(x, ys, period === "all" ? 700 : 260);
    vol = [{ name: "rv", x: bx, y: by[0], color: COLORS.rv, width: 1.2 },
      ...shown.map((m, i) => ({ name: m, x: bx, y: by[i + 1], color: COLORS[m], width: 1.8 }))];
    const rm = M.filter((m) => m !== "HAR" && on[m]).map((m) => ({ m, r: rollingRatio(d, m) }));
    const [rx, ry] = bucket(d.t, rm.map((o) => o.r), 500);
    roll = rm.map((o, i) => ({ name: o.m, x: rx, y: ry[i], color: COLORS[o.m] }));
    win = Object.fromEntries(M.map((m) => [m, losses(d, m, lo, hi)]));
  }
  const first = P[S.tickers[0]];

  return (
    <div className="wrap">
      <header>
        <div className="eyebrow">Walk-forward · {S.tickers.length} ETFs · {first.oos_first} to {first.oos_last}</div>
        <h1>One-day-ahead volatility forecasts</h1>
        <p className="lede">Next-day Garman-Klass variance for SPY and nine sector ETFs, using only data through the previous close.
          Each model is refit once a year. The LSTM has the lowest QLIKE on {A.LSTM.wins_qlike} of {S.tickers.length} ETFs
          ({A.LSTM.sig_better_qlike} significantly better than HAR at 5%). {mseSig === 0
            ? "No model's MSE difference versus HAR is significant at 5%."
            : `${mseSig} MSE comparisons versus HAR are significant at 5%.`}</p>
      </header>

      <div className="kpis" style={{ marginBottom: 16 }}>
        <div className="kpi"><div className="v pos">{pct(rel("LSTM", "qlike"))}</div><div className="l">LSTM mean QLIKE vs HAR</div></div>
        <div className="kpi"><div className="v">{A.LSTM.sig_better_qlike}/{S.tickers.length}</div><div className="l">LSTM sig. better (DM, QLIKE, 5%)</div></div>
        <div className="kpi"><div className="v">{pct(rel("GBM", "qlike"))}</div><div className="l">Gradient boosting QLIKE vs HAR</div></div>
        <div className="kpi"><div className="v neg">{pct(rel("GARCH", "qlike"))}</div><div className="l">GARCH QLIKE vs HAR</div></div>
        <div className="kpi"><div className="v">{M.reduce((s, m) => s + (A[m].sig_better_mse || 0), 0)}</div><div className="l">Sig. MSE wins vs HAR (any model)</div></div>
        <div className="kpi"><div className="v">{first.n_oos.toLocaleString()}</div><div className="l">Out-of-sample days per ETF</div></div>
      </div>

      <div className="grid">
        <section className="card span-12" aria-label="Controls">
          <div className="controls">
            <div className="pills" role="group" aria-label="ETF">
              {S.tickers.map((t) => <button key={t} className="pill" aria-pressed={t === tk} onClick={() => setTk(t)}>{t}</button>)}
            </div>
            <div className="pills" role="group" aria-label="Models">
              {M.map((m) => (
                <button key={m} className="pill" aria-pressed={on[m]} onClick={() => setOn({ ...on, [m]: !on[m] })}
                  style={on[m] ? { background: COLORS[m], borderColor: COLORS[m] } : { borderColor: COLORS[m] }}>{LABEL[m]}</button>
              ))}
            </div>
            <label className="field">Period
              <select value={period} onChange={(e) => setPeriod(e.target.value)}>
                <option value="all">All out-of-sample</option>
                {years.map((y) => <option key={y} value={y}>{y}</option>)}
              </select>
            </label>
          </div>
        </section>

        <section className="card span-8">
          <h2>{tk}: realized vs forecast volatility (annualized %)</h2>
          {vol ? <XYChart series={vol} xfmt={fmtDate} yfmt={(v) => v.toFixed(0)} height={320} label="Realized and forecast volatility" /> : <p className="note">Loading…</p>}
          <div className="legend"><span><i style={{ background: COLORS.rv }} />Realized (GK)</span>
            {shown.map((m) => <span key={m}><i style={{ background: COLORS[m] }} />{LABEL[m]}</span>)}</div>
          <p className="note">{period === "all" ? "Averaged into buckets for drawing." : "Daily values."} Losses use every day.</p>
        </section>

        <section className="card span-4">
          <h2>Losses, {period === "all" ? "full sample" : period}</h2>
          {win && <div className="scroll"><table>
            <thead><tr><th>Model</th><th>QLIKE</th><th>MSE</th></tr></thead>
            <tbody>{M.map((m) => {
              const bq = Math.min(...M.map((k) => win[k].qlike)), bm = Math.min(...M.map((k) => win[k].mse));
              return <tr key={m}><td style={{ color: COLORS[m] }}>{LABEL[m]}</td>
                <td className={win[m].qlike === bq ? "pos" : ""}>{win[m].qlike.toFixed(4)}</td>
                <td className={win[m].mse === bm ? "pos" : ""}>{win[m].mse.toFixed(3)}</td></tr>;
            })}</tbody></table></div>}
          <p className="note">{win ? win.HAR.n : ""} days. Best in green. Full-sample Diebold-Mariano vs HAR (negative = beats HAR):</p>
          <div className="scroll"><table>
            <thead><tr><th>Model</th><th>DM QLIKE</th><th>p</th><th>DM MSE</th><th>p</th></tr></thead>
            <tbody>{M.slice(1).map((m) => {
              const q = P[tk].models[m].dm_qlike_vs_har, s = P[tk].models[m].dm_mse_vs_har;
              return <tr key={m}><td style={{ color: COLORS[m] }}>{m}</td>
                <td className={q.p < 0.05 ? (q.stat < 0 ? "pos" : "neg") : ""}>{q.stat.toFixed(2)}</td><td>{q.p.toFixed(3)}</td>
                <td className={s.p < 0.05 ? (s.stat < 0 ? "pos" : "neg") : ""}>{s.stat.toFixed(2)}</td><td>{s.p.toFixed(3)}</td></tr>;
            })}</tbody></table></div>
        </section>

        <section className="card span-6">
          <h2>{tk}: rolling 1-year QLIKE ratio vs HAR</h2>
          {roll && roll.length ? <XYChart series={roll} xfmt={fmtDate} yfmt={(v) => v.toFixed(2)} height={240} label="Rolling QLIKE ratio" />
            : <p className="note">Turn on GARCH, gradient boosting or LSTM to compare with HAR.</p>}
          <p className="note">Below 1.00 means the model beat HAR over the trailing 252 days.</p>
        </section>

        <section className="card span-6">
          <h2>All ETFs: QLIKE vs HAR (full sample)</h2>
          <div className="scroll"><table>
            <thead><tr><th>ETF</th>{M.slice(1).map((m) => <th key={m}>{m}</th>)}<th>Best</th></tr></thead>
            <tbody>{S.tickers.map((t) => (
              <tr key={t} className={t === tk ? "sel" : ""} onClick={() => setTk(t)}>
                <td>{t}</td>
                {M.slice(1).map((m) => {
                  const r = P[t].models[m].qlike / P[t].models.HAR.qlike - 1, dm = P[t].models[m].dm_qlike_vs_har;
                  return <td key={m} className={dm.p < 0.05 ? (dm.stat < 0 ? "pos" : "neg") : ""}>{pct(r)}{dm.p < 0.05 ? "*" : ""}</td>;
                })}
                <td>{P[t].best_qlike}</td>
              </tr>))}</tbody></table></div>
          <p className="note">* = Diebold-Mariano p &lt; 0.05. Click a row to load that ETF.</p>
        </section>

        <section className="card span-12">
          <h2>Method</h2>
          <p className="note">Yahoo daily OHLC from 2005 through {first.oos_last}, stored under data/raw with SHA-256 hashes.
            The target is next-day Garman-Klass variance from the daily bar (open to close), not intraday realized variance.
            The first training sample is 2005-2012; each later year is fit only on earlier rows. HAR is OLS in levels.
            GARCH(1,1) uses open-to-close returns. Gradient boosting is scikit-learn's histogram booster on log variance,
            with the library defaults. The LSTM has 16 units, a 22-day window, 15 epochs, and seed 0.
            The largest HAR squared errors fall on {topMseDates(S)}.</p>
        </section>
      </div>
      <footer>Par Bahrae · <a href="https://github.com/ParBproject/vol-forecast-ml">Source and full results on GitHub</a> ·{" "}
        <a href="https://parbproject.github.io/">Portfolio</a> · <a href="https://www.linkedin.com/in/parbahrae/">LinkedIn</a></footer>
    </div>
  );
}
