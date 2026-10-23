// Loss math mirrored from src/volml/evaluate.py; tested against results/summary.json.
export const qlike = (rv, f) => rv / f - Math.log(rv / f) - 1;
export const sqerr = (rv, f) => (rv - f) ** 2;
export const mean = (a) => a.reduce((s, v) => s + v, 0) / a.length;
export const annVol = (v) => Math.sqrt(252 * v); // daily variance %^2 -> annualized vol %

export function losses(d, model, lo = 0, hi = d.rv.length) {
  const q = [], m = [];
  for (let i = lo; i < hi; i++) { q.push(qlike(d.rv[i], d[model][i])); m.push(sqerr(d.rv[i], d[model][i])); }
  return { qlike: mean(q), mse: mean(m), n: hi - lo };
}

// Rolling mean QLIKE ratio model/HAR over `win` days (values < 1 mean the model beats HAR).
export function rollingRatio(d, model, win = 252) {
  const a = d.rv.map((r, i) => qlike(r, d[model][i])), b = d.rv.map((r, i) => qlike(r, d.HAR[i]));
  const out = [];
  let sa = 0, sb = 0;
  for (let i = 0; i < a.length; i++) {
    sa += a[i]; sb += b[i];
    if (i >= win) { sa -= a[i - win]; sb -= b[i - win]; }
    out.push(i >= win - 1 ? sa / sb : NaN);
  }
  return out;
}

// Downsample a series to <= n points by averaging buckets (for drawing only).
export function bucket(x, ys, n) {
  const k = Math.max(1, Math.ceil(x.length / n)), X = [], Y = ys.map(() => []);
  for (let i = 0; i < x.length; i += k) {
    X.push(x[i]);
    ys.forEach((y, j) => { const s = y.slice(i, i + k).filter(Number.isFinite); Y[j].push(s.length ? mean(s) : NaN); });
  }
  return [X, Y];
}
