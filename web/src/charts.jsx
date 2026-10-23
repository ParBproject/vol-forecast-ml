import { useEffect, useRef, useState } from "react";

export function useWidth(initial = 600) {
  const ref = useRef(null);
  const [w, setW] = useState(initial);
  useEffect(() => {
    const ro = new ResizeObserver(([e]) => setW(Math.max(260, Math.round(e.contentRect.width))));
    if (ref.current) ro.observe(ref.current);
    return () => ro.disconnect();
  }, []);
  return [ref, w];
}

// Simple multi-series line/scatter chart in real pixels.
export function XYChart({ series, xfmt, yfmt, height = 260, label, xLabel, zeroLine, vline }) {
  const [ref, W] = useWidth();
  const H = height, L = 52, R = 12, T = 12, B = 34;
  const xs = series.flatMap((s) => s.x), ys = series.flatMap((s) => s.y).filter(Number.isFinite);
  const x0 = Math.min(...xs), x1 = Math.max(...xs);
  let y0 = Math.min(...ys, zeroLine ? 0 : Infinity), y1 = Math.max(...ys, zeroLine ? 0 : -Infinity);
  const pad = (y1 - y0) * 0.06 || 0.01; y0 -= pad; y1 += pad;
  const X = (v) => L + ((v - x0) / (x1 - x0 || 1)) * (W - L - R);
  const Y = (v) => T + (1 - (v - y0) / (y1 - y0)) * (H - T - B);
  const nt = Math.max(3, Math.min(7, Math.floor(W / 90)));
  const xt = Array.from({ length: nt }, (_, i) => x0 + ((x1 - x0) * i) / (nt - 1));
  const yt = Array.from({ length: 5 }, (_, i) => y0 + ((y1 - y0) * i) / 4);
  return (
    <div ref={ref}>
      <svg viewBox={`0 0 ${W} ${H}`} width="100%" role="img" aria-label={label}>
        {yt.map((t) => (
          <g key={t}><line x1={L} x2={W - R} y1={Y(t)} y2={Y(t)} stroke="#1f2630" />
            <text x={L - 6} y={Y(t) + 4} textAnchor="end">{yfmt(t)}</text></g>
        ))}
        {xt.map((t) => <text key={t} x={X(t)} y={H - 18} textAnchor="middle">{xfmt(t)}</text>)}
        {xLabel && <text x={(L + W - R) / 2} y={H - 3} textAnchor="middle">{xLabel}</text>}
        {zeroLine && <line x1={L} x2={W - R} y1={Y(0)} y2={Y(0)} stroke="#9aa7b4" strokeDasharray="4 3" />}
        {vline != null && <line x1={X(vline)} x2={X(vline)} y1={T} y2={H - B} stroke="#9aa7b4" strokeDasharray="3 3" />}
        {series.map((s) => s.dots ? (
          <g key={s.name}>{s.x.map((v, i) => <circle key={i} cx={X(v)} cy={Y(s.y[i])} r={s.r || 2.5} fill={s.color} opacity={0.75} />)}</g>
        ) : (
          <path key={s.name} fill="none" stroke={s.color} strokeWidth={s.width || 2} strokeDasharray={s.dash}
            d={s.x.map((v, i) => (Number.isFinite(s.y[i]) ? `${i && Number.isFinite(s.y[i - 1]) ? "L" : "M"}${X(v).toFixed(1)},${Y(s.y[i]).toFixed(1)}` : "")).join("")} />
        ))}
      </svg>
    </div>
  );
}
