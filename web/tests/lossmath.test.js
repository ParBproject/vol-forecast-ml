import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { test } from "node:test";
import { losses, qlike, rollingRatio } from "../src/lossmath.js";

const S = JSON.parse(readFileSync(new URL("../public/data/summary.json", import.meta.url)));

test("qlike basics", () => {
  assert.equal(qlike(2, 2), 0);
  assert.ok(qlike(1, 2) > 0 && qlike(4, 2) > 0);
});

test("browser losses reproduce the Python results for every ETF and model", () => {
  for (const t of S.tickers) {
    const d = JSON.parse(readFileSync(new URL(`../public/data/${t}.json`, import.meta.url)));
    assert.equal(d.rv.length, S.per_ticker[t].n_oos);
    for (const m of S.models) {
      const l = losses(d, m);
      const p = S.per_ticker[t].models[m];
      assert.ok(Math.abs(l.qlike / p.qlike - 1) < 1e-3, `${t} ${m} qlike ${l.qlike} vs ${p.qlike}`);
      assert.ok(Math.abs(l.mse / p.mse - 1) < 1e-3, `${t} ${m} mse`);
    }
    const r = rollingRatio(d, "HAR");
    assert.ok(Math.abs(r[r.length - 1] - 1) < 1e-12);
  }
});
