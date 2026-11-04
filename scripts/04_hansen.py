"""Application: the blood alcohol thresholds for drunk driving (Hansen 2015).

Writes:
  results/hansen_full.csv        every estimator on all of the data, by cutoff
  results/hansen_curves.csv      fitted curves for the figure
  results/hansen_bins.csv        mean recidivism by BAC value
  results/hansen_subsamples.csv  random subsamples at the real and placebo cutoffs

Usage:
  uv run python scripts/04_hansen.py --reps 300 --jobs 2
"""

from __future__ import annotations

import argparse
import os
import sys
import time
import warnings
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from joblib import Parallel, delayed  # noqa: E402
from scipy import stats  # noqa: E402

from splinerdd import estimators as E  # noqa: E402
from splinerdd import hansen as H  # noqa: E402

OUT = Path(__file__).resolve().parents[1] / "results"
SIZES = (5000, 10000, 25000)
PLACEBO_SIZE = 10000
HANSEN_H = 0.05


def fit_hansen(x, y):
    """Hansen's specification: local linear, rectangular kernel, h = 0.05,
    separate slopes on each side, HC3 standard errors."""
    m = np.abs(x) <= HANSEN_H + 1e-9
    d = (x[m] >= 0).astype(float)
    X = np.column_stack([np.ones(m.sum()), d, x[m], d * x[m]])
    b, _, _, _, _, V = E.ols(X, y[m])
    se = np.sqrt(V[1, 1])
    t = stats.t.ppf(0.975, m.sum() - 4)
    return {"est": b[1], "lo": b[1] - t * se, "hi": b[1] + t * se}


def fit_spline_covs(x, y, d, z, s):
    """Spline with a knot at the cutoff, the knot spacing s from the model
    without covariates, plus the covariates."""
    lo, hi = float(x.min()), float(x.max())
    design, _ = E._spline_design("spline_at", 0.0, lo, hi, s)
    X = np.column_stack([design(x, d), z])
    b, _, _, _, _, V = E.ols(X, y)
    se = np.sqrt(V[1, 1])
    t = stats.t.ppf(0.975, len(y) - X.shape[1])
    return {"est": b[1], "lo": b[1] - t * se, "hi": b[1] + t * se}


def fit_all(x, y, d, seed=0, trees=False, z=None):
    """Every estimator on one dataset. Returns rows and the fitted objects."""
    rows, fits = [], {}
    t0 = time.perf_counter()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            kr = E.fit_rdrobust(x, y, 0.0)
        except Exception as exc:  # noqa: BLE001
            kr = None
            rows.append({"model": "rdrobust", "error": repr(exc)})
    h = kr["h"] if kr else np.nan
    w = E.triangular(x, 0.0, h) if kr else np.zeros(len(x))

    def oos_w(res):
        ok = np.isfinite(res) & (w > 0)
        return float(np.sum(w[ok] * res[ok] ** 2) / np.sum(w[ok])) if ok.any() else np.nan

    if kr:
        rows.append({"model": "rdrobust", "est": kr["est"], "lo": kr["lo"], "hi": kr["hi"],
                     "est_bc": kr["est_bc"], "h": h, "oos_w": oos_w(kr["loo"]),
                     "seconds": time.perf_counter() - t0})
        fits["rdrobust"] = kr
    rows.append(dict(model="hansen", **fit_hansen(x, y)))
    for m in E.SPLINE_SPECS:
        t0 = time.perf_counter()
        f = E.fit_spline(x, y, d, m, 0.0)
        fits[m] = f
        rows.append({"model": m, "est": f["est"], "lo": f["lo"], "hi": f["hi"], "h": h,
                     "nknots": f["nknots"], "oos_w": oos_w(f["loo"]),
                     "seconds": time.perf_counter() - t0})
    if z is not None:
        rows.append(dict(model="spline_at_covs", **fit_spline_covs(x, y, d, z, fits["spline_at"]["s"])))
    if trees:
        for m in ("rf", "xgb"):
            t0 = time.perf_counter()
            f = E.fit_trees(x, y, d, m, seed=seed)
            fits[m] = f
            rows.append({"model": m, "est": f["est"], "h": h, "oos_w": oos_w(f["loo"]),
                         "seconds": time.perf_counter() - t0})
    return rows, fits


def full_data(df):
    rows, curves, bins = [], [], []
    for kind, cuts in (("real", H.CUTOFFS), ("placebo", H.PLACEBOS)):
        for c, (lo, hi) in cuts.items():
            x, y, d, z = H.window(df, c, lo, hi)
            print(f"full data, {kind} cutoff {c}: {len(y):,} stops", flush=True)
            r, fits = fit_all(x, y, d, trees=kind == "real", z=z if kind == "real" else None)
            for row in r:
                rows.append(dict(row, cutoff=c, kind=kind, n=len(y), mean_y=y.mean()))
            if kind != "real":
                continue
            b = pd.DataFrame({"x": x, "y": y}).groupby("x")["y"].agg(["mean", "size"]).reset_index()
            bins.append(b.assign(cutoff=c))
            grid = np.round(np.arange(lo, hi + 1e-9, 0.0005) - c, 6)
            gd = (grid >= 0).astype(float)
            cur = pd.DataFrame({"cutoff": c, "x": grid, "d": gd})
            for m in E.SPLINE_SPECS:
                cur[m] = fits[m]["predict"](grid, gd)
            kr = fits["rdrobust"]
            hh = kr["h"]
            (a_l, b_l) = kr["line_l"]
            cur["rdrobust"] = np.nan
            left = (grid < 0) & (grid >= -hh)
            cur.loc[left, "rdrobust"] = a_l + b_l * grid[left]
            right = (grid >= 0) & (grid <= hh)
            wr = E.triangular(x, 0.0, hh) * (d == 1)
            u = wr > 0
            br = E.ols(np.column_stack([np.ones(u.sum()), x[u]]), y[u], wr[u])[0]
            cur.loc[right, "rdrobust"] = br[0] + br[1] * grid[right]
            curves.append(cur)
    pd.DataFrame(rows).to_csv(OUT / "hansen_full.csv", index=False)
    pd.concat(curves).to_csv(OUT / "hansen_curves.csv", index=False)
    pd.concat(bins).to_csv(OUT / "hansen_bins.csv", index=False)


def one_sub(df_idx, xs, ys, ds, n, rep, c, kind):
    rng = np.random.default_rng([int(c * 1000), n, rep])
    i = rng.choice(len(ys), size=n, replace=False)
    rows, _ = fit_all(xs[i], ys[i], ds[i])
    return [dict(r, cutoff=c, kind=kind, n=n, rep=rep) for r in rows]


def subsamples(df, reps, jobs):
    tasks = []
    for kind, cuts, sizes in (("real", H.CUTOFFS, SIZES), ("placebo", H.PLACEBOS, (PLACEBO_SIZE,))):
        for c, (lo, hi) in cuts.items():
            x, y, d, _ = H.window(df, c, lo, hi)
            for n in sizes:
                for rep in range(reps):
                    tasks.append((x, y, d, n, rep, c, kind))
    print(f"{len(tasks)} subsamples", flush=True)
    t0 = time.time()
    out = Parallel(n_jobs=jobs, batch_size=10, verbose=0)(
        delayed(one_sub)(None, *t) for t in tasks)
    print(f"subsamples done in {(time.time() - t0) / 60:.1f} min", flush=True)
    pd.DataFrame([r for rows in out for r in rows]).to_csv(OUT / "hansen_subsamples.csv", index=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=300)
    ap.add_argument("--jobs", type=int, default=2)
    ap.add_argument("--skip-full", action="store_true")
    args = ap.parse_args()
    df = H.load()
    if not args.skip_full:
        full_data(df)
    if args.reps > 0:
        subsamples(df, args.reps, args.jobs)
    return 0


if __name__ == "__main__":
    sys.exit(main())
