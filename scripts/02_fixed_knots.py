"""Sensitivity of the two common spline models to the knot spacing rule.

The main simulation picks the knot spacing by leave-one-out error. Here the
same samples are refit with fixed spacings (1/4 and 1/6 of the range of x)
to see what the data-driven choice does to coverage and bias.

Writes results/fixed_knots.csv.gz (one row per sample, model, and rule).

Usage:
  uv run python scripts/02_fixed_knots.py --reps 1000 --jobs 2
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from joblib import Parallel, delayed  # noqa: E402

from splinerdd import estimators as E  # noqa: E402
from splinerdd.dgp import DGPS  # noqa: E402

RULES = {"loo": E.SPACING_DIVISORS, "fixed_4": (4,), "fixed_6": (6,)}
MODELS = ("spline_mid", "spline_at")
OUT = Path(__file__).resolve().parents[1] / "results" / "fixed_knots.csv.gz"


def run(dgp, n, rep):
    g = DGPS[dgp]
    rng = np.random.default_rng([list(DGPS).index(dgp), n, rep])
    x, y, d = g.sample(n, rng)
    rows = []
    for m in MODELS:
        for rule, divs in RULES.items():
            try:
                r = E.fit_spline(x, y, d, m, divisors=divs)
            except ValueError:
                continue
            rows.append({"dgp": dgp, "n": n, "rep": rep, "model": m, "rule": rule,
                         "sigma": g.sigma, "est": r["est"], "lo": r["lo"], "hi": r["hi"],
                         "nknots": r["nknots"]})
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=1000)
    ap.add_argument("--jobs", type=int, default=2)
    args = ap.parse_args()
    tasks = [(dgp, n, rep) for dgp in DGPS for n in (100, 500, 1000, 5000) for rep in range(args.reps)]
    out = Parallel(n_jobs=args.jobs, batch_size=50)(delayed(run)(*t) for t in tasks)
    df = pd.DataFrame([r for rows in out for r in rows])
    OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT, index=False, compression="gzip")
    print(f"wrote {len(df)} rows to {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
