"""Run the simulation grid. Resumable: results are written in chunks of
replications to results/sims/<group>/<dgp>_<n>_<chunk>.csv and existing
chunks are skipped.

Groups:
  fast   rdrobust and the four spline models (cheap)
  trees  random forest and XGBoost (the slow part)

Usage:
  uv run python scripts/01_sim.py --group fast --reps 2000 --jobs 2
  uv run python scripts/01_sim.py --group trees --reps 500 --jobs 2
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import pandas as pd  # noqa: E402
from joblib import Parallel, delayed  # noqa: E402

from splinerdd.dgp import DGPS  # noqa: E402
from splinerdd.sim import SPLINES, TREES, one_rep  # noqa: E402

GROUPS = {"fast": ("rdrobust",) + SPLINES, "trees": TREES}
SIZES = (100, 500, 1000, 5000)
CHUNK = 100
OUT = Path(__file__).resolve().parents[1] / "results" / "sims"


def run_chunk(group, dgp, n, chunk):
    rows = []
    for rep in range(chunk * CHUNK, (chunk + 1) * CHUNK):
        rows.extend(one_rep(dgp, n, rep, GROUPS[group]))
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--group", choices=list(GROUPS), required=True)
    ap.add_argument("--reps", type=int, default=1000)
    ap.add_argument("--jobs", type=int, default=2)
    ap.add_argument("--dgps", default=",".join(DGPS))
    ap.add_argument("--sizes", default=",".join(map(str, SIZES)))
    args = ap.parse_args()

    out = OUT / args.group
    out.mkdir(parents=True, exist_ok=True)
    tasks = []
    for n in map(int, args.sizes.split(",")):
        for dgp in args.dgps.split(","):
            for chunk in range(args.reps // CHUNK):
                f = out / f"{dgp}_{n}_{chunk:03d}.csv"
                if not f.exists():
                    tasks.append((dgp, n, chunk, f))
    print(f"{len(tasks)} chunks of {CHUNK} to run", flush=True)
    t0 = time.time()
    done = 0
    gen = Parallel(n_jobs=args.jobs, return_as="generator_unordered")(
        delayed(lambda t: (t, run_chunk(args.group, *t[:3])))(t) for t in tasks
    )
    for (dgp, n, chunk, f), df in gen:
        df.to_csv(f, index=False)
        done += 1
        el = time.time() - t0
        eta = el / done * (len(tasks) - done)
        print(f"[{time.strftime('%H:%M:%S')}] {done}/{len(tasks)} {dgp} n={n} chunk {chunk} "
              f"elapsed {el / 60:.1f} min, eta {eta / 60:.1f} min", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
