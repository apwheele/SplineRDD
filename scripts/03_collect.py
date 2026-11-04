"""Combine the per-chunk simulation results into one compressed file per group.

results/sims/<group>/*.csv  ->  results/sims_<group>.csv.gz

The chunk folders are only needed to resume an interrupted run and are not
committed; the paper reads the combined files.

Usage:
  uv run python scripts/03_collect.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

RES = Path(__file__).resolve().parents[1] / "results"


def main():
    for group in ("fast", "trees"):
        files = sorted((RES / "sims" / group).glob("*.csv"))
        if not files:
            print(f"no chunks for {group}")
            continue
        df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
        df = df.sort_values(["dgp", "n", "rep", "model"]).reset_index(drop=True)
        out = RES / f"sims_{group}.csv.gz"
        df.to_csv(out, index=False, float_format="%.8g", compression="gzip")
        print(f"{group}: {len(files)} chunks, {len(df):,} rows -> {out.name} ({out.stat().st_size / 1e6:.1f} MB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
