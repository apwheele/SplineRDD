#!/usr/bin/env bash
# The full pipeline, in order, with two worker processes (about half of a
# 4 core machine). Takes about 3.5 hours; every step can be rerun on its own.
set -e
cd "$(dirname "$0")/.."
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
uv run python scripts/01_sim.py --group fast --reps 2000 --jobs 2
uv run python scripts/01_sim.py --group trees --reps 300 --jobs 2
uv run python scripts/02_fixed_knots.py --reps 1000 --jobs 2
uv run python scripts/03_collect.py
uv run python scripts/04_hansen.py --reps 300 --jobs 2
quarto render paper.qmd
