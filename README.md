# SplineRDD

A simulation comparing the kernel estimator for regression discontinuity
designs (rdrobust: local linear, triangular kernel, MSE-optimal bandwidth,
robust bias-corrected intervals) to regression splines fit to all of the data
with a dummy variable at the cutoff, plus a random forest and XGBoost with
conformal intervals for the counterfactual.

Four spline models are compared: one natural spline through the cutoff with
knots on a regular grid that either skips or includes the cutoff, the same
with a change in slope and curvature allowed at the cutoff, and separate
splines on each side. The designs are a linear curve, a smooth curve, and the
Lee (2008) and Ludwig-Miller (2007) benchmark designs from Calonico, Cattaneo
and Titiunik (2014), with 100 to 5,000 cases. The paper also checks whether
leave-one-out and k-fold prediction error pick the most accurate model, and
applies the estimators to the blood alcohol thresholds for drunk driving in
Hansen (2015).

The paper is `paper.pdf` (also built as `paper.md`), from `paper.qmd` and
`references.bib`. Working title: "Just use regression (most of the time) for
RDD". `SplineRDD_Convo.txt` is the Claude Code conversation that generated the
code, analysis, and paper.

## Reproducing

This project uses [uv](https://docs.astral.sh/uv/) for the Python environment
and [Quarto](https://quarto.org/) to render the paper. Everything runs on a
CPU; the scripts use two worker processes.

```bash
uv sync
uv run python -m ipykernel install --user --name splinerdd --display-name "Python (splinerdd)"

# kernel and spline models, 2,000 samples per design and size (about 30 minutes)
uv run python scripts/01_sim.py --group fast --reps 2000 --jobs 2
# random forest and XGBoost, 300 samples (about 2 hours)
uv run python scripts/01_sim.py --group trees --reps 300 --jobs 2
# fixed knot spacing check (results/fixed_knots.csv.gz)
uv run python scripts/02_fixed_knots.py --reps 1000 --jobs 2
# combine the simulation chunks into results/sims_fast.csv.gz and sims_trees.csv.gz
uv run python scripts/03_collect.py
# drunk driving application (about 40 minutes)
uv run python scripts/04_hansen.py --reps 300 --jobs 2

quarto render paper.qmd
```

`scripts/run_all.sh` runs every step in order. The simulation scripts are resumable: results are saved in chunks of 100
samples under `results/sims/` (not committed) and finished chunks are skipped.
`03_collect.py` combines them into the files the paper reads. The results the
paper reads are committed, so the paper can be rendered without rerunning
anything. Run the tests with `uv run pytest`; they include checks that
every estimator shifts by exactly tau when tau is added to the treated
outcomes, which is why each sample only needs to be simulated once.

## Layout

- `paper.qmd`, `references.bib` -- paper source and bibliography.
- `SplineRDD_Convo.txt` -- the Claude Code conversation that generated the paper.
- `src/splinerdd/` -- the designs (`dgp.py`), the estimators, out-of-sample
  errors and conformal pieces (`estimators.py`), one simulation replication
  (`sim.py`), the drunk driving data (`hansen.py`), and the summaries used in
  the paper (`report.py`).
- `data/` -- the Hansen (2015) drunk driving data.
- `scripts/` -- the simulation runners.
- `results/` -- the simulation output.
- `filters/` -- pandoc Lua filter for the Markdown build.
- `tests/` -- pytest unit tests.

## Data

`data/hansen_dwi.dta` is the Washington State DUI stop data from Hansen (2015),
"Punishment and Deterrence: Evidence from Drunk Driving," *American Economic
Review* 105(4), in the version distributed with Scott Cunningham's *Causal
Inference: The Mixtape* (<https://github.com/scunning1975/mixtape>).

## License

MIT, see `LICENSE`.
