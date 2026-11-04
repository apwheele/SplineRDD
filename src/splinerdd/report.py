"""Summaries of the simulation results for the paper.

All error metrics are in units of the noise standard deviation sigma.
Every replication is simulated with no effect; the estimators are exactly
shift-equivariant in tau (see tests), so the interval for an effect tau is
[lo + tau, hi + tau] and power is the share of those intervals that exclude
zero. Power is averaged over tau and -tau so a biased estimator does not get
credit for its bias.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
SIMS = ROOT / "results" / "sims"

LABELS = {
    "rdrobust": "Kernel (rdrobust)",
    "spline_mid": "Spline, cutoff between knots",
    "spline_at": "Spline, knot at cutoff",
    "spline_kink": "Spline + slope/curvature change",
    "spline_sep": "Separate splines each side",
    "rf": "Random forest",
    "xgb": "XGBoost",
}
SHORT = {
    "rdrobust": "Kernel",
    "spline_mid": "Spline (between)",
    "spline_at": "Spline (at)",
    "spline_kink": "Spline + kink",
    "spline_sep": "Separate splines",
    "rf": "Random forest",
    "xgb": "XGBoost",
}
DGP_LABELS = {"linear": "Linear", "curved": "Smooth curve", "lee": "Lee", "lm": "Ludwig-Miller"}
DGP_ORDER = ["linear", "curved", "lee", "lm"]
CI_MODELS = ["rdrobust", "spline_mid", "spline_at", "spline_kink", "spline_sep"]
ALL_MODELS = CI_MODELS + ["rf", "xgb"]


@lru_cache(maxsize=4)
def load(group: str) -> pd.DataFrame:
    """Results for one group, from results/sims_<group>.csv.gz (written by
    scripts/03_collect.py) or, if that is missing, from the chunk files."""
    combined = ROOT / "results" / f"sims_{group}.csv.gz"
    if combined.exists():
        df = pd.read_csv(combined)
    else:
        files = sorted((SIMS / group).glob("*.csv"))
        df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    s = df["sigma"]
    df["err"] = df["est"] / s
    df["cover"] = (df["lo"] <= 0) & (df["hi"] >= 0)
    df["length"] = (df["hi"] - df["lo"]) / s
    if "lo_conv" in df:
        df["cover_conv"] = (df["lo_conv"] <= 0) & (df["hi_conv"] >= 0)
        df["length_conv"] = (df["hi_conv"] - df["lo_conv"]) / s
    df["oos"] = df["oos_w"] / s**2
    return df


def sims() -> pd.DataFrame:
    """Kernel and spline results (all replications) plus the tree models."""
    fast = load("fast")
    try:
        trees = load("trees")
    except (ValueError, FileNotFoundError):
        return fast
    return pd.concat([fast, trees], ignore_index=True)


def power(lo: np.ndarray, hi: np.ndarray, tau: float) -> float:
    """Share of intervals [lo + t, hi + t] excluding 0, averaged over t = +/-tau."""
    lo, hi = np.asarray(lo), np.asarray(hi)
    ok = np.isfinite(lo) & np.isfinite(hi)
    lo, hi = lo[ok], hi[ok]
    up = np.mean((lo + tau > 0) | (hi + tau < 0))
    dn = np.mean((lo - tau > 0) | (hi - tau < 0))
    return 0.5 * (up + dn)


def power_curve(lo, hi, taus) -> np.ndarray:
    return np.array([power(lo, hi, t) for t in taus])


def mde(lo, hi, target=0.8, grid=None) -> float:
    """Smallest effect (in the units of lo/hi) with average power >= target."""
    if grid is None:
        grid = np.linspace(0, 6, 1201)
    pw = power_curve(lo, hi, grid)
    hit = np.nonzero(pw >= target)[0]
    return float(grid[hit[0]]) if len(hit) else np.nan


def summary(df: pd.DataFrame | None = None) -> pd.DataFrame:
    if df is None:
        df = sims()
    rows = []
    for (dgp, n, model), g in df.groupby(["dgp", "n", "model"]):
        e = g["err"].dropna()
        lo, hi = g["lo"] / g["sigma"], g["hi"] / g["sigma"]
        has_ci = lo.notna().any()
        rows.append({
            "dgp": dgp, "n": n, "model": model, "reps": len(g),
            "failed": g["est"].isna().mean(),
            "bias": e.mean(), "sd": e.std(), "rmse": np.sqrt(np.mean(e**2)),
            "mae": np.median(np.abs(e)),
            "coverage": g.loc[g["est"].notna(), "cover"].mean() if has_ci else np.nan,
            "length": g["length"].median() if has_ci else np.nan,
            "mde80": mde(lo, hi) if has_ci else np.nan,
            "size": power(lo, hi, 0.0) if has_ci else np.nan,
            "oos": g["oos"].mean(),
            "knots": g["nknots"].mean() if "nknots" in g else np.nan,
            "h": g["h"].median(),
            "seconds": g["seconds"].mean(),
        })
    return pd.DataFrame(rows)


def selection(df: pd.DataFrame, models, by="oos") -> pd.DataFrame:
    """Per replication, pick the model with the lowest out-of-sample error.

    Returns one row per (dgp, n, rep) with the chosen model, its error, and
    the error of the model that turned out best in that replication.
    """
    d = df[df["model"].isin(models)].dropna(subset=["err", by])
    keys = ["dgp", "n", "rep"]
    # keep replications where every candidate is present
    cnt = d.groupby(keys)["model"].transform("nunique")
    d = d[cnt == len(models)]
    pick = d.loc[d.groupby(keys)[by].idxmin()]
    best = d.assign(abs_err=d["err"].abs()).sort_values("abs_err").drop_duplicates(keys)
    out = pick[keys + ["model", "err", "cover", "lo", "hi", "sigma"]].rename(
        columns={"model": "chosen"}).merge(
        best[keys + ["model"]].rename(columns={"model": "best"}), on=keys)
    return out


def selection_summary(df: pd.DataFrame, models, by="oos") -> pd.DataFrame:
    sel = selection(df, models, by)
    rows = []
    for (dgp, n), g in sel.groupby(["dgp", "n"]):
        row = {"dgp": dgp, "n": n, "reps": len(g),
               "rmse": np.sqrt(np.mean(g["err"] ** 2)),
               "coverage": g["cover"].mean(),
               "agree": np.mean(g["chosen"] == g["best"])}
        for m in models:
            row[f"share_{m}"] = np.mean(g["chosen"] == m)
        rows.append(row)
    return pd.DataFrame(rows)


def conformal_summary(df: pd.DataFrame | None = None, max_rep: int | None = None) -> pd.DataFrame:
    """Conformal intervals for the treated cases within the bandwidth.

    max_rep restricts every model to the same samples (rep < max_rep), e.g.
    the samples the tree models were run on.
    """
    if df is None:
        df = sims()
    d = df[df["model"].isin(["spline_sep", "rf", "xgb"])].dropna(subset=["est_cf"])
    if max_rep is not None:
        d = d[d["rep"] < max_rep]
    rows = []
    for (dgp, n, model), g in d.groupby(["dgp", "n", "model"]):
        s = g["sigma"]
        row = {"dgp": dgp, "n": n, "model": model, "reps": len(g),
               "m": g["m"].median(),
               "bias_cf": (g["est_cf"] / s).mean(),
               "ind_cov": g["ind_cov"].mean(), "ind_len": (g["ind_len"] / s).median(),
               "indro_cov": g["indro_cov"].mean(), "indro_len": (g["indro_len"] / s).median()}
        for col in [c for c in g.columns if c.startswith("ind_rej_") or c.startswith("indro_rej_")]:
            row[col] = g[col].mean()
        for kind in ("avg", "avgro"):
            lo, hi = g[f"{kind}_lo"] / s, g[f"{kind}_hi"] / s
            row[f"{kind}_cov"] = np.mean((lo <= 0) & (hi >= 0))
            row[f"{kind}_len"] = (hi - lo).median()
            row[f"{kind}_mde80"] = mde(lo, hi)
        rows.append(row)
    return pd.DataFrame(rows)


def fixed_knots_summary() -> pd.DataFrame:
    """Common splines with knot spacing chosen by leave-one-out error versus
    fixed at 1/4 or 1/6 of the range of x (scripts/02_fixed_knots.py)."""
    d = pd.read_csv(ROOT / "results" / "fixed_knots.csv.gz")
    s = d["sigma"]
    d["err"] = d["est"] / s
    d["lo_s"], d["hi_s"] = d["lo"] / s, d["hi"] / s
    rows = []
    for (dgp, n, model, rule), g in d.groupby(["dgp", "n", "model", "rule"]):
        rows.append({"dgp": dgp, "n": n, "model": model, "rule": rule, "reps": len(g),
                     "bias": g["err"].mean(), "rmse": np.sqrt(np.mean(g["err"] ** 2)),
                     "coverage": np.mean((g["lo_s"] <= 0) & (g["hi_s"] >= 0)),
                     "length": (g["hi_s"] - g["lo_s"]).median(),
                     "size": power(g["lo_s"], g["hi_s"], 0.0),
                     "mde80": mde(g["lo_s"], g["hi_s"]),
                     "knots": g["nknots"].mean()})
    return pd.DataFrame(rows)
