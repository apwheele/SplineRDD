"""One simulation replication: draw a sample with no effect, fit every model.

All estimators shift exactly by tau when tau * D is added to the outcome
(tests/test_equivariance.py checks this), so each replication is simulated
once at tau = 0 and power for any effect size is computed afterwards by
shifting the intervals.

Out-of-sample error. Linear smoothers (splines, the local lines of the kernel
estimator) use exact leave-one-out residuals. The random forest uses
out-of-bag residuals and XGBoost 5-fold cross-validation. The comparison
metric is the triangular-kernel weighted mean squared out-of-sample error for
the observations within the kernel estimator's bandwidth, so it weights
observations near the cutoff the most.

Conformal intervals (spline_sep, rf, xgb). For a treated case i the estimated individual effect is
e_i = y_i - cf_i, where cf_i is the model's prediction with D = 0. The
interval is e_i +/- q, q the conformal quantile of the absolute out-of-sample
residuals of the control cases. Every case has the same effect in these
designs, so the interval covers if it contains tau. For the average effect
over the m treated cases inside the bandwidth, the interval is mean(e) +/- q_m,
where q_m is the conformal quantile of the absolute means of every run of m
consecutive (in x) control residuals.
"""

from __future__ import annotations

import time

import numpy as np

from .dgp import CUTOFF, DGPS
from .estimators import (
    ALPHA,
    SPLINE_SPECS,
    conformal_quantile,
    fit_rdrobust,
    fit_spline,
    fit_trees,
    rolling_origin,
    spline_left_refit,
    triangular,
)

# effect sizes (in sigma units) at which individual conformal rejection
# rates are stored; model-based power is computed afterwards from lo/hi
IND_TAUS = (0.0, 0.25, 0.5, 1.0, 2.0, 3.0, 4.0)

SPLINES = tuple(SPLINE_SPECS)
TREES = ("rf", "xgb")
MODELS = ("rdrobust",) + SPLINES + TREES
# models whose d = 0 prediction uses only the control cases, so it is a
# genuine counterfactual for the treated cases
CONFORMAL = ("spline_sep", "rf", "xgb")


def _block_q(resid_sorted: np.ndarray, m: int) -> float:
    if m < 1 or m > len(resid_sorted):
        return np.inf
    cs = np.concatenate([[0.0], np.cumsum(resid_sorted)])
    means = (cs[m:] - cs[:-m]) / m
    return conformal_quantile(np.abs(means), ALPHA)


def _conformal(x, y, d, fr, h, sigma, refit, c=CUTOFF):
    """Conformal intervals for the treated cases within h of the cutoff.

    Standard: quantiles of the usual out-of-sample residuals of the controls
    (LOO / out-of-bag / k-fold). Rolling: one-sided rolling-origin errors.
    """
    cf, oos = fr["cf"], fr["loo"]
    left = d == 0
    order = np.argsort(x[left])
    r0 = oos[left][order]
    win = (d == 1) & (x - c <= h)
    m = int(win.sum())
    out = {"m": m}
    q = conformal_quantile(np.abs(r0), ALPHA)
    blocks, ind = rolling_origin(x[left], y[left], h, refit, c)
    q_ro = conformal_quantile(np.abs(ind), ALPHA) if len(ind) else np.inf
    qm_ro = conformal_quantile(np.abs(blocks), ALPHA) if len(blocks) else np.inf
    out["n_origins"] = len(blocks)
    if m == 0:
        return out
    e = y[win] - cf[win]
    eb = float(e.mean())
    qm = _block_q(r0, m)
    out.update({"est_cf": eb,
                "ind_len": 2 * q, "ind_cov": float(np.mean(np.abs(e) <= q)),
                "indro_len": 2 * q_ro, "indro_cov": float(np.mean(np.abs(e) <= q_ro)),
                "avg_lo": eb - qm, "avg_hi": eb + qm,
                "avgro_lo": eb - qm_ro, "avgro_hi": eb + qm_ro})
    for t in IND_TAUS:
        out[f"ind_rej_{t:g}"] = float(np.mean(np.abs(e + t * sigma) > q))
        out[f"indro_rej_{t:g}"] = float(np.mean(np.abs(e + t * sigma) > q_ro))
    return out


def one_rep(dgp_name: str, n: int, rep: int, models=MODELS, tau: float = 0.0) -> list[dict]:
    dgp = DGPS[dgp_name]
    rng = np.random.default_rng([list(DGPS).index(dgp_name), n, rep])
    x, y, d = dgp.sample(n, rng, tau)
    sig = dgp.sigma
    rows = []
    base = {"dgp": dgp_name, "n": n, "rep": rep, "sigma": sig,
            "n_treated": int(d.sum())}

    t0 = time.perf_counter()
    try:
        kr = fit_rdrobust(x, y, CUTOFF)
        h = kr["h"]
    except Exception as exc:  # rdrobust can fail in tiny samples
        kr = None
        h = np.nan
        err = repr(exc)
    secs = time.perf_counter() - t0
    w = triangular(x, CUTOFF, h) if np.isfinite(h) else np.zeros(n)
    sw = w.sum()

    def oos_w(res):
        ok = np.isfinite(res) & (w > 0)
        if sw == 0 or ok.sum() == 0:
            return np.nan
        return float(np.sum(w[ok] * res[ok] ** 2) / np.sum(w[ok]))

    if "rdrobust" in models:
        row = dict(base, model="rdrobust", h=h, seconds=secs)
        if kr is None:
            row["error"] = err
        else:
            row.update({k: kr[k] for k in ("est", "est_bc", "lo", "hi", "lo_conv", "hi_conv", "b")})
            row["oos_w"] = oos_w(kr["loo"])
        rows.append(row)

    for name in models:
        if name == "rdrobust":
            continue
        t0 = time.perf_counter()
        try:
            if name in SPLINES:
                fr = fit_spline(x, y, d, name, CUTOFF)
            else:
                fr = fit_trees(x, y, d, name, seed=rep, c=CUTOFF)
        except Exception as exc:
            rows.append(dict(base, model=name, h=h, error=repr(exc)))
            continue
        secs = time.perf_counter() - t0
        row = dict(base, model=name, h=h, seconds=secs, est=fr["est"],
                   lo=fr.get("lo", np.nan), hi=fr.get("hi", np.nan),
                   nknots=fr.get("nknots", np.nan),
                   oos_w=oos_w(fr["loo"]), oos_all=float(np.mean(fr["loo"] ** 2)))
        if np.isfinite(h) and name in CONFORMAL:
            refit = spline_left_refit(fr["s"]) if name in SPLINES else fr["refit"]
            row.update(_conformal(x, y, d, fr, h, sig, refit))
        rows.append(row)
    return rows
