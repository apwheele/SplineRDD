"""RDD estimators: kernel (rdrobust), regression splines, and tree ensembles.

Every estimator returns a dict with the effect estimate at the cutoff and
whatever interval it supplies, plus out-of-sample (held-out) residuals that
are used to compare the models and to build conformal intervals.

Conventions: x is the running variable, the cutoff is c, d = 1(x >= c).
"""

from __future__ import annotations

import warnings

import numpy as np
from scipy import stats

ALPHA = 0.05


# ---------------------------------------------------------------------------
# Linear algebra helpers
# ---------------------------------------------------------------------------


def ols(X: np.ndarray, y: np.ndarray, w: np.ndarray | None = None):
    """(Weighted) least squares with leverage, LOO residuals and HC3 vcov.

    Returns beta, fitted, resid, leverage, loo_resid, vcov_hc3.
    """
    if w is None:
        sw = np.ones(len(y))
    else:
        sw = np.sqrt(w)
    Xw = X * sw[:, None]
    yw = y * sw
    Q, R = np.linalg.qr(Xw, mode="reduced")
    # drop numerically redundant columns would complicate contrasts; we
    # instead rely on well-conditioned designs and use pinv for safety
    Rinv = np.linalg.pinv(R)
    beta = Rinv @ (Q.T @ yw)
    fitted = X @ beta
    resid = y - fitted
    lev = np.sum(Q**2, axis=1)
    lev = np.clip(lev, 0.0, 1 - 1e-10)
    loo = resid / (1.0 - lev)
    XtX_inv = Rinv @ Rinv.T
    # HC3: (X'WX)^-1 X'W diag(e^2/(1-h)^2) W X (X'WX)^-1
    meat_w = (sw * resid / (1.0 - lev)) ** 2
    meat = (Xw * meat_w[:, None]).T @ Xw
    vcov = XtX_inv @ meat @ XtX_inv
    return beta, fitted, resid, lev, loo, vcov


def conformal_quantile(abs_resid: np.ndarray, alpha: float = ALPHA) -> float:
    """Split-conformal style quantile: the ceil((1-alpha)(n+1))-th smallest."""
    r = np.sort(np.asarray(abs_resid)[np.isfinite(abs_resid)])
    n = len(r)
    k = int(np.ceil((1 - alpha) * (n + 1)))
    if n == 0 or k > n:
        return np.inf
    return float(r[k - 1])


def kfold_index(n: int, k: int, rng: np.random.Generator) -> np.ndarray:
    folds = np.arange(n) % k
    rng.shuffle(folds)
    return folds


def triangular(x: np.ndarray, c: float, h: float) -> np.ndarray:
    return np.clip(1.0 - np.abs(x - c) / h, 0.0, None)


# ---------------------------------------------------------------------------
# Kernel RDD (rdrobust)
# ---------------------------------------------------------------------------


def fit_rdrobust(x, y, c=0.0):
    """rdrobust defaults: local linear, triangular kernel, MSE-optimal common
    bandwidth (mserd), nearest neighbour variance, robust bias-corrected CI.

    Also returns the leave-one-out residuals of the two local lines for the
    observations inside the bandwidth, computed from the weighted hat matrix.
    """
    from rdrobust import rdrobust

    import contextlib
    import io

    with warnings.catch_warnings(), contextlib.redirect_stdout(io.StringIO()):
        warnings.simplefilter("ignore")
        r = rdrobust(y, x, c=c)
    coef = r.coef.iloc[:, 0].to_numpy()
    ci = r.ci.to_numpy()
    h = float(r.bws.iloc[0, 0])
    b = float(r.bws.iloc[1, 0])

    loo = np.full(len(x), np.nan)
    fit_c = {}
    slope = {}
    for side, mask in (("l", x < c), ("r", x >= c)):
        w = triangular(x, c, h) * mask
        use = w > 0
        if use.sum() < 3:
            continue
        X = np.column_stack([np.ones(use.sum()), x[use] - c])
        beta, _, _, _, loo_s, _ = ols(X, y[use], w[use])
        loo[np.where(use)[0]] = loo_s
        fit_c[side] = beta[0]
        slope[side] = beta[1]
    return {
        "est": coef[0],
        "est_bc": coef[1],
        "lo_conv": ci[0, 0],
        "hi_conv": ci[0, 1],
        "lo": ci[2, 0],
        "hi": ci[2, 1],
        "h": h,
        "b": b,
        "loo": loo,
        "line_l": (fit_c.get("l", np.nan), slope.get("l", np.nan)),
    }


# ---------------------------------------------------------------------------
# Regression splines
# ---------------------------------------------------------------------------


def ns_basis(x: np.ndarray, knots) -> np.ndarray:
    """Natural (restricted) cubic spline basis, Harrell's truncated power form.

    knots include the two boundary knots. Returns columns x, C_1..C_{k-2}; the
    function is linear below the first and above the last knot.
    """
    t = np.asarray(sorted(knots), dtype=float)
    k = len(t)
    cols = [x]
    if k >= 3:
        norm = (t[-1] - t[0]) ** 2
        tk, tk1 = t[-1], t[-2]
        for j in range(k - 2):
            cj = (
                np.maximum(x - t[j], 0) ** 3
                - np.maximum(x - tk1, 0) ** 3 * (tk - t[j]) / (tk - tk1)
                + np.maximum(x - tk, 0) ** 3 * (tk1 - t[j]) / (tk - tk1)
            )
            cols.append(cj / norm)
    return np.column_stack(cols)


def grid_knots(lo: float, hi: float, c: float, s: float, offset: float) -> list[float]:
    """Regularly spaced interior knots c + (j + offset) * s inside (lo, hi),
    kept at least s/2 from the boundary knots."""
    jmin = int(np.floor((lo - c) / s)) - 1
    jmax = int(np.ceil((hi - c) / s)) + 1
    ks = [c + (j + offset) * s for j in range(jmin, jmax + 1)]
    return [k for k in ks if lo + s / 2 <= k <= hi - s / 2]


def bs_basis(x: np.ndarray, lo: float, hi: float, interior) -> np.ndarray:
    """Cubic B-spline basis on [lo, hi] (clamped knots), with no constraint at
    either end. Outside [lo, hi] the curve is continued as a straight line
    with the value and slope it has at the end, so a control side fit
    extrapolates past the cutoff linearly rather than as a cubic."""
    from scipy.interpolate import BSpline

    t = np.r_[[lo] * 4, list(interior), [hi] * 4]
    xx = np.asarray(x, dtype=float)

    def dm(z):
        return BSpline.design_matrix(np.asarray(z, dtype=float), t, 3, extrapolate=True).toarray()

    B = dm(np.clip(xx, lo, hi))
    eps = 1e-6 * (hi - lo)
    for end, inside in ((hi, hi - eps), (lo, lo + eps)):
        out = xx > hi if end == hi else xx < lo
        if out.any():
            b_end, b_in = dm([end]), dm([inside])
            slope = (b_end - b_in) / (end - inside)
            B[out] = b_end + (xx[out] - end)[:, None] * slope
    return B


SPLINE_SPECS = {
    # spline_mid: one natural spline through the cutoff, knots on a regular
    #   grid that puts the cutoff halfway between two knots, plus a jump.
    # spline_at: the same grid shifted so that a knot falls on the cutoff.
    # spline_kink: spline_at plus (x - c) and (x - c)^2 on the treated side,
    #   so the slope and curvature can change at the cutoff.
    # spline_sep: a separate cubic B-spline on each side of the cutoff.
    "spline_mid": 0.5,
    "spline_at": 0.0,
    "spline_kink": 0.0,
    "spline_sep": 0.0,
}

# knot spacing as a fraction of the range of x
SPACING_DIVISORS = (2, 3, 4, 5, 6, 8, 10, 12)


def _spline_design(name, c, lo, hi, s):
    """Return design(x, d) for one knot spacing, and the number of knots."""
    if name != "spline_sep":
        interior = grid_knots(lo, hi, c, s, SPLINE_SPECS[name])
        knots = [lo] + interior + [hi]
        kink = name == "spline_kink"

        def design(xx, dd):
            cols = [np.ones(len(xx)), dd, ns_basis(xx, knots)]
            if kink:
                cols += [dd * (xx - c), dd * (xx - c) ** 2]
            return np.column_stack(cols)

        return design, len(interior)

    left = [k for k in grid_knots(lo, c, c, s, 0.0) if k < c]
    right = [k for k in grid_knots(c, hi, c, s, 0.0) if k > c]

    def design(xx, dd):
        bl = bs_basis(xx, lo, c, left) * (1 - dd)[:, None]
        br = bs_basis(xx, c, hi, right) * dd[:, None]
        return np.column_stack([bl, br])

    return design, len(left) + len(right)


def fit_spline(x, y, d, name: str, c=0.0, divisors=SPACING_DIVISORS):
    """OLS spline + jump at the cutoff, knot spacing chosen by exact LOO MSE.

    Returns the estimate, a t interval with HC3 standard errors, the LOO
    residuals, and the counterfactual (d = 0) predictions for every x.
    """
    lo, hi = float(x.min()), float(x.max())
    rng_x = hi - lo
    best = None
    for div in divisors:
        s = rng_x / div
        design, nk = _spline_design(name, c, lo, hi, s)
        X = design(x, d)
        if X.shape[1] >= len(y) - 2 or np.linalg.matrix_rank(X) < X.shape[1]:
            continue
        beta, fitted, resid, lev, loo, vcov = ols(X, y)
        mse = float(np.mean(loo**2))
        if best is None or mse < best["loo_mse"]:
            best = dict(loo_mse=mse, div=div, nk=nk, design=design, beta=beta,
                        loo=loo, vcov=vcov, p=X.shape[1])
    if best is None:
        raise ValueError("no knot spacing gives a full rank design")
    design, beta, vcov = best["design"], best["beta"], best["vcov"]
    one = np.array([c])
    a = design(one, np.ones(1))[0] - design(one, np.zeros(1))[0]
    est = float(a @ beta)
    se = float(np.sqrt(a @ vcov @ a))
    tcrit = stats.t.ppf(1 - ALPHA / 2, len(y) - best["p"])
    cf = design(x, np.zeros(len(x))) @ beta
    return {
        "est": est,
        "lo": est - tcrit * se,
        "hi": est + tcrit * se,
        "se": se,
        "loo": best["loo"],
        "cf": cf,
        "nknots": best["nk"],
        "div": best["div"],
        "s": rng_x / best["div"],
        "predict": lambda xx, dd: design(np.asarray(xx, float), np.asarray(dd, float)) @ beta,
    }


# ---------------------------------------------------------------------------
# Tree ensembles: separate models on each side of the cutoff
# ---------------------------------------------------------------------------

# minimum leaf size as a share of the cases on that side of the cutoff
RF_LEAF_SHARES = (0.01, 0.025, 0.05, 0.1)


def _rf(leaf, n_trees, seed, oob=False):
    from sklearn.ensemble import RandomForestRegressor

    return RandomForestRegressor(n_estimators=n_trees, min_samples_leaf=leaf, max_features=1,
                                 bootstrap=True, oob_score=oob, n_jobs=1, random_state=seed)


def _fit_rf_side(x, y, seed, n_trees):
    """Random forest with the minimum leaf size picked by out-of-bag MSE."""
    X = x[:, None]
    best = None
    leaves = sorted({max(3, int(round(f * len(y)))) for f in RF_LEAF_SHARES})
    for leaf in leaves:
        if leaf * 2 > len(y):
            continue
        rf = _rf(leaf, n_trees, seed, oob=True).fit(X, y)
        oob = rf.oob_prediction_
        ok = np.isfinite(oob)
        mse = float(np.mean((y[ok] - oob[ok]) ** 2))
        if best is None or mse < best[0]:
            best = (mse, rf, y - oob, leaf)
    if best is None:  # tiny side: fall back to the mean
        return (lambda z: np.full(len(z), y.mean())), y - y.mean(), None
    rf = best[1]
    return (lambda z: rf.predict(np.asarray(z)[:, None])), best[2], best[3]


XGB_ROUNDS = (10, 25, 50, 100, 200, 400)
XGB_PARAMS = dict(max_depth=2, learning_rate=0.05, subsample=0.8, min_child_weight=5,
                  reg_lambda=1.0, tree_method="hist", n_jobs=1)


def _xgb(rounds, seed, base):
    import xgboost as xgb

    return xgb.XGBRegressor(n_estimators=rounds, random_state=seed, base_score=base, **XGB_PARAMS)


def _fit_xgb_side(x, y, seed, folds):
    """XGBoost with the number of rounds picked by k-fold CV MSE."""
    X = x[:, None]
    n = len(y)
    if n < 10:
        return (lambda z: np.full(len(z), y.mean())), y - y.mean(), None
    rng = np.random.default_rng(seed)
    fold = kfold_index(n, min(folds, n), rng)
    preds = np.zeros((len(XGB_ROUNDS), n))
    for f in np.unique(fold):
        tr, te = fold != f, fold == f
        m = _xgb(max(XGB_ROUNDS), seed, float(y[tr].mean())).fit(X[tr], y[tr])
        for i, r in enumerate(XGB_ROUNDS):
            preds[i, te] = m.predict(X[te], iteration_range=(0, r))
    mse = ((preds - y[None, :]) ** 2).mean(axis=1)
    i = int(np.argmin(mse))
    m = _xgb(XGB_ROUNDS[i], seed, float(y.mean())).fit(X, y)
    return (lambda z: m.predict(np.asarray(z)[:, None])), y - preds[i], XGB_ROUNDS[i]


def fit_trees(x, y, d, kind: str, seed: int, c=0.0, n_trees=100, folds=5):
    """T-learner: one model per side; effect = f1(c) - f0(c).

    Out-of-sample residuals are out-of-bag (random forest) or 5-fold
    cross-validated (XGBoost). cf is the control model's prediction for every
    x, which is the counterfactual for the treated cases. refit(xtr, ytr, xte)
    refits the control model with its tuned settings (for rolling origins).
    """
    left, right = d == 0, d == 1
    if kind == "rf":
        f0, r0, par = _fit_rf_side(x[left], y[left], seed, n_trees)
        f1, r1, _ = _fit_rf_side(x[right], y[right], seed, n_trees)

        def refit(xtr, ytr, xte):
            if par is None or len(ytr) < 2 * par:
                return np.full(len(xte), ytr.mean())
            return _rf(par, max(50, n_trees // 2), seed).fit(xtr[:, None], ytr).predict(xte[:, None])
    else:
        f0, r0, par = _fit_xgb_side(x[left], y[left], seed, folds)
        f1, r1, _ = _fit_xgb_side(x[right], y[right], seed, folds)

        def refit(xtr, ytr, xte):
            if par is None or len(ytr) < 10:
                return np.full(len(xte), ytr.mean())
            return _xgb(par, seed, float(ytr.mean())).fit(xtr[:, None], ytr).predict(xte[:, None])
    oos = np.empty(len(x))
    oos[left] = r0
    oos[right] = r1
    one = np.array([c])
    est = float(f1(one)[0] - f0(one)[0])
    return {"est": est, "loo": oos, "cf": f0(x), "refit": refit, "par": par}


# ---------------------------------------------------------------------------
# Control-side spline refit (for rolling origins)
# ---------------------------------------------------------------------------


def spline_left_refit(s: float):
    """Cubic B-spline on the control side with knot spacing s, anchored at the
    pseudo-cutoff (the largest training x) and continued beyond it, as
    spline_sep does for the real counterfactual."""

    def refit(xtr, ytr, xte):
        lo, o = float(xtr.min()), float(xtr.max())
        interior = [k for k in grid_knots(lo, o, o, s, 0.0) if k < o]
        X = bs_basis(xtr, lo, o, interior)
        if X.shape[1] >= len(ytr) - 1 or np.linalg.matrix_rank(X) < X.shape[1]:
            b = np.polyfit(xtr, ytr, 1)
            return np.polyval(b, xte)
        beta = np.linalg.lstsq(X, ytr, rcond=None)[0]
        return bs_basis(xte, lo, o, interior) @ beta

    return refit


def rolling_origin(x0, y0, h, refit, c=0.0, k=20, min_frac=0.25, min_n=20):
    """One-sided out-of-sample errors on the control side.

    For k pseudo-cutoffs o between c - h and the point that leaves at least
    max(min_n, min_frac * n0) training cases, fit on controls with x < o and
    predict the controls in [o, o + h), the same extrapolation the real
    counterfactual makes across the cutoff. Returns the block mean errors
    and the pooled individual errors.
    """
    n0 = len(x0)
    xs = np.sort(x0)
    need = max(min_n, int(min_frac * n0))
    if need >= n0:
        return np.array([]), np.array([])
    o_min, o_max = xs[need - 1] + 1e-12, c - h
    if o_max <= o_min:
        return np.array([]), np.array([])
    blocks, ind = [], []
    for o in np.linspace(o_max, o_min, k):
        tr = x0 < o
        te = (x0 >= o) & (x0 < o + h)
        if te.sum() == 0 or tr.sum() < 3:
            continue
        e = y0[te] - refit(x0[tr], y0[tr], x0[te])
        blocks.append(e.mean())
        ind.append(e)
    if not blocks:
        return np.array([]), np.array([])
    return np.array(blocks), np.concatenate(ind)
