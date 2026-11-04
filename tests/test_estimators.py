import numpy as np
import pandas as pd
import pytest

from splinerdd import estimators as E
from splinerdd.dgp import DGPS
from splinerdd.sim import MODELS, one_rep


def test_ns_basis_linear_outside_boundary_knots():
    knots = [-1.0, -0.3, 0.2, 0.6, 1.0]
    x = np.linspace(1.0, 3.0, 9)
    B = E.ns_basis(x, knots)
    # second differences vanish beyond the last knot
    assert np.allclose(np.diff(B, 2, axis=0), 0.0, atol=1e-10)
    x = np.linspace(-3.0, -1.0, 9)
    assert np.allclose(np.diff(E.ns_basis(x, knots), 2, axis=0), 0.0, atol=1e-10)


def test_ols_loo_matches_brute_force():
    rng = np.random.default_rng(0)
    X = np.column_stack([np.ones(40), rng.normal(size=40), rng.normal(size=40)])
    y = X @ [1.0, 2.0, -1.0] + rng.normal(size=40)
    w = rng.uniform(0.1, 1.0, 40)
    for weights in (None, w):
        _, _, _, _, loo, _ = E.ols(X, y, weights)
        for i in range(0, 40, 7):
            keep = np.arange(40) != i
            sw = np.ones(39) if weights is None else np.sqrt(weights[keep])
            b = np.linalg.lstsq(X[keep] * sw[:, None], y[keep] * sw, rcond=None)[0]
            assert np.isclose(loo[i], y[i] - X[i] @ b)


def test_local_lines_reproduce_rdrobust_estimate():
    rng = np.random.default_rng(3)
    x, y, d = DGPS["lee"].sample(800, rng, 0.04)
    r = E.fit_rdrobust(x, y)
    h = r["h"]
    ints = []
    for mask in (x < 0, x >= 0):
        w = E.triangular(x, 0.0, h) * mask
        u = w > 0
        X = np.column_stack([np.ones(u.sum()), x[u]])
        ints.append(E.ols(X, y[u], w[u])[0][0])
    assert np.isclose(ints[1] - ints[0], r["est"], atol=1e-8)


@pytest.mark.parametrize("name", list(E.SPLINE_SPECS))
def test_spline_recovers_jump_without_noise(name):
    x = np.linspace(-1, 1, 401)
    d = (x >= 0).astype(float)
    y = 0.5 * x + 0.7 * d
    assert abs(E.fit_spline(x, y, d, name)["est"] - 0.7) < 1e-8
    # a quadratic is only approximated by a natural spline
    y = 0.5 * x + 0.3 * x**2 + 0.7 * d
    assert abs(E.fit_spline(x, y, d, name)["est"] - 0.7) < 1e-3


def test_conformal_quantile():
    r = np.arange(1, 20, dtype=float)  # 19 values: ceil(0.95 * 20) = 19th
    assert E.conformal_quantile(r) == 19.0
    assert np.isinf(E.conformal_quantile(r[:10]))


@pytest.mark.parametrize("dgp", ["linear", "lee"])
@pytest.mark.parametrize("n", [100, 1000])
def test_equivariance_in_tau(dgp, n):
    """Adding tau * D to y shifts estimates and intervals by exactly tau and
    leaves out-of-sample errors unchanged, so power can be computed by
    shifting the tau = 0 results."""
    tau = 0.7 * DGPS[dgp].sigma
    a = pd.DataFrame(one_rep(dgp, n, 11, MODELS, 0.0)).set_index("model")
    b = pd.DataFrame(one_rep(dgp, n, 11, MODELS, tau)).set_index("model")
    tol = 1e-4 * DGPS[dgp].sigma
    for col in ["est", "lo", "hi", "est_bc", "lo_conv", "hi_conv", "est_cf",
                "avg_lo", "avg_hi", "avgro_lo", "avgro_hi"]:
        if col not in a:
            continue
        ok = a[col].notna()
        assert np.allclose(b.loc[ok, col] - a.loc[ok, col], tau, atol=tol), col
    for col in ["h", "oos_w", "oos_all", "ind_len", "indro_len", "nknots"]:
        if col not in a:
            continue
        ok = a[col].notna()
        assert np.allclose(b.loc[ok, col], a.loc[ok, col], atol=tol), col


def test_bs_basis_continues_linearly():
    interior = [-0.6, -0.3]
    x_in = np.linspace(-1, 0, 50)
    B_in = E.bs_basis(x_in, -1.0, 0.0, interior)
    assert np.allclose(B_in.sum(axis=1), 1.0)  # partition of unity inside
    x_out = np.linspace(0.0, 0.5, 9)
    B_out = E.bs_basis(x_out, -1.0, 0.0, interior)
    assert np.allclose(np.diff(B_out, 2, axis=0), 0.0, atol=1e-8)
    assert np.allclose(B_out[0], B_in[-1])
