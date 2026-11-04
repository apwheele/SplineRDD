"""Data generating processes for the simulations.

Every design is defined by its no-effect world, mu0(x), the mean outcome
everyone would have without treatment. Treatment is sharp at the cutoff
c = 0: D = 1(x >= 0), and Y = mu0(X) + tau * D + sigma * e, e ~ N(0, 1).

The two benchmark designs (Lee and Ludwig-Miller) are the fifth-order
polynomials from Calonico, Cattaneo and Titiunik (2014), taken from the
authors' replication code (rdpackages-replication on GitHub). Those designs
have a jump at the cutoff; here the jump is subtracted from the right-hand
polynomial so that tau = 0 is the baseline, and tau is then added back as a
constant. The right-hand curve still has a different slope and curvature at
the cutoff than the left-hand curve, so mu0 has a kink at c in those designs.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

CUTOFF = 0.0

LEE_L = (0.48, 1.27, 7.18, 20.21, 21.54, 7.33)
LEE_R = (0.52, 0.84, -3.00, 7.99, -9.01, 3.56)
LM_L = (3.71, 2.30, 3.28, 1.45, 0.23, 0.03)
LM_R = (0.26, 18.49, -54.81, 74.30, -45.02, 9.83)


def _poly(x: np.ndarray, coefs: tuple[float, ...]) -> np.ndarray:
    out = np.zeros_like(x, dtype=float)
    for k, a in enumerate(coefs):
        out = out + a * x**k
    return out


def _x_uniform(n: int, rng: np.random.Generator) -> np.ndarray:
    return rng.uniform(-1.0, 1.0, n)


def _x_beta(n: int, rng: np.random.Generator) -> np.ndarray:
    return 2.0 * rng.beta(2.0, 4.0, n) - 1.0


def _mu_linear(x):
    return x


def _mu_curved(x):
    return np.tanh(3.0 * (x + 0.2))


def _mu_lee(x):
    jump = LEE_R[0] - LEE_L[0]
    return np.where(x < CUTOFF, _poly(x, LEE_L), _poly(x, LEE_R) - jump)


def _mu_lm(x):
    jump = LM_R[0] - LM_L[0]
    return np.where(x < CUTOFF, _poly(x, LM_L), _poly(x, LM_R) - jump)


@dataclass(frozen=True)
class DGP:
    name: str
    label: str
    sigma: float
    draw_x: Callable[[int, np.random.Generator], np.ndarray]
    mu0: Callable[[np.ndarray], np.ndarray]

    def sample(self, n: int, rng: np.random.Generator, tau: float = 0.0):
        """Return x, y, d for one simulated sample."""
        x = self.draw_x(n, rng)
        d = (x >= CUTOFF).astype(float)
        y = self.mu0(x) + tau * d + self.sigma * rng.standard_normal(n)
        return x, y, d


DGPS: dict[str, DGP] = {
    "linear": DGP("linear", "Linear", 1.0, _x_uniform, _mu_linear),
    "curved": DGP("curved", "Smooth curve", 1.0, _x_uniform, _mu_curved),
    "lee": DGP("lee", "Lee (2008)", 0.1295, _x_beta, _mu_lee),
    "lm": DGP("lm", "Ludwig-Miller (2007)", 0.6136, _x_beta, _mu_lm),
}
