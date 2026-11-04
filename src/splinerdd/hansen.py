"""The Hansen (2015) drunk driving data.

data/hansen_dwi.dta is the version of the Washington State DUI stop data
distributed with Cunningham's Causal Inference: The Mixtape
(https://github.com/scunning1975/mixtape, file hansen_dwi.dta). The running
variable is the lower of the two breath tests, as in Hansen (2015), since the
lower test determines guilt. The outcome is any new DUI stop within four years.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data" / "hansen_dwi.dta"
COVARIATES = ["male", "white", "aged", "acc"]

# real thresholds: DUI at 0.08, aggravated DUI at 0.15. Each is analyzed with
# the stops between the neighboring thresholds (and 0.03 / 0.25 at the ends).
CUTOFFS = {0.08: (0.03, 0.15), 0.15: (0.08, 0.25)}
# placebo cutoffs with no change in the law, inside a window that has none
PLACEBOS = {0.05: (0.03, 0.08), 0.115: (0.08, 0.15), 0.20: (0.15, 0.25)}


def load() -> pd.DataFrame:
    df = pd.read_stata(DATA)
    df["bac"] = df["low_score"].astype(float) / 1000
    return df


def window(df: pd.DataFrame, c: float, lo: float, hi: float):
    """Cases with lo <= bac < hi, centered at the cutoff c.

    Returns x (bac - c), y (recidivism), d, and the covariate matrix.
    """
    w = df[(df["bac"] >= lo - 1e-9) & (df["bac"] < hi - 1e-9)]
    x = np.round(w["bac"].to_numpy(float) - c, 6)
    y = w["recidivism"].to_numpy(float)
    d = (x >= -1e-9).astype(float)
    z = w[COVARIATES].to_numpy(float)
    return x, y, d, z
