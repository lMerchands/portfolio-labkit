import matplotlib
matplotlib.use('Agg')

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def prices():
    """Precios sintéticos de 3 activos, ~3 años de días hábiles."""
    rng = np.random.default_rng(0)
    idx = pd.bdate_range('2021-01-01', periods=800)
    rets = rng.normal(0.0004, 0.01, size=(800, 3))
    return pd.DataFrame(100 * np.exp(rets.cumsum(axis=0)), index=idx, columns=['AAA', 'BBB', 'CCC'])


@pytest.fixture
def ar1():
    rng = np.random.default_rng(1)
    e = rng.normal(size=600)
    x = np.zeros(600)
    for t in range(1, 600):
        x[t] = 0.7 * x[t - 1] + e[t]
    return pd.Series(x, index=pd.bdate_range('2020-01-01', periods=600))
