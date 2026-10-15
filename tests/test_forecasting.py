import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from portfolio_labkit.forecasting import fit_sarima, grid_search_sarima


def _ar2(n=500, seed=3):
    rng = np.random.default_rng(seed)
    e = rng.normal(size=n)
    x = np.zeros(n)
    for t in range(2, n):
        x[t] = 0.5 * x[t - 1] - 0.4 * x[t - 2] + e[t]
    return pd.Series(x, index=pd.bdate_range('2020-01-01', periods=n))


def test_fit_and_forecast_index_and_intervals():
    fit = fit_sarima(_ar2(), order=(2, 0, 0))
    assert abs(fit.results.params[0] - 0.5) < 0.15 and abs(fit.results.params[1] + 0.4) < 0.15
    fc = fit.forecast(10)
    assert len(fc) == 10
    assert (fc['lower'] < fc['forecast']).all() and (fc['forecast'] < fc['upper']).all()
    assert (fc.index > fit.series.index[-1]).all()
    assert (fc.index.dayofweek < 5).all()          # días hábiles
    _, ax = fit.plot_forecast(10)
    assert ax is not None
    plt.close('all')


def test_grid_search_recovers_ar2_by_bic():
    res = grid_search_sarima(_ar2(), p=3, d=[0], q=2, criterion='bic', n_jobs=1)
    assert res.best_order == (2, 0, 0)
    assert res.table['bic'].is_monotonic_increasing
    assert res.fit_best(_ar2()).order == res.best_order


def test_grid_search_parallel_matches_serial():
    x = _ar2(300)
    a = grid_search_sarima(x, p=2, d=[0], q=1, n_jobs=1).table
    b = grid_search_sarima(x, p=2, d=[0], q=1, n_jobs=2).table
    assert list(a['order']) == list(b['order'])
    assert np.allclose(a['aic'], b['aic'])


def test_seasonal_grid_and_oos():
    rng = np.random.default_rng(4)
    t = np.arange(240)
    y = pd.Series(5 * np.sin(2 * np.pi * t / 12) + rng.normal(0, 0.5, 240),
                  index=pd.date_range('2000-01-31', periods=240, freq='ME'))
    res = grid_search_sarima(y, p=1, d=[0], q=0, P=[0, 1], D=[0], Q=[0], s=12,
                             criterion='oos', n_jobs=1)
    assert res.best_seasonal_order == (1, 0, 0, 12)
    assert res.table['rmse'].is_monotonic_increasing


def test_failures_are_reported_not_raised():
    res = grid_search_sarima(_ar2(60), p=1, d=[0], q=1, n_jobs=1)
    assert 'error' in res.table.columns
    with pytest.raises(ValueError):
        grid_search_sarima(_ar2(60), criterion='x')
