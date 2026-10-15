import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from portfolio_labkit.timeseries import acf_pacf, decompose


def test_ar1_acf_decays_pacf_cuts(ar1):
    res = acf_pacf(ar1, nlags=20)
    assert 1 in res.significant_pacf
    assert 1 in res.significant_acf and 2 in res.significant_acf
    assert not any(l in res.significant_pacf for l in (5, 6, 7, 8))
    assert res.adf['pvalue'] < 0.05
    assert (res.ljung_box['lb_pvalue'] < 0.05).all()
    assert 'estacionaria' in res.summary()
    assert res.figure is not None
    plt.close('all')


def test_random_walk_not_stationary(prices):
    res = acf_pacf(prices['AAA'], plot=False)
    assert res.adf['pvalue'] > 0.05
    assert 'NO estacionaria' in res.summary()


def test_white_noise_no_autocorr():
    x = pd.Series(np.random.default_rng(5).normal(size=500))
    res = acf_pacf(x, plot=False)
    assert (res.ljung_box['lb_pvalue'] > 0.01).all()


def _seasonal(period=12, n=240, mult=False):
    t = np.arange(n)
    rng = np.random.default_rng(2)
    season = np.sin(2 * np.pi * t / period)
    if mult:
        y = (50 + 0.2 * t) * (1 + 0.3 * season) * np.exp(rng.normal(0, 0.01, n))
    else:
        y = 0.1 * t + 3 * season + rng.normal(0, 0.2, n)
    return pd.Series(y, index=pd.date_range('2000-01-31', periods=n, freq='ME'))


@pytest.mark.parametrize('method', ['stl', 'classical'])
@pytest.mark.parametrize('model', ['additive', 'multiplicative'])
def test_decompose_recovers_seasonality(method, model):
    res = decompose(_seasonal(mult=model == 'multiplicative'), model=model,
                    method=method, plot=False)
    assert res.period == 12                      # inferido del índice mensual
    assert res.seasonal_strength > 0.9 and res.trend_strength > 0.9
    assert list(res.components.columns) == ['observed', 'trend', 'seasonal', 'resid']
    if method == 'stl':
        recon = (res.components[['trend', 'seasonal', 'resid']].sum(axis=1)
                 if model == 'additive' else
                 res.components[['trend', 'seasonal', 'resid']].prod(axis=1))
        assert np.allclose(recon, res.components['observed'])


def test_decompose_errors(prices):
    with pytest.raises(ValueError):
        decompose(prices['AAA'].iloc[:30], period=21)
    with pytest.raises(ValueError):
        decompose(-prices['AAA'], period=5, model='multiplicative')
    assert decompose(prices['AAA'], period=21, plot=False).period == 21
    plt.close('all')
