import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from portfolio_labkit.portfolio import (Portfolio, black_litterman, implied_returns,
                                        views_from_dict)


@pytest.fixture
def port(prices):
    return Portfolio(prices)


def test_estimators_are_annual(port, prices):
    mu, sigma = port.estimate_mean(), port.estimate_cov()
    rets = np.log(prices / prices.shift(1)).dropna()
    assert np.allclose(mu, rets.mean() * 252)
    assert np.allclose(sigma, rets.cov() * 252)
    assert port.estimate_cov('ledoit').shape == (3, 3)


def test_random_weights_sum_to_one(port):
    for short in (False, True):
        W = port.random_weights(500, short=short, seed=1)
        assert np.allclose(W.sum(axis=1), 1)
    assert (port.random_weights(500, short=False, seed=1) >= 0).all()
    assert (port.random_weights(500, short=True, seed=1) < 0).any()


def test_monte_carlo_matches_metrics(port):
    df = port.monte_carlo(50, seed=2, rf=0.02)
    w = df.filter(like='w_').iloc[0].values
    m = port.metrics(pd.Series(w), rf=0.02)
    assert np.isclose(df.loc[0, 'ann_return'], m.ann_return)
    assert np.isclose(df.loc[0, 'ann_vol'], m.ann_vol)
    assert np.isclose(df.loc[0, 'Sharpe'], m.sharpe)
    best = port.pick_candidates(2000, seed=3)
    all_mc = port.monte_carlo(2000, seed=3)
    assert np.isclose(best['max Sharpe']['Sharpe'], all_mc['Sharpe'].max())
    assert np.isclose(best['min Vol']['ann_vol'], all_mc['ann_vol'].min())


def test_markowitz_min_vol_beats_random(port):
    w = port.markowitz('min_vol')
    assert np.isclose(w.sum(), 1) and (w >= -1e-9).all()
    vol = port.metrics(w).ann_vol
    assert vol <= port.monte_carlo(3000, seed=0)['ann_vol'].min() + 1e-6
    ws = port.markowitz('min_vol', short=True)
    assert np.isclose(ws.sum(), 1)
    assert port.metrics(ws).ann_vol <= vol + 1e-9


def test_markowitz_target_and_errors(port):
    mu = port.estimate_mean()
    r_min = port.metrics(port.markowitz('min_vol')).ann_return
    r = (r_min + float(mu.max())) / 2     # entre la mínima varianza y el mayor retorno
    w = port.markowitz('efficient_return', target_return=r)
    assert np.isclose(port.metrics(w).ann_return, r, atol=1e-4)
    with pytest.raises(ValueError):
        port.markowitz('efficient_return')
    with pytest.raises(ValueError):
        port.markowitz('otro')


def test_frontier_is_monotone(port):
    fr = port.efficient_frontier(15)
    assert len(fr) >= 5
    assert fr['ann_vol'].is_monotonic_increasing
    port.plot_frontier(n_points=10, n_random=200)
    plt.close('all')


def test_merton_formula(port):
    mu, sigma = port.estimate_mean(), port.estimate_cov()
    w = port.merton(3.0, rf=0.02)
    assert np.allclose(w.values, np.linalg.inv(sigma.values) @ (mu.values - 0.02) / 3.0)
    df = port.merton([2, 4, 8], rf=0.02)
    assert np.allclose(df[port.tickers].values[0] * 2, df[port.tickers].values[2] * 8)
    assert np.allclose(df.drop(columns='w_rf').sum(axis=1) + df['w_rf'], 1)
    with pytest.raises(ValueError):
        port.merton(0)


def test_hrp(port):
    w = port.hrp()
    assert np.isclose(w.sum(), 1) and (w > 0).all()
    assert list(w.index) == port.tickers


# ---------------------------------------------------------- Black-Litterman

@pytest.fixture
def setup(port):
    sigma = port.estimate_cov()
    pi = implied_returns(sigma, delta=2.5)
    return sigma, pi


def test_views_from_dict(port):
    P, Q = views_from_dict({'AAA': 0.1, ('BBB', 'CCC'): 0.02}, port.tickers)
    assert P.values.tolist() == [[1, 0, 0], [0, 1, -1]]
    assert Q.tolist() == [0.1, 0.02]
    P2, _ = views_from_dict([({'AAA': 1, 'BBB': -0.5, 'CCC': -0.5}, 0.0)], port.tickers)
    assert P2.values.tolist() == [[1, -0.5, -0.5]]
    with pytest.raises(KeyError):
        views_from_dict({'ZZZ': 0.1}, port.tickers)


def test_bl_view_equal_to_prior_changes_nothing(port, setup):
    sigma, pi = setup
    P, _ = views_from_dict({'AAA': 0, ('BBB', 'CCC'): 0}, port.tickers)
    Q = P.values @ pi.values
    mu, _ = black_litterman(sigma, pi, P, Q)
    assert np.allclose(mu, pi)


def test_bl_certain_views_are_reproduced(port, setup):
    sigma, pi = setup
    P, Q = views_from_dict({'AAA': 0.20, ('BBB', 'CCC'): 0.03}, port.tickers)
    mu, sig = black_litterman(sigma, pi, P, Q, omega=np.zeros((2, 2)) + 1e-12 * np.eye(2))
    assert np.allclose(P.values @ mu.values, Q.values, atol=1e-6)
    assert np.allclose(sig, sig.T)
    assert np.all(np.linalg.eigvalsh(sig.values) > 0)


def test_bl_confidence_moves_between_prior_and_view(port, setup):
    sigma, pi = setup
    P, Q = views_from_dict({'AAA': 0.30}, port.tickers)
    lo = black_litterman(sigma, pi, P, Q, confidences=[0.05])[0]['AAA']
    hi = black_litterman(sigma, pi, P, Q, confidences=[0.95])[0]['AAA']
    assert pi['AAA'] < lo < hi < 0.30


def test_bl_matches_pypfopt(port, setup):
    from pypfopt import BlackLittermanModel
    sigma, pi = setup
    P, Q = views_from_dict({'AAA': 0.12, ('BBB', 'CCC'): 0.02}, port.tickers)
    omega = np.diag([0.001, 0.002])
    mu, sig = black_litterman(sigma, pi, P, Q, tau=0.05, omega=omega)
    ref = BlackLittermanModel(sigma, pi=pi, Q=Q.values, P=P.values, omega=omega, tau=0.05)
    assert np.allclose(mu.values, ref.bl_returns().values, atol=1e-8)
    assert np.allclose(sig.values, ref.bl_cov().values, atol=1e-8)


def test_portfolio_black_litterman_pipeline(port):
    port.estimate_cov()
    mu, sig = port.black_litterman({'AAA': 0.25, ('BBB', 'CCC'): 0.02},
                                   confidences=[0.8, 0.5], tau=0.05)
    base = port.markowitz('max_sharpe', rf=0.0, mu=port.pi, sigma=port.sigma)
    w = port.optimize_bl('markowitz', objective='max_sharpe', rf=0.0)
    assert np.isclose(w.sum(), 1)
    assert w['AAA'] >= base['AAA'] - 1e-9          # opinión alcista sobre AAA
    wm = port.optimize_bl('merton', gamma=3, rf=0.0)
    assert len(wm) == 3
    with pytest.raises(ValueError):
        port.black_litterman()
    with pytest.raises(ValueError):
        Portfolio(port.prices).optimize_bl()
