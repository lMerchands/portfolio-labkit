import numpy as np
import pandas as pd
import pytest

from portfolio_labkit.risk import historical_cvar, historical_var, risk_table


@pytest.fixture
def normal():
    return pd.Series(np.random.default_rng(0).normal(size=200_000))


def test_normal_matches_theory(normal):
    assert historical_var(normal, 0.05) == pytest.approx(1.645, rel=0.02)
    assert historical_cvar(normal, 0.05) == pytest.approx(2.063, rel=0.02)
    assert historical_var(normal, 0.01) == pytest.approx(2.326, rel=0.02)


def test_alpha_and_confidence_are_equivalent(normal):
    assert historical_var(normal, 0.95) == historical_var(normal, 0.05)
    assert historical_cvar(normal, 0.99) == historical_cvar(normal, 0.01)


def test_cvar_exceeds_var_and_levels_ordered(normal):
    for a in (0.1, 0.05, 0.01):
        assert historical_cvar(normal, a) >= historical_var(normal, a)
    assert historical_var(normal, 0.01) > historical_var(normal, 0.05)


def test_known_small_sample():
    r = pd.Series(np.arange(-10, 10) / 100)       # -10% ... +9%
    q = np.quantile(r, 0.1)
    assert historical_var(r, 0.1) == pytest.approx(-q)
    assert historical_cvar(r, 0.1) == pytest.approx(-r[r <= q].mean())


def test_sqrt_scaling(normal):
    assert historical_var(normal, 0.05, horizon=10) == pytest.approx(
        historical_var(normal, 0.05) * np.sqrt(10))


def test_rolling_matches_manual(normal):
    x = normal.iloc[:1000]
    manual = x.rolling(5).sum().dropna()
    assert historical_var(x, 0.05, 5, 'rolling') == pytest.approx(-np.quantile(manual, 0.05))
    q = np.quantile(manual, 0.05)
    assert historical_cvar(x, 0.05, 5, 'rolling') == pytest.approx(-manual[manual <= q].mean())


def test_fat_tails_have_heavier_cvar_ratio(normal):
    t3 = pd.Series(np.random.default_rng(1).standard_t(3, size=200_000))
    ratio = lambda s: historical_cvar(s, 0.01) / historical_var(s, 0.01)
    assert ratio(t3) > ratio(normal)


def test_nan_are_dropped():
    r = pd.Series([np.nan, -0.02, 0.01, 0.0, np.nan, 0.03, -0.01])
    assert np.isfinite(historical_var(r, 0.2))


def test_errors(normal):
    for bad in (0, 1, -0.1, 1.5):
        with pytest.raises(ValueError):
            historical_var(normal, bad)
    with pytest.raises(ValueError):
        historical_var(normal, 0.05, horizon=0)
    with pytest.raises(ValueError):
        historical_var(normal, 0.05, horizon=2.5)
    with pytest.raises(ValueError):
        historical_var(normal, 0.05, horizon=5, horizon_method='x')
    with pytest.raises(ValueError):
        historical_var(normal.iloc[:50], 0.05, horizon=5, horizon_method='rolling')
    with pytest.raises(ValueError):
        historical_var(pd.Series([0.1]), 0.05)


def test_risk_table(normal):
    t = risk_table(normal, (0.95, 0.99), value=1_000_000)
    assert list(t.index) == ['95%', '99%']
    assert t.loc['99%', 'VaR'] > t.loc['95%', 'VaR']
    assert t.loc['95%', 'VaR_monto'] == pytest.approx(t.loc['95%', 'VaR'] * 1e6)
    assert 'VaR_monto' not in risk_table(normal).columns
