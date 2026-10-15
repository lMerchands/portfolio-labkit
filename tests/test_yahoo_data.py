import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from portfolio_labkit import yahoo_data as yd


@pytest.fixture
def datos(prices, monkeypatch):
    """YahooData sin red: yf.download devuelve los precios sintéticos."""
    monkeypatch.setattr(yd.yf, 'download',
                        lambda *a, **k: pd.concat({'Adj Close': prices}, axis=1))
    return yd.YahooData(['AAA', 'BBB', 'CCC'], start='2021-01-01')


def test_download_and_returns(datos, prices):
    assert list(datos.data.columns) == ['AAA', 'BBB', 'CCC']
    r = datos.compute_returns('simple')
    assert np.allclose(r.iloc[0], prices.iloc[1] / prices.iloc[0] - 1)
    with pytest.raises(ValueError):
        datos.compute_returns('x')


def test_shortcuts(datos):
    assert datos.plot(['AAA', 'BBB'], kind='normalized') is not None
    assert datos.acf_pacf('AAA', nlags=10, plot=False).adf['pvalue'] <= 1
    assert datos.decompose('AAA', period=21, plot=False).period == 21
    fc = datos.forecast('AAA', steps=5, order=(1, 1, 0), plot=False)
    assert len(fc) == 5
    gs = datos.grid_search('AAA', kind='log', p=1, d=[0], q=1, n_jobs=1)
    assert len(gs.table) == 4
    p = datos.portfolio()
    assert np.isclose(p.markowitz('min_vol').sum(), 1)
    with pytest.raises(KeyError):
        datos.series('ZZZ')
    plt.close('all')
