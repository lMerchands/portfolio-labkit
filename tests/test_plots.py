import matplotlib.pyplot as plt
import pytest


from portfolio_labkit.plots import plot_prices


def _data_lines(ax):
    # seaborn agrega líneas auxiliares en la leyenda: solo cuentan las de datos
    return [l for l in ax.get_lines() if len(l.get_xdata()) > 10]


def test_multiple_assets_one_canvas(prices):
    ax = plot_prices(prices)
    assert len(_data_lines(ax)) == 3
    plt.close('all')


def test_single_asset(prices):
    ax = plot_prices(prices, 'AAA', kind='returns')
    assert len(_data_lines(ax)) == 1
    plt.close('all')


def test_normalized_starts_at_100(prices):
    ax = plot_prices(prices, ['AAA', 'BBB'], kind='normalized')
    assert all(abs(l.get_ydata()[0] - 100) < 1e-9 for l in _data_lines(ax))
    plt.close('all')


def test_single_panels(prices):
    axes = plot_prices(prices, single=True)
    assert len(axes) == 3
    plt.close('all')


def test_errors(prices):
    with pytest.raises(KeyError):
        plot_prices(prices, 'ZZZ')
    with pytest.raises(ValueError):
        plot_prices(prices, kind='x')
