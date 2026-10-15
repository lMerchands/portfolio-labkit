"""Gráficas de precios y retornos con seaborn."""

from typing import Iterable, Optional, Union

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

_KINDS = ('price', 'returns', 'normalized')


def _select(df: pd.DataFrame, tickers: Optional[Union[str, Iterable[str]]]) -> pd.DataFrame:
    if tickers is None:
        return df
    if isinstance(tickers, str):
        tickers = [tickers]
    tickers = list(tickers)
    missing = [t for t in tickers if t not in df.columns]
    if missing:
        raise KeyError(f'Tickers no disponibles: {missing}. Disponibles: {list(df.columns)}')
    return df[tickers]


def plot_prices(
    prices: pd.DataFrame,
    tickers: Optional[Union[str, Iterable[str]]] = None,
    kind: str = 'price',
    single: bool = False,
    ax: Optional[plt.Axes] = None,
    title: Optional[str] = None,
):
    """Grafica uno o varios activos.

    tickers: un símbolo, una lista, o None para todos.
    kind: 'price' (precio), 'returns' (retorno simple diario) o
          'normalized' (base 100 al inicio, comparable entre activos).
    single: False dibuja todos sobre el mismo lienzo; True, un panel por
            activo (ignora `ax`).
    Devuelve el Axes (o un arreglo de Axes si single=True y hay varios).
    """
    if kind not in _KINDS:
        raise ValueError(f'kind debe ser uno de {_KINDS}')
    df = _select(prices, tickers)
    if kind == 'returns':
        df = df.pct_change().dropna(how='all')
    elif kind == 'normalized':
        df = df / df.apply(lambda c: c.dropna().iloc[0]) * 100

    ylabel = {'price': 'Precio ajustado', 'returns': 'Retorno', 'normalized': 'Base 100'}[kind]
    long = df.rename_axis('Fecha').reset_index().melt(
        id_vars='Fecha', var_name='Activo', value_name=ylabel)

    sns.set_theme(style='whitegrid')
    if single and df.shape[1] > 1:
        fig, axes = plt.subplots(df.shape[1], 1, sharex=True,
                                 figsize=(10, 2.6 * df.shape[1]))
        for a, t in zip(axes, df.columns):
            sns.lineplot(data=long[long['Activo'] == t], x='Fecha', y=ylabel, ax=a)
            a.set_ylabel(f'{t}\n{ylabel}')
        axes[-1].set_xlabel('Fecha')
        fig.suptitle(title or ylabel)
        fig.tight_layout()
        return axes

    if ax is None:
        _, ax = plt.subplots(figsize=(10, 5))
    sns.lineplot(data=long, x='Fecha', y=ylabel, hue='Activo', ax=ax)
    ax.set_title(title or ylabel)
    if df.shape[1] == 1 and ax.get_legend() is not None:
        ax.get_legend().set_title('')
    return ax
