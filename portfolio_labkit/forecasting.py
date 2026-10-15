"""Forecasting con modelos SARIMA y búsqueda en malla (grid search)."""

import itertools
import warnings
from dataclasses import dataclass, field
from typing import Iterable, Optional, Sequence, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from joblib import Parallel, delayed
from statsmodels.tsa.statespace.sarimax import SARIMAX

from .timeseries import _as_series

Order = Tuple[int, int, int]
SeasonalOrder = Tuple[int, int, int, int]


def _future_index(index: pd.Index, steps: int) -> pd.Index:
    """Índice de las fechas futuras (días hábiles si la serie no tiene fines de semana)."""
    if not isinstance(index, pd.DatetimeIndex):
        start = index[-1] + 1 if len(index) and np.issubdtype(index.dtype, np.integer) else len(index)
        return pd.RangeIndex(start, start + steps)
    freq = index.freq or pd.infer_freq(index)
    if freq is None:
        freq = 'B' if (index.dayofweek < 5).all() else 'D'
    return pd.date_range(index[-1], periods=steps + 1, freq=freq)[1:]


def _sarimax(values, order, seasonal_order, trend, **kw):
    return SARIMAX(values, order=order, seasonal_order=seasonal_order, trend=trend,
                   enforce_stationarity=False, enforce_invertibility=False, **kw)


@dataclass
class SarimaFit:
    """Modelo SARIMA ajustado."""
    results: object = field(repr=False)
    order: Order
    seasonal_order: SeasonalOrder
    series: pd.Series = field(repr=False)

    @property
    def aic(self) -> float:
        return float(self.results.aic)

    @property
    def bic(self) -> float:
        return float(self.results.bic)

    def summary(self):
        return self.results.summary()

    def forecast(self, steps: int = 30, alpha: float = 0.05) -> pd.DataFrame:
        """Pronóstico a `steps` períodos: media e intervalo de confianza (1 - alpha)."""
        pred = self.results.get_forecast(steps=steps)
        ci = pred.conf_int(alpha=alpha)
        idx = _future_index(self.series.index, steps)
        return pd.DataFrame({
            'forecast': np.asarray(pred.predicted_mean),
            'lower': np.asarray(ci)[:, 0],
            'upper': np.asarray(ci)[:, 1],
        }, index=idx)

    def plot_forecast(self, steps: int = 30, alpha: float = 0.05, history: int = 250,
                      ax: Optional[plt.Axes] = None) -> Tuple[pd.DataFrame, plt.Axes]:
        """Grafica los últimos `history` datos y el pronóstico con su intervalo."""
        fc = self.forecast(steps, alpha)
        if ax is None:
            sns.set_theme(style='whitegrid')
            _, ax = plt.subplots(figsize=(10, 5))
        hist = self.series.iloc[-history:]
        sns.lineplot(x=hist.index, y=hist.values, ax=ax, label='Histórico')
        sns.lineplot(x=fc.index, y=fc['forecast'].values, ax=ax, label='Pronóstico', color='#c44e52')
        ax.fill_between(fc.index, fc['lower'], fc['upper'], color='#c44e52', alpha=0.2,
                        label=f'IC {int((1 - alpha) * 100)}%')
        ax.set_title(f'SARIMA{self.order}x{self.seasonal_order}')
        ax.legend()
        return fc, ax


def fit_sarima(
    series,
    order: Order = (1, 0, 0),
    seasonal_order: SeasonalOrder = (0, 0, 0, 0),
    trend: Optional[str] = None,
    maxiter: int = 200,
) -> SarimaFit:
    """Ajusta un SARIMA(p,d,q)x(P,D,Q,s).

    Para precios usa d=1 (o modela los retornos con d=0). `trend` puede ser
    'c' (constante), 't' (tendencia lineal) o 'ct'; con d>0 se omite.
    """
    s = _as_series(series)
    order, seasonal_order = tuple(order), tuple(seasonal_order)
    if order[1] > 0 or seasonal_order[1] > 0:
        trend = None   # con diferenciación la constante/tendencia no es identificable
    model = _sarimax(s.values, order, seasonal_order, trend)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        res = model.fit(disp=False, maxiter=maxiter)
    return SarimaFit(res, order, seasonal_order, s)


# ------------------------------------------------------------------ Grid search

def _as_range(x: Union[int, Iterable[int]]) -> Sequence[int]:
    """Un entero n equivale a range(n + 1); un iterable se usa tal cual."""
    return list(range(x + 1)) if isinstance(x, (int, np.integer)) else list(x)


def _evaluate(values, order, seasonal_order, trend, criterion, n_train, maxiter):
    """Ajusta un candidato; nunca lanza: los fallos quedan registrados."""
    row = {'order': order, 'seasonal_order': seasonal_order,
           'aic': np.nan, 'bic': np.nan, 'rmse': np.nan, 'error': ''}
    try:
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            if criterion == 'oos':
                res = _sarimax(values[:n_train], order, seasonal_order, trend).fit(
                    disp=False, maxiter=maxiter)
                pred = np.asarray(res.forecast(steps=len(values) - n_train))
                row['rmse'] = float(np.sqrt(np.mean((values[n_train:] - pred) ** 2)))
            else:
                res = _sarimax(values, order, seasonal_order, trend).fit(
                    disp=False, maxiter=maxiter)
            row['aic'], row['bic'] = float(res.aic), float(res.bic)
        if not np.isfinite(row[criterion if criterion != 'oos' else 'rmse']):
            row['error'] = 'resultado no finito'
    except Exception as exc:   # LinAlgError, ValueError por no convergencia, etc.
        row['error'] = f'{type(exc).__name__}: {exc}'[:120]
    return row


@dataclass
class GridSearchResult:
    """Tabla de modelos ordenada por el criterio y acceso al mejor."""
    table: pd.DataFrame
    criterion: str
    trend: Optional[str] = None

    @property
    def best_order(self) -> Order:
        return self.table.iloc[0]['order']

    @property
    def best_seasonal_order(self) -> SeasonalOrder:
        return self.table.iloc[0]['seasonal_order']

    def fit_best(self, series, maxiter: int = 200) -> SarimaFit:
        """Reajusta el mejor modelo con toda la serie."""
        return fit_sarima(series, self.best_order, self.best_seasonal_order,
                          trend=self.trend, maxiter=maxiter)

    def __str__(self) -> str:
        return (f'Mejor modelo por {self.criterion.upper()}: SARIMA{self.best_order}x'
                f'{self.best_seasonal_order}\n{self.table.head(10).to_string()}')


def grid_search_sarima(
    series,
    p: Union[int, Iterable[int]] = 2,
    d: Union[int, Iterable[int]] = (0, 1),
    q: Union[int, Iterable[int]] = 2,
    P: Union[int, Iterable[int]] = 0,
    D: Union[int, Iterable[int]] = 0,
    Q: Union[int, Iterable[int]] = 0,
    s: int = 0,
    trend: Optional[str] = None,
    criterion: str = 'aic',
    train_frac: float = 0.8,
    n_jobs: int = -1,
    maxiter: int = 100,
) -> GridSearchResult:
    """Búsqueda en malla del mejor SARIMA, ajustando los candidatos en paralelo.

    p, d, q, P, D, Q: un entero n prueba 0..n; un iterable prueba esos valores.
    s: período estacional (0 o 1 = sin componente estacional).
    criterion: 'aic', 'bic' o 'oos' (RMSE sobre el último 1 - train_frac de la serie).
    n_jobs: núcleos de CPU (-1 = todos).
    Los modelos que no convergen quedan al final de la tabla con su error.
    """
    if criterion not in ('aic', 'bic', 'oos'):
        raise ValueError("criterion debe ser 'aic', 'bic' o 'oos'")
    if not 0 < train_frac < 1:
        raise ValueError('train_frac debe estar entre 0 y 1')
    values = _as_series(series).values
    n_train = int(len(values) * train_frac)

    seasonal = s not in (0, 1)
    seas_grid = (itertools.product(_as_range(P), _as_range(D), _as_range(Q))
                 if seasonal else [(0, 0, 0)])
    candidates = []
    for (pp, dd, qq), (PP, DD, QQ) in itertools.product(
            itertools.product(_as_range(p), _as_range(d), _as_range(q)), list(seas_grid)):
        candidates.append(((pp, dd, qq), (PP, DD, QQ, s if seasonal else 0)))

    rows = Parallel(n_jobs=n_jobs)(
        delayed(_evaluate)(values, o, so, trend if o[1] == 0 and so[1] == 0 else None,
                           criterion, n_train, maxiter)
        for o, so in candidates)

    key = 'rmse' if criterion == 'oos' else criterion
    table = (pd.DataFrame(rows)
             .sort_values(key, na_position='last', kind='stable')
             .reset_index(drop=True))
    return GridSearchResult(table, criterion, trend)
