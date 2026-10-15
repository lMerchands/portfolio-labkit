"""Análisis de series de tiempo: ACF/PACF, pruebas de estacionariedad y descomposición."""

import warnings
from dataclasses import dataclass, field
from typing import Dict, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.seasonal import STL, seasonal_decompose
from statsmodels.tsa.stattools import acf, adfuller, pacf


def _as_series(x) -> pd.Series:
    """Convierte la entrada a una Series numérica sin NaN."""
    if isinstance(x, pd.DataFrame):
        if x.shape[1] != 1:
            raise ValueError('Se esperaba una serie; recibí un DataFrame con '
                             f'{x.shape[1]} columnas. Selecciona una columna.')
        x = x.iloc[:, 0]
    s = pd.Series(x).astype(float).dropna()
    if len(s) < 10:
        raise ValueError('La serie necesita al menos 10 observaciones.')
    return s


# ---------------------------------------------------------------- ACF / PACF

@dataclass
class ACFResult:
    """Resultados de acf_pacf.

    table: ACF y PACF por rezago con intervalos y bandera de significancia.
    ljung_box: prueba de autocorrelación conjunta (H0: no hay autocorrelación).
    adf: prueba Dickey-Fuller aumentada (H0: raíz unitaria, no estacionaria).
    """
    table: pd.DataFrame
    ljung_box: pd.DataFrame
    adf: Dict[str, float]
    alpha: float
    figure: Optional[plt.Figure] = field(default=None, repr=False)

    @property
    def significant_acf(self):
        return self.table.index[self.table['acf_significant']].tolist()

    @property
    def significant_pacf(self):
        return self.table.index[self.table['pacf_significant']].tolist()

    def summary(self) -> str:
        a = self.alpha
        stationary = self.adf['pvalue'] < a
        lb = self.ljung_box
        lines = [
            f"ADF: estadístico={self.adf['statistic']:.3f}, p-valor={self.adf['pvalue']:.4f} "
            f"-> {'estacionaria' if stationary else 'NO estacionaria'} (alpha={a})",
            'Ljung-Box (H0: sin autocorrelación):',
        ]
        for lag, row in lb.iterrows():
            verdict = 'hay autocorrelación' if row['lb_pvalue'] < a else 'sin evidencia'
            lines.append(f"  rezago {lag}: Q={row['lb_stat']:.2f}, p={row['lb_pvalue']:.4f} -> {verdict}")
        lines.append(f'Rezagos significativos en ACF:  {self.significant_acf[:15]}')
        lines.append(f'Rezagos significativos en PACF: {self.significant_pacf[:15]}')
        return '\n'.join(lines)

    def __str__(self) -> str:
        return self.summary()


def _bar_panel(ax, table, col, lo, hi, title):
    lags = table.index.values
    sns.barplot(x=lags, y=table[col].values, color='#4c72b0', width=0.25, ax=ax)
    band_hi = (hi - table[col]).values
    band_lo = (table[col] - lo).values
    ax.fill_between(np.arange(len(lags)), -band_lo.clip(min=0), band_hi.clip(min=0),
                    color='#c44e52', alpha=0.2, label='Banda de confianza')
    ax.axhline(0, color='black', linewidth=0.8)
    ax.set_title(title)
    ax.set_xlabel('Rezago')
    step = max(1, len(lags) // 20)
    ax.set_xticks(np.arange(0, len(lags), step))
    ax.set_xticklabels(lags[::step])


def acf_pacf(
    series,
    nlags: int = 40,
    alpha: float = 0.05,
    plot: bool = True,
    ljung_lags=(5, 10, 20),
) -> ACFResult:
    """ACF y PACF con intervalos de confianza, Ljung-Box y ADF.

    series: Series (o DataFrame de una columna). Para precios, usa retornos
    o una serie diferenciada; la ACF de un precio no estacionario decae lento.
    """
    s = _as_series(series)
    n = len(s)
    nlags = int(min(nlags, n // 2 - 1))

    with warnings.catch_warnings():
        warnings.simplefilter('ignore', FutureWarning)
        acf_vals, acf_ci = acf(s, nlags=nlags, alpha=alpha, fft=True)
    pacf_vals, pacf_ci = pacf(s, nlags=nlags, alpha=alpha)
    table = pd.DataFrame({
        'acf': acf_vals, 'acf_lo': acf_ci[:, 0], 'acf_hi': acf_ci[:, 1],
        'pacf': pacf_vals, 'pacf_lo': pacf_ci[:, 0], 'pacf_hi': pacf_ci[:, 1],
    }, index=pd.RangeIndex(nlags + 1, name='lag'))
    # el rezago 0 vale 1 por definición: no se evalúa
    table['acf_significant'] = (table['acf_lo'] > 0) | (table['acf_hi'] < 0)
    table['pacf_significant'] = (table['pacf_lo'] > 0) | (table['pacf_hi'] < 0)
    table.loc[0, ['acf_significant', 'pacf_significant']] = False

    lags = [l for l in ljung_lags if l <= nlags] or [nlags]
    lb = acorr_ljungbox(s, lags=lags)
    lb.index.name = 'lag'

    with warnings.catch_warnings():
        warnings.simplefilter('ignore', FutureWarning)
        stat, pval, usedlag, nobs, crit, _ = adfuller(s, autolag='AIC')
    adf = {'statistic': float(stat), 'pvalue': float(pval), 'used_lag': int(usedlag),
           'nobs': int(nobs), **{f'crit_{k}': float(v) for k, v in crit.items()}}

    fig = None
    if plot:
        sns.set_theme(style='whitegrid')
        fig, axes = plt.subplots(2, 1, figsize=(10, 7))
        t = table.iloc[1:]
        _bar_panel(axes[0], t, 'acf', t['acf_lo'], t['acf_hi'], 'ACF')
        _bar_panel(axes[1], t, 'pacf', t['pacf_lo'], t['pacf_hi'], 'PACF')
        fig.tight_layout()
    return ACFResult(table, lb, adf, alpha, fig)


# ------------------------------------------------------------- Descomposición

_FREQ_PERIOD = {'M': 12, 'Q': 4, 'W': 52, 'Y': 1, 'A': 1}


def infer_period(index: pd.Index, n: int) -> int:
    """Período estacional probable según la frecuencia del índice.

    Mensual 12, trimestral 4, semanal 52. Diario: 252 (año bursátil) si hay
    más de dos años de datos y 5 (semana bursátil) en caso contrario.
    """
    freq = None
    if isinstance(index, pd.DatetimeIndex):
        freq = index.freqstr or pd.infer_freq(index)
        if freq is None and len(index) > 2:
            gap = pd.Series(index).diff().dt.days.median()
            freq = 'D' if gap <= 4 else ('W' if gap <= 10 else ('M' if gap <= 40 else 'Q'))
    if freq is None:
        raise ValueError('No pude inferir el período; indícalo con period=.')
    key = freq.upper().lstrip('0123456789')[:1]
    if key in ('D', 'B'):
        return 252 if n >= 2 * 252 else 5
    if key in _FREQ_PERIOD:
        return _FREQ_PERIOD[key]
    raise ValueError(f'Frecuencia {freq!r} no reconocida; indica period=.')


@dataclass
class DecompositionResult:
    """Componentes observado, tendencia, estacionalidad y residuo."""
    components: pd.DataFrame
    period: int
    model: str
    method: str
    trend_strength: float
    seasonal_strength: float
    figure: Optional[plt.Figure] = field(default=None, repr=False)

    def summary(self) -> str:
        return (f'Descomposición {self.method.upper()} ({self.model}), período={self.period}\n'
                f'  Fuerza de la tendencia:      {self.trend_strength:.3f}\n'
                f'  Fuerza de la estacionalidad: {self.seasonal_strength:.3f}  '
                '(0 = ninguna, 1 = total)')

    def __str__(self) -> str:
        return self.summary()


def decompose(
    series,
    period: Optional[int] = None,
    model: str = 'additive',
    method: str = 'stl',
    plot: bool = True,
) -> DecompositionResult:
    """Descompone la serie en tendencia, estacionalidad y residuo.

    method: 'stl' (robusto, por defecto) o 'classical' (medias móviles).
    model: 'additive' u 'multiplicative' (requiere valores > 0).
    period: observaciones por ciclo; si es None se infiere del índice.

    Los precios diarios no tienen estacionalidad fija: con period=5 o 21 se
    detectan efectos semanales/mensuales, y con 252 el ciclo anual solo si
    hay más de dos años de datos. Para ciclos largos conviene remuestrear a
    mensual (series.resample('ME').last()).
    """
    if model not in ('additive', 'multiplicative'):
        raise ValueError("model debe ser 'additive' o 'multiplicative'")
    if method not in ('stl', 'classical'):
        raise ValueError("method debe ser 'stl' o 'classical'")
    s = _as_series(series)
    period = int(period) if period else infer_period(s.index, len(s))
    if period < 2:
        raise ValueError('period debe ser >= 2')
    if len(s) < 2 * period:
        raise ValueError(f'Se necesitan al menos {2 * period} observaciones '
                         f'para period={period}; hay {len(s)}.')
    mult = model == 'multiplicative'
    if mult and (s <= 0).any():
        raise ValueError('El modelo multiplicativo requiere valores positivos.')

    values = pd.Series(s.values, index=pd.RangeIndex(len(s)))
    if method == 'stl':
        # STL es aditivo: el multiplicativo se resuelve en logaritmos
        fit = STL(np.log(values) if mult else values, period=period, robust=True).fit()
        trend, seasonal, resid = fit.trend, fit.seasonal, fit.resid
        log_space = mult
    else:
        fit = seasonal_decompose(values, model=model, period=period)
        trend, seasonal, resid = fit.trend, fit.seasonal, fit.resid
        log_space = False

    def strength(component):
        r = resid.dropna()
        c = component.loc[r.index]
        if log_space or not mult:
            num, den = r.var(), (r + c).var()
        else:   # clásico multiplicativo: fuerzas en logaritmos
            num, den = np.log(r).var(), (np.log(r) + np.log(c)).var()
        return float(max(0.0, 1 - num / den)) if den > 0 else 0.0

    t_strength, s_strength = strength(trend), strength(seasonal)
    if method == 'stl' and mult:
        trend, seasonal, resid = np.exp(trend), np.exp(seasonal), np.exp(resid)

    comp = pd.DataFrame({'observed': s.values, 'trend': trend.values,
                         'seasonal': seasonal.values, 'resid': resid.values},
                        index=s.index)
    fig = None
    if plot:
        sns.set_theme(style='whitegrid')
        fig, axes = plt.subplots(4, 1, figsize=(10, 9), sharex=True)
        titles = ['Observado', 'Tendencia', 'Estacionalidad', 'Residuo']
        for ax, col, name in zip(axes, comp.columns, titles):
            sns.lineplot(x=comp.index, y=comp[col], ax=ax, linewidth=1)
            ax.set_ylabel(name)
        fig.suptitle(f'Descomposición {method.upper()} ({model}), período={period}')
        fig.tight_layout()
    return DecompositionResult(comp, period, model, method, t_strength, s_strength, fig)
