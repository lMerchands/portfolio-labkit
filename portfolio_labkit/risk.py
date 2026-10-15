"""VaR y CVaR históricos (por cuantiles) de una serie de retornos.

Las pérdidas se reportan como número positivo: un VaR de 0.02 significa que,
con la confianza indicada, no se espera perder más de 2% en el horizonte.
"""

from typing import Iterable, Optional

import numpy as np
import pandas as pd

_METHODS = ('sqrt', 'rolling')


def _tail_prob(alpha: float) -> float:
    """Probabilidad de la cola izquierda.

    Acepta alpha = 0.05 (probabilidad de cola) o 0.95 / 0.99 (nivel de
    confianza); cualquier valor mayor a 0.5 se interpreta como confianza.
    """
    if not 0 < alpha < 1:
        raise ValueError('alpha debe estar entre 0 y 1 (0.05 o 0.95 para el 95%)')
    return 1.0 - alpha if alpha > 0.5 else alpha


def _clean(returns) -> np.ndarray:
    x = np.asarray(pd.Series(returns).dropna(), dtype=float)
    if x.size < 2:
        raise ValueError('Se necesitan al menos 2 retornos.')
    return x


def _horizon_returns(x: np.ndarray, horizon: int, method: str):
    """Retornos a h períodos y factor de escala. Devuelve (retornos, factor)."""
    if method not in _METHODS:
        raise ValueError(f'horizon_method debe ser uno de {_METHODS}')
    if int(horizon) != horizon or horizon < 1:
        raise ValueError('horizon debe ser un entero >= 1')
    horizon = int(horizon)
    if horizon == 1:
        return x, 1.0
    if method == 'sqrt':
        return x, float(np.sqrt(horizon))
    if x.size - horizon + 1 < 100:
        raise ValueError(f'Con horizon={horizon} quedan solo {x.size - horizon + 1} ventanas '
                         "móviles; usa horizon_method='sqrt' o más datos.")
    # suma móvil de retornos logarítmicos (los log-retornos son aditivos)
    csum = np.concatenate(([0.0], np.cumsum(x)))
    return csum[horizon:] - csum[:-horizon], 1.0


def historical_var(returns, alpha: float = 0.05, horizon: int = 1,
                   horizon_method: str = 'sqrt') -> float:
    """VaR histórico: pérdida en el cuantil `alpha` de la distribución empírica.

    returns: retornos del portafolio (logarítmicos si usas horizon_method='rolling').
    alpha: 0.05 o 0.95 para el 95% de confianza.
    horizon: días del horizonte. 'sqrt' escala el VaR de 1 día por raíz de h
        (supone retornos independientes); 'rolling' usa sumas móviles de h
        retornos, sin supuesto de distribución pero con ventanas solapadas.
    """
    p = _tail_prob(alpha)
    r, scale = _horizon_returns(_clean(returns), horizon, horizon_method)
    return float(-np.quantile(r, p) * scale)


def historical_cvar(returns, alpha: float = 0.05, horizon: int = 1,
                    horizon_method: str = 'sqrt') -> float:
    """CVaR (expected shortfall) histórico: pérdida promedio más allá del VaR.

    Mismos parámetros que historical_var. Siempre es >= al VaR.
    """
    p = _tail_prob(alpha)
    r, scale = _horizon_returns(_clean(returns), horizon, horizon_method)
    q = np.quantile(r, p)
    return float(-r[r <= q].mean() * scale)


def risk_table(returns, alphas: Iterable[float] = (0.95, 0.99), horizon: int = 1,
               horizon_method: str = 'sqrt', value: Optional[float] = None) -> pd.DataFrame:
    """VaR y CVaR históricos para varios niveles de confianza.

    value: monto del portafolio; si se da, agrega las columnas en dinero.
    """
    rows = {}
    for a in alphas:
        conf = 1.0 - _tail_prob(a)
        rows[f'{conf:.0%}'] = {
            'VaR': historical_var(returns, a, horizon, horizon_method),
            'CVaR': historical_cvar(returns, a, horizon, horizon_method),
        }
    table = pd.DataFrame(rows).T
    table.index.name = 'confianza'
    if value is not None:
        table['VaR_monto'] = table['VaR'] * value
        table['CVaR_monto'] = table['CVaR'] * value
    return table
