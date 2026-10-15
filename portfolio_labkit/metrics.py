"""Métricas de portafolio.

Basado en metrics_portfolio.py de camilocastillo (agosto de 2025).
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class PortfolioMetrics:
    ann_return: float
    ann_vol: float
    sharpe: float
    cv: float


def annualize_stats(
    rets: pd.DataFrame,
    weights: np.ndarray,
    rf: float = 0.0,
    periods_per_year: int = 252,
) -> PortfolioMetrics:
    """Retorno y volatilidad anualizados, Sharpe y coeficiente de variación.

    rets: retornos (diarios por defecto); weights: pesos que suman 1;
    rf: tasa libre de riesgo anual (0.05 = 5%).
    """
    w = np.asarray(weights, dtype=float).reshape(-1, 1)
    mu_vec = rets.mean().values.reshape(-1, 1) * periods_per_year
    cov_mat = rets.cov().values * periods_per_year
    port_mu = float((w.T @ mu_vec)[0, 0])
    port_sigma = float(np.sqrt(w.T @ cov_mat @ w)[0, 0])
    sharpe = (port_mu - rf) / port_sigma if port_sigma > 0 else np.nan
    cv = port_sigma / port_mu if abs(port_mu) > 1e-12 else np.nan
    return PortfolioMetrics(port_mu, port_sigma, sharpe, cv)
