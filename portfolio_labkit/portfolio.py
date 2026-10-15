"""Optimización de portafolios: Monte Carlo, Markowitz, Merton, HRP y Black-Litterman.

Basado en Methodologies_Port.py (clase Do_portfolio) de camilocastillo
(agosto de 2025). Cambios respecto al original: estimadores anualizados de
forma consistente, retornos calculados una sola vez, pesos devueltos como
pd.Series, EfficientFrontier clásico de pypfopt y Black-Litterman.
"""

from typing import Dict, Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd

from .metrics import annualize_stats

Views = Dict[Union[str, Tuple[str, str]], Union[float, Dict[str, float]]]


# ------------------------------------------------------------ Black-Litterman

def implied_returns(sigma: pd.DataFrame, w_mkt: Union[pd.Series, np.ndarray, None] = None,
                    delta: float = 2.5) -> pd.Series:
    """Retornos de equilibrio (prior): pi = delta * Sigma * w_mkt.

    w_mkt: pesos de mercado (p. ej. capitalización); si es None, pesos iguales.
    delta: aversión al riesgo del mercado.
    """
    n = sigma.shape[0]
    w = np.full(n, 1.0 / n) if w_mkt is None else np.asarray(w_mkt, dtype=float)
    return pd.Series(delta * sigma.values @ w, index=sigma.index, name='pi')


def views_from_dict(views: Views, tickers: Sequence[str]) -> Tuple[pd.DataFrame, pd.Series]:
    """Construye las matrices de opiniones P (k x n) y Q (k) desde un diccionario.

    - 'AAPL': 0.12            -> opinión absoluta: AAPL rinde 12% anual.
    - ('AAPL', 'KO'): 0.03    -> relativa: AAPL supera a KO en 3%.

    Para combinaciones arbitrarias (las claves dict no son hashables) pasa una
    lista de pares: [({'AAPL': 1, 'KO': -0.5, 'MU': -0.5}, 0.02), ...].
    """
    items = list(views.items()) if isinstance(views, dict) else list(views)
    tickers = list(tickers)
    P = np.zeros((len(items), len(tickers)))
    Q = np.zeros(len(items))
    names = []
    for i, (key, q) in enumerate(items):
        if isinstance(key, str):
            row, name = {key: 1.0}, f'{key}'
        elif isinstance(key, tuple) and len(key) == 2:
            row, name = {key[0]: 1.0, key[1]: -1.0}, f'{key[0]} - {key[1]}'
        elif isinstance(key, dict):
            row, name = key, ' '.join(f'{v:+g}*{k}' for k, v in key.items())
        else:
            raise ValueError(f'Opinión no reconocida: {key!r}')
        for t, v in row.items():
            if t not in tickers:
                raise KeyError(f'{t!r} no está en los activos: {tickers}')
            P[i, tickers.index(t)] = v
        Q[i] = q
        names.append(name)
    return (pd.DataFrame(P, index=names, columns=tickers),
            pd.Series(Q, index=names, name='Q'))


def black_litterman(
    sigma: pd.DataFrame,
    pi: pd.Series,
    P: Union[pd.DataFrame, np.ndarray],
    Q: Union[pd.Series, np.ndarray],
    tau: float = 0.05,
    omega: Union[np.ndarray, pd.DataFrame, None] = None,
    confidences: Optional[Sequence[float]] = None,
) -> Tuple[pd.Series, pd.DataFrame]:
    """Retornos y covarianza posteriores de Black-Litterman.

    mu_BL    = pi + tau Sigma P' (P tau Sigma P' + Omega)^-1 (Q - P pi)
    Sigma_BL = Sigma + tau Sigma - tau Sigma P' (P tau Sigma P' + Omega)^-1 P tau Sigma

    P: matriz de opiniones (k x n); Q: retornos anuales de cada opinión (k).
    tau: incertidumbre del prior (típico 0.01-0.05).
    omega: covarianza de los errores de las opiniones (k x k). Si es None se usa
           la de He-Litterman, diag(P tau Sigma P'), escalada por `confidences`.
    confidences: confianza en cada opinión en (0, 1]: 0.5 equivale a He-Litterman,
           cercana a 1 la opinión domina y cercana a 0 se ignora.
    """
    cols = list(sigma.columns)
    S = sigma.values
    p_arr = np.atleast_2d(np.asarray(P, dtype=float))
    q_arr = np.asarray(Q, dtype=float).reshape(-1)
    pi_arr = np.asarray(pi, dtype=float).reshape(-1)
    if p_arr.shape[1] != len(cols):
        raise ValueError(f'P tiene {p_arr.shape[1]} columnas y hay {len(cols)} activos')
    if p_arr.shape[0] != len(q_arr):
        raise ValueError('P y Q deben tener el mismo número de opiniones')

    tS = tau * S
    base = p_arr @ tS @ p_arr.T
    if omega is None:
        if confidences is None:
            omega_m = np.diag(np.diag(base))
        else:
            c = np.asarray(confidences, dtype=float).reshape(-1)
            if len(c) != len(q_arr) or ((c <= 0) | (c > 1)).any():
                raise ValueError('confidences: una por opinión, en el rango (0, 1]')
            omega_m = np.diag((1.0 / c - 1.0) * np.diag(base))   # c=0.5 -> He-Litterman
    else:
        omega_m = np.asarray(omega, dtype=float)
        if omega_m.ndim < 2:
            omega_m = np.diag(np.atleast_1d(omega_m))

    gain = tS @ p_arr.T @ np.linalg.inv(base + omega_m)
    mu_bl = pi_arr + gain @ (q_arr - p_arr @ pi_arr)
    sigma_bl = S + tS - gain @ p_arr @ tS
    sigma_bl = (sigma_bl + sigma_bl.T) / 2
    return (pd.Series(mu_bl, index=cols, name='mu_BL'),
            pd.DataFrame(sigma_bl, index=cols, columns=cols))


# ------------------------------------------------------------------ Portafolio

class Portfolio:
    """Estimación de parámetros y optimización de portafolios.

    source: lista de tickers (descarga con YahooData), una instancia de
    YahooData, o un DataFrame de precios (una columna por activo).
    mu y sigma son anuales. Cada método de optimización devuelve los pesos
    como pd.Series y los guarda en `weights`.
    """

    def __init__(self, source, start: str = '2024-01-01', end: Optional[str] = None,
                 kind: str = 'log', periods_per_year: int = 252):
        from .yahoo_data import YahooData
        if isinstance(source, pd.DataFrame):
            self.prices = source.dropna(how='all')
        elif isinstance(source, YahooData):
            self.prices = source.data
            start, end = source.start, source.end
        else:
            self.prices = YahooData(list(source), start=start, end=end).data
        self.start, self.end = start, end
        self.tickers: List[str] = list(self.prices.columns)
        self.kind = kind
        self.ppy = periods_per_year
        self.rets = self._returns(self.prices)
        self.mu: Optional[pd.Series] = None
        self.sigma: Optional[pd.DataFrame] = None
        self.weights = pd.Series(0.0, index=self.tickers, name='weights')

    def _returns(self, prices: pd.DataFrame) -> pd.DataFrame:
        if self.kind == 'log':
            r = np.log(prices / prices.shift(1))
        elif self.kind == 'simple':
            r = prices.pct_change()
        else:
            raise ValueError("kind debe ser 'log' o 'simple'")
        return r.dropna()

    # ------------------------------------------------------------ estimación
    def estimate_mean(self, method: str = 'hist', rf: float = 0.01,
                      market: str = '^GSPC') -> pd.Series:
        """Retorno esperado anual: 'hist' (media histórica) o 'capm'."""
        if method == 'hist':
            self.mu = self.rets.mean() * self.ppy
        elif method == 'capm':
            from pypfopt import expected_returns
            from .yahoo_data import YahooData
            mkt = YahooData([market], start=self.start, end=self.end).data
            self.mu = expected_returns.capm_return(
                self.rets, market_prices=self._returns(mkt), returns_data=True,
                risk_free_rate=rf, compounding=False, frequency=self.ppy,
                log_returns=self.kind == 'log')
        else:
            raise ValueError("method debe ser 'hist' o 'capm'")
        self.mu.name = 'mu'
        return self.mu

    def estimate_cov(self, method: str = 'hist') -> pd.DataFrame:
        """Covarianza anual: 'hist' o 'ledoit' (Ledoit-Wolf, requiere scikit-learn)."""
        if method == 'hist':
            self.sigma = self.rets.cov() * self.ppy
        elif method == 'ledoit':
            from sklearn.covariance import LedoitWolf
            lw = LedoitWolf().fit(self.rets.values)
            self.sigma = pd.DataFrame(lw.covariance_ * self.ppy,
                                      index=self.tickers, columns=self.tickers)
        else:
            raise ValueError("method debe ser 'hist' o 'ledoit'")
        return self.sigma

    def _params(self) -> Tuple[pd.Series, pd.DataFrame]:
        if self.mu is None:
            self.estimate_mean()
        if self.sigma is None:
            self.estimate_cov()
        return self.mu, self.sigma

    def _set_weights(self, w) -> pd.Series:
        self.weights = pd.Series(np.asarray(list(w.values()) if isinstance(w, dict) else w,
                                            dtype=float),
                                 index=list(w.keys()) if isinstance(w, dict) else self.tickers,
                                 name='weights').reindex(self.tickers)
        return self.weights

    # ----------------------------------------------------------- Monte Carlo
    def random_weights(self, n_port: int, short: bool = False,
                       seed: Optional[int] = None) -> np.ndarray:
        """Pesos aleatorios que suman 1.

        Sin cortos: Dirichlet. Con cortos: diferencia de dos Dirichlet
        a - b con sumas 1 + L y L (L en U(0, 0.5)): neto 1, exposición bruta <= 2.
        """
        rng = np.random.default_rng(seed)
        k = len(self.tickers)
        W = rng.dirichlet(np.ones(k), size=n_port)
        if short:
            L = rng.uniform(0, 0.5, size=(n_port, 1))
            W = W * (1 + L) - rng.dirichlet(np.ones(k), size=n_port) * L
        return W

    def monte_carlo(self, n_port: int = 5000, short: bool = False, seed: Optional[int] = None,
                    rf: float = 0.01) -> pd.DataFrame:
        """Simula portafolios aleatorios: retorno, volatilidad, Sharpe, CV y pesos."""
        W = self.random_weights(n_port, short, seed)
        mu = self.rets.mean().values * self.ppy
        cov = self.rets.cov().values * self.ppy
        ret = W @ mu
        vol = np.sqrt(np.einsum('ij,jk,ik->i', W, cov, W))
        df = pd.DataFrame({'ann_return': ret, 'ann_vol': vol,
                           'Sharpe': (ret - rf) / vol, 'CV': vol / np.where(ret == 0, np.nan, ret)})
        for j, t in enumerate(self.tickers):
            df[f'w_{t}'] = W[:, j]
        return df

    def pick_candidates(self, n_port: int = 5000, short: bool = False, rf: float = 0.01,
                        seed: Optional[int] = None) -> Dict[str, pd.Series]:
        """Portafolios candidatos del Monte Carlo: máximo Sharpe y mínima volatilidad."""
        df = self.monte_carlo(n_port, short, seed, rf)
        return {'max Sharpe': df.loc[df['Sharpe'].idxmax()],
                'min Vol': df.loc[df['ann_vol'].idxmin()]}

    # ------------------------------------------------------------- Markowitz
    def markowitz(self, objective: str = 'min_vol', short: bool = False, rf: float = 0.01,
                  target_return: Optional[float] = None,
                  mu: Optional[pd.Series] = None,
                  sigma: Optional[pd.DataFrame] = None) -> pd.Series:
        """Portafolio de media-varianza.

        objective: 'min_vol', 'max_sharpe' o 'efficient_return' (requiere
        target_return anual). short=True permite pesos en [-1, 1].
        mu, sigma: sobrescriben los estimadores (p. ej. los de Black-Litterman).
        """
        from pypfopt import EfficientFrontier
        m, s = (mu, sigma) if mu is not None and sigma is not None else self._params()
        bounds = (-1, 1) if short else (0, 1)
        ef = EfficientFrontier(m, s, weight_bounds=bounds)
        if objective == 'min_vol':
            w = ef.min_volatility()
        elif objective == 'max_sharpe':
            w = ef.max_sharpe(risk_free_rate=rf)
        elif objective == 'efficient_return':
            if target_return is None:
                raise ValueError('efficient_return requiere target_return')
            w = ef.efficient_return(target_return=target_return)
        else:
            raise ValueError("objective debe ser 'min_vol', 'max_sharpe' o 'efficient_return'")
        return self._set_weights(w)

    def efficient_frontier(self, n_points: int = 40, short: bool = False,
                           mu: Optional[pd.Series] = None,
                           sigma: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """Frontera eficiente: retorno y volatilidad de n_points portafolios."""
        from pypfopt import EfficientFrontier
        m, s = (mu, sigma) if mu is not None and sigma is not None else self._params()
        bounds = (-1, 1) if short else (0, 1)
        ef = EfficientFrontier(m, s, weight_bounds=bounds)
        ef.min_volatility()
        lo = ef.portfolio_performance()[0]
        hi = float(m.max()) if not short else float(m.max()) * 0.999
        rows = []
        for r in np.linspace(lo, hi, n_points):
            try:
                ef = EfficientFrontier(m, s, weight_bounds=bounds)
                ef.efficient_return(r)
                ret, vol, _ = ef.portfolio_performance()
                rows.append((ret, vol))
            except Exception:    # objetivo no alcanzable con las restricciones
                continue
        return pd.DataFrame(rows, columns=['ann_return', 'ann_vol'])

    def plot_frontier(self, n_points: int = 40, short: bool = False, n_random: int = 3000,
                      rf: float = 0.01, ax=None):
        """Nube de portafolios aleatorios coloreada por Sharpe y frontera eficiente."""
        import matplotlib.pyplot as plt
        import seaborn as sns
        sns.set_theme(style='whitegrid')
        if ax is None:
            _, ax = plt.subplots(figsize=(9, 6))
        cloud = self.monte_carlo(n_random, short=short, seed=0, rf=rf)
        sc = ax.scatter(cloud['ann_vol'], cloud['ann_return'], c=cloud['Sharpe'],
                        cmap='viridis', s=8, alpha=0.5)
        ax.figure.colorbar(sc, ax=ax, label='Sharpe')
        fr = self.efficient_frontier(n_points, short)
        ax.plot(fr['ann_vol'], fr['ann_return'], color='#c44e52', linewidth=2,
                label='Frontera eficiente')
        ax.set_xlabel('Volatilidad anual')
        ax.set_ylabel('Retorno anual')
        ax.legend()
        return ax

    # ---------------------------------------------------------------- Merton
    def merton(self, gamma: Union[float, Iterable[float]], rf: float = 0.01,
               mu: Optional[pd.Series] = None,
               sigma: Optional[pd.DataFrame] = None) -> Union[pd.Series, pd.DataFrame]:
        """Asignación de Merton: pi = (1/gamma) Sigma^-1 (mu - rf), mu y Sigma anuales.

        gamma escalar -> pd.Series con los pesos riesgosos (y los guarda en `weights`);
        gamma iterable -> DataFrame indexado por gamma con la columna 'w_rf'
        (posición en el activo libre de riesgo = 1 - suma de los pesos).
        """
        m, s = (mu, sigma) if mu is not None and sigma is not None else self._params()
        base = np.linalg.solve(s.values, m.values - rf)
        g = np.atleast_1d(np.asarray(gamma, dtype=float))
        if (g <= 0).any():
            raise ValueError('gamma debe ser positivo')
        W = np.outer(1.0 / g, base)
        if np.ndim(gamma) == 0:
            return self._set_weights(W[0])
        out = pd.DataFrame(W, index=pd.Index(g, name='gamma'), columns=m.index)
        out['w_rf'] = 1.0 - out.sum(axis=1)
        return out

    # ------------------------------------------------------------------- HRP
    def hrp(self, linkage_method: str = 'ward', shrinkage: bool = True) -> pd.Series:
        """Hierarchical Risk Parity (López de Prado)."""
        from scipy.cluster.hierarchy import leaves_list, linkage
        from scipy.spatial.distance import squareform
        rets = self.rets
        if shrinkage:
            from sklearn.covariance import LedoitWolf
            cov = pd.DataFrame(LedoitWolf().fit(rets.values).covariance_,
                               index=rets.columns, columns=rets.columns)
        else:
            cov = rets.cov()
        dist = np.sqrt(0.5 * (1 - rets.corr().clip(-1, 1)))
        order = list(dist.index[leaves_list(
            linkage(squareform(dist.values, checks=False), method=linkage_method))])

        def cluster_var(items):
            sub = cov.loc[items, items].values
            ivp = 1.0 / np.diag(sub)
            ivp = ivp / ivp.sum()
            return float(ivp @ sub @ ivp)

        w = pd.Series(1.0, index=order)
        clusters = [order]
        while clusters:
            c = clusters.pop(0)
            if len(c) <= 1:
                continue
            c1, c2 = c[:len(c) // 2], c[len(c) // 2:]
            v1, v2 = cluster_var(c1), cluster_var(c2)
            alpha = 0.5 if v1 + v2 <= 0 else 1.0 - v1 / (v1 + v2)
            w[c1] *= alpha
            w[c2] *= 1.0 - alpha
            clusters += [c1, c2]
        return self._set_weights((w / w.sum()).reindex(self.tickers).values)

    # ------------------------------------------------------- Black-Litterman
    def black_litterman(
        self,
        views: Optional[Views] = None,
        P: Optional[Union[pd.DataFrame, np.ndarray]] = None,
        Q: Optional[Union[pd.Series, np.ndarray]] = None,
        tau: float = 0.05,
        omega=None,
        confidences: Optional[Sequence[float]] = None,
        w_mkt: Union[pd.Series, np.ndarray, None] = None,
        delta: float = 2.5,
        prior: Union[str, pd.Series] = 'equilibrium',
    ) -> Tuple[pd.Series, pd.DataFrame]:
        """Combina un prior con tus opiniones; guarda y devuelve (mu_BL, Sigma_BL).

        Las opiniones se dan con `views` (ver views_from_dict) o con las
        matrices P y Q. prior: 'equilibrium' (delta * Sigma * w_mkt), 'hist',
        o una Series con los retornos anuales previos. El resultado se pasa a
        markowitz(mu=..., sigma=...) o merton(..., mu=..., sigma=...), o usa
        optimize_bl().
        """
        if self.sigma is None:
            self.estimate_cov()
        if views is not None:
            if P is not None or Q is not None:
                raise ValueError('Usa views o (P, Q), no ambos')
            P, Q = views_from_dict(views, self.tickers)
        if P is None or Q is None:
            raise ValueError('Indica las opiniones con views o con P y Q')
        if isinstance(prior, pd.Series):
            pi = prior.reindex(self.tickers)
        elif prior == 'equilibrium':
            pi = implied_returns(self.sigma, w_mkt, delta)
        elif prior == 'hist':
            pi = self.rets.mean() * self.ppy
        else:
            raise ValueError("prior debe ser 'equilibrium', 'hist' o una Series")
        self.pi = pi
        self.mu_bl, self.sigma_bl = black_litterman(self.sigma, pi, P, Q, tau, omega, confidences)
        return self.mu_bl, self.sigma_bl

    def optimize_bl(self, method: str = 'markowitz', **kw) -> pd.Series:
        """Optimiza con los parámetros posteriores de Black-Litterman.

        method: 'markowitz' (kw: objective, short, rf, target_return) o
        'merton' (kw: gamma, rf).
        """
        if not hasattr(self, 'mu_bl'):
            raise ValueError('Ejecuta black_litterman() primero')
        if method == 'markowitz':
            return self.markowitz(mu=self.mu_bl, sigma=self.sigma_bl, **kw)
        if method == 'merton':
            return self.merton(mu=self.mu_bl, sigma=self.sigma_bl, **kw)
        raise ValueError("method debe ser 'markowitz' o 'merton'")

    # ---------------------------------------------------------------- métricas
    def metrics(self, weights: Optional[pd.Series] = None, rf: float = 0.01):
        """Retorno, volatilidad, Sharpe y CV anualizados de unos pesos (por defecto `weights`)."""
        w = self.weights if weights is None else weights
        return annualize_stats(self.rets, np.asarray(w, dtype=float), rf=rf,
                               periods_per_year=self.ppy)
