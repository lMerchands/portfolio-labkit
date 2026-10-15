#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 13 18:28:03 2025

@author: camilocastillo
"""

from typing import List, Optional

import numpy as np
import pandas as pd
import yfinance as yf


class YahooData:
    """Descarga los precios de cierre ajustados para una lista de tickers."""

    def __init__(
        self,
        tickers: List[str],
        start: str = '2021-01-01',
        end: Optional[str] = None,
    ) -> None:
        self.tickers = tickers
        self.start = start
        self.end = end
        self.data = yf.download(
            self.tickers,
            start=start,
            end=end,
            progress=False,
            auto_adjust=False,
        )['Adj Close']
        if isinstance(self.data, pd.Series):
            self.data = self.data.to_frame()
        self.data.dropna(how='all', inplace=True)

    def compute_returns(self, kind: str = 'log') -> pd.DataFrame:
        """Calcula los retornos, simples ('simple') o logarítmicos ('log')."""
        if kind == 'log':
            self.rets = np.log(self.data / self.data.shift(1))
        elif kind == 'simple':
            self.rets = self.data.pct_change()
        else:
            raise ValueError("kind debe ser 'log' o 'simple'")
        self.rets.dropna(inplace=True)
        return self.rets

    # ------------------------------------------------------------------
    # Atajos: delegan en los módulos de análisis (se importan al usarlos,
    # así descargar datos no exige seaborn ni statsmodels).
    # ------------------------------------------------------------------
    def series(self, ticker: str, kind: str = 'price') -> pd.Series:
        """Serie de un activo: 'price' (precio), 'log' o 'simple' (retornos)."""
        if ticker not in self.data.columns:
            raise KeyError(f'{ticker!r} no está descargado. Disponibles: {list(self.data.columns)}')
        if kind == 'price':
            return self.data[ticker].dropna()
        return self.compute_returns(kind)[ticker]

    def plot(self, tickers=None, kind: str = 'price', single: bool = False, **kwargs):
        """Grafica uno o varios activos (ver plots.plot_prices)."""
        from .plots import plot_prices
        return plot_prices(self.data, tickers, kind=kind, single=single, **kwargs)

    def acf_pacf(self, ticker: str, kind: str = 'log', **kwargs):
        """ACF/PACF, Ljung-Box y ADF de un activo (ver timeseries.acf_pacf)."""
        from .timeseries import acf_pacf
        return acf_pacf(self.series(ticker, kind), **kwargs)

    def decompose(self, ticker: str, kind: str = 'price', period=None, **kwargs):
        """Tendencia, estacionalidad y residuo (ver timeseries.decompose)."""
        from .timeseries import decompose
        return decompose(self.series(ticker, kind), period=period, **kwargs)

    def fit_sarima(self, ticker: str, order=(1, 1, 1), seasonal_order=(0, 0, 0, 0),
                   kind: str = 'price', **kwargs):
        """Ajusta un SARIMA a un activo (ver forecasting.fit_sarima)."""
        from .forecasting import fit_sarima
        return fit_sarima(self.series(ticker, kind), order, seasonal_order, **kwargs)

    def forecast(self, ticker: str, steps: int = 30, order=(1, 1, 1),
                 seasonal_order=(0, 0, 0, 0), kind: str = 'price', plot: bool = True):
        """Ajusta un SARIMA y pronostica `steps` períodos; devuelve el DataFrame del pronóstico."""
        fit = self.fit_sarima(ticker, order, seasonal_order, kind)
        if plot:
            return fit.plot_forecast(steps)[0]
        return fit.forecast(steps)

    def grid_search(self, ticker: str, kind: str = 'price', **kwargs):
        """Búsqueda en malla del mejor SARIMA (ver forecasting.grid_search_sarima)."""
        from .forecasting import grid_search_sarima
        return grid_search_sarima(self.series(ticker, kind), **kwargs)

    def portfolio(self, **kwargs):
        """Crea un Portfolio con estos precios (ver portfolio.Portfolio)."""
        from .portfolio import Portfolio
        return Portfolio(self, **kwargs)
