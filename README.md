# portfolio-labkit

Descarga precios de Yahoo Finance y, con pocas líneas, grafica activos, hace pruebas ACF/PACF, descompone la serie, pronostica con SARIMA (con grid search en paralelo) y optimiza portafolios con Markowitz, Merton, HRP y Black–Litterman.

## Instalación

```bash
git clone https://github.com/lMerchands/portfolio-labkit.git
cd portfolio-labkit
python3 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -e .
```

`pip install -e .` instala el paquete y sus dependencias (`numpy`, `pandas`, `yfinance`, `matplotlib`, `seaborn`, `statsmodels`, `scipy`, `scikit-learn`, `joblib`, `pyportfolioopt`). Para correr las pruebas: `pip install -e ".[dev]"` y `pytest`.

> Si tu carpeta está en un directorio sincronizado con iCloud (Documentos en macOS), crea el entorno virtual fuera de él (por ejemplo `~/.venvs/portfolio-labkit`): macOS puede vaciar archivos del `.venv` y Python se queda esperando a que se descarguen.

Como el paquete está instalado, se puede importar desde cualquier carpeta o cuaderno de ese entorno.

## Guía rápida de inicio a fin

```python
from portfolio_labkit import YahooData

datos = YahooData(['AAPL', 'KO', 'MU'], start='2023-01-01', end='2025-12-31')
datos.data.tail()                 # precios de cierre ajustados (una columna por ticker)
datos.compute_returns('log')      # retornos 'log' o 'simple'
```

| Parámetro | Descripción | Por defecto |
|-----------|-------------|-------------|
| `tickers` | Lista de símbolos de Yahoo Finance (`'AAPL'`, `'^GSPC'`, `'^IRX'`) | obligatorio |
| `start` | Fecha inicial `'AAAA-MM-DD'` | `'2021-01-01'` |
| `end` | Fecha final; `None` = hasta hoy | `None` |

## 1. Gráficas

```python
datos.plot()                                     # todos en el mismo lienzo
datos.plot('AAPL')                               # un solo activo
datos.plot(['AAPL', 'MU'], kind='normalized')    # base 100: compara activos de distinto precio
datos.plot(kind='returns', single=True)          # un panel por activo
```

`kind`: `'price'` (por defecto), `'returns'` o `'normalized'`. Devuelve el `Axes` de matplotlib para seguir personalizando (`ax=` permite dibujar sobre uno existente).

## 2. ACF, PACF y estacionariedad

```python
res = datos.acf_pacf('AAPL', kind='log', nlags=40)
print(res)              # resumen: ADF, Ljung-Box y rezagos significativos
res.table               # ACF/PACF por rezago con intervalos de confianza
res.figure              # gráficas ACF y PACF
```

`kind` indica si se analizan los retornos (`'log'`, `'simple'`) o el precio (`'price'`; no estacionario, la ACF decae lento). Interpretación habitual: PACF que se corta en *p* y ACF que decae sugiere AR(*p*); al revés, MA(*q*).

## 3. Descomposición: tendencia, estacionalidad y residuo

```python
dec = datos.decompose('AAPL', period=21)         # STL (robusto)
print(dec)                                       # fuerza de tendencia y de estacionalidad
dec.components                                   # observed, trend, seasonal, resid
dec.figure
```

Opciones: `method='stl'|'classical'`, `model='additive'|'multiplicative'`. Si no indicas `period` se infiere del índice (mensual 12, trimestral 4, semanal 52; diario 252 con más de dos años de datos y 5 en caso contrario). Los precios diarios no tienen estacionalidad fija: con series cortas es más informativo remuestrear a mensual.

## 4. Pronóstico SARIMA

```python
fc = datos.forecast('AAPL', steps=30, order=(1, 1, 1))              # SARIMA(1,1,1) sobre el precio
fc = datos.forecast('AAPL', steps=12, order=(1, 0, 1),
                    seasonal_order=(1, 0, 0, 12), kind='log')       # sobre retornos, con estacionalidad
```

`fc` es un DataFrame con `forecast`, `lower` y `upper` (intervalo al 95%). Para más control:

```python
fit = datos.fit_sarima('AAPL', order=(2, 1, 1))
fit.summary(); fit.aic; fit.bic
fit.plot_forecast(steps=30, alpha=0.1)
```

### Grid search

```python
gs = datos.grid_search('AAPL', kind='log', p=3, d=[0], q=3, criterion='bic')
print(gs)                       # los 10 mejores modelos
gs.table                        # todos, ordenados; incluye errores de convergencia
fit = gs.fit_best(datos.series('AAPL', 'log'))
```

- `p, d, q, P, D, Q`: un entero *n* prueba 0…*n*; una lista prueba esos valores. `s` es el período estacional (0 = sin estacionalidad).
- `criterion`: `'aic'`, `'bic'` o `'oos'` (RMSE sobre el último tramo, `train_frac=0.8`).
- Los candidatos se ajustan **en paralelo** en todos los núcleos (`n_jobs=-1`). Se usa statsmodels (máxima verosimilitud con filtro de Kalman), que no se acelera con GPU/PyTorch; la paralelización por CPU es lo que reduce el tiempo.

## 5. Portafolios: Markowitz, Merton y HRP

```python
p = datos.portfolio()               # también: Portfolio(['AAPL','KO','MU'], start='2023-01-01')
p.estimate_mean('hist')             # o 'capm' ; mu anual
p.estimate_cov('hist')              # o 'ledoit' ; Sigma anual

w = p.markowitz('max_sharpe', rf=0.04)                       # pd.Series de pesos
w = p.markowitz('min_vol', short=True)                       # permite cortos
w = p.markowitz('efficient_return', target_return=0.25)
p.merton(gamma=3, rf=0.04)                                   # (1/γ) Σ⁻¹ (μ − rf)
p.merton([2, 4, 8], rf=0.04)                                 # barrido de γ, con columna w_rf
p.hrp()                                                      # Hierarchical Risk Parity

p.metrics(w, rf=0.04)               # retorno, volatilidad, Sharpe, CV
p.plot_frontier()                   # frontera eficiente + portafolios aleatorios
p.pick_candidates(5000)             # candidatos por Monte Carlo
```

`mu` y `Sigma` son **anuales** (252 períodos por año; `periods_per_year` cambia el valor).

## 6. Black–Litterman

Las opiniones se escriben en un diccionario o directamente con las matrices `P` (k×n) y `Q` (k):

```python
p.estimate_cov()
mu_bl, sigma_bl = p.black_litterman(
    views={'AAPL': 0.20,              # absoluta: AAPL rinde 20% anual
           ('KO', 'MU'): 0.05},       # relativa: KO supera a MU en 5%
    confidences=[0.6, 0.4],           # confianza de 0 a 1 en cada opinión
    tau=0.05,
)
w = p.optimize_bl('markowitz', objective='max_sharpe', rf=0.04)
w = p.optimize_bl('merton', gamma=3, rf=0.04)
```

- **Prior** (`prior=`): `'equilibrium'` (por defecto, π = δΣw_mkt con `w_mkt` y `delta` opcionales; pesos iguales si no das `w_mkt`), `'hist'` o una `Series` propia de retornos anuales.
- **Matrices explícitas:** `p.black_litterman(P=P, Q=Q, omega=Omega)`. Sin `omega`, se usa la regla de He–Litterman, `diag(P τΣ Pᵀ)`, escalada por `confidences` (0.5 equivale a He–Litterman).
- Combinaciones arbitrarias: `views=[({'AAPL': 1, 'KO': -0.5, 'MU': -0.5}, 0.02)]`.
- Funciones sueltas: `portfolio_labkit.portfolio.black_litterman`, `implied_returns`, `views_from_dict`.

La implementación se verifica contra `pypfopt.BlackLittermanModel` en las pruebas.

## 7. Métricas del portafolio y riesgo (VaR/CVaR histórico)

Con los pesos de cualquier método (`markowitz`, `merton`, `hrp`, `optimize_bl`) guardados en `p.weights`, o con pesos propios (`weights=[0.4, 0.4, 0.2]`):

```python
p.markowitz('max_sharpe', rf=0.04)
p.report(rf=0.04)                                   # retorno, vol, CV, Sharpe, VaR y CVaR al 95% y 99%
p.report(rf=0.04, horizon=10, value=1_000_000)      # a 10 días y en dinero
p.var(0.05); p.cvar(0.05)                           # un solo nivel
p.portfolio_returns(kind='log')                     # retornos diarios del portafolio
p.portfolio_value()                                 # valor acumulado, base 100
```

- **Retorno, volatilidad, CV, Sharpe:** anualizados (`periods_per_year`), con `rf` anual.
- **VaR histórico:** pérdida en el cuantil `alpha` de los retornos diarios del portafolio (rebalanceo diario a pesos fijos). Se reporta como número positivo: 0.019 significa perder 1.9%. Acepta `alpha=0.05` o `0.95`.
- **CVaR (expected shortfall):** pérdida promedio de los días que superan el VaR; siempre ≥ VaR.
- **Horizonte** `horizon_method='sqrt'` escala el VaR de 1 día por √h; `'rolling'` usa sumas móviles de h retornos (sin supuesto de distribución, pero con ventanas solapadas y necesita ≥ 100).
- Funciones sueltas sobre cualquier serie de retornos: `from portfolio_labkit.risk import historical_var, historical_cvar, risk_table`.

> El VaR/CVaR simulado con t de Student y movimiento browniano (colas más pesadas que la normal) queda como siguiente etapa.

## 8. Pronóstico del portafolio completo

```python
fc = p.forecast(30, order=(1, 1, 1))                       # valor del portafolio (base 100) a 30 días
fc = p.forecast(20, order=(1, 0, 1), kind='log')           # retornos logarítmicos diarios
gs = p.grid_search(p=2, d=[0], q=2, criterion='bic')       # mejor SARIMA para el portafolio
fit = p.fit_sarima(order=(2, 1, 1)); fit.summary()
```

Usa los mismos modelos y el mismo grid search paralelo de la sección 4, aplicados a la serie del portafolio con los pesos actuales. Con `kind='price'` los intervalos son del valor del portafolio; con `kind='log'` son de cada día y no se acumulan.

## Estructura

```
portfolio_labkit/
  yahoo_data.py    YahooData: descarga, retornos y atajos
  plots.py         gráficas seaborn
  timeseries.py    ACF/PACF, Ljung-Box, ADF, descomposición
  forecasting.py   SARIMA y grid search paralelo
  portfolio.py     Portfolio, Black–Litterman, métricas y forecast del portafolio
  risk.py          VaR y CVaR históricos
  metrics.py       métricas anualizadas
tests/             pruebas con datos sintéticos (sin red)
```

## Créditos

El código original (`yahoo_data.py`, `Methodologies_Port.py` y `metrics_portfolio.py`) fue escrito por **camilocastillo** ([CamiloCastillo88/Portfolio](https://github.com/CamiloCastillo88/Portfolio), agosto de 2025), como indican los encabezados de los archivos. `YahooData`, `Portfolio` (antes `Do_portfolio`) y `metrics` parten de esa base; el historial de git muestra cada cambio.
