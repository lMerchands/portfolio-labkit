# yahoo-data

Clase `YahooData` para descargar precios de cierre ajustados desde Yahoo Finance (`yfinance`) y calcular retornos simples o logarítmicos.

## Guía de uso paso a paso

### 1. Requisitos

- Python 3.8 o superior
- Conexión a internet (los datos se descargan de Yahoo Finance)

### 2. Clonar el repositorio

```bash
git clone https://github.com/lMerchands/yahoo-data.git
cd yahoo-data
```

### 3. Crear un entorno virtual (opcional, recomendado)

```bash
python3 -m venv .venv
source .venv/bin/activate
```

En Windows: `.venv\Scripts\activate`.

### 4. Instalar las dependencias

```bash
pip install numpy pandas yfinance
```

### 5. Importar la clase

Desde un script o cuaderno ubicado en la misma carpeta que `yahoo_data.py` (el módulo se llama `yahoo_data`, la clase `YahooData`):

```python
from yahoo_data import YahooData
```

Si trabajas desde otra carpeta, copia `yahoo_data.py` junto a tu script o agrega la carpeta del repositorio al `PYTHONPATH`.

### 6. Descargar los precios

Crear el objeto descarga los precios de cierre ajustados (`Adj Close`):

```python
datos = YahooData(['AAPL', 'KO', 'MU'], start='2025-01-01', end='2025-12-31')
```

| Parámetro | Descripción | Por defecto |
|-----------|-------------|-------------|
| `tickers` | Lista de símbolos de Yahoo Finance (por ejemplo `'AAPL'`, `'^GSPC'`, `'^IRX'`) | obligatorio |
| `start` | Fecha inicial, formato `'AAAA-MM-DD'` | `'2021-01-01'` |
| `end` | Fecha final, formato `'AAAA-MM-DD'`. Con `None` llega hasta hoy | `None` |

### 7. Consultar los precios

Los precios quedan en `datos.data`, un `DataFrame` con las fechas como índice y un ticker por columna:

```python
datos.data.head()
datos.data.tail()
```

### 8. Calcular los retornos

```python
rets_log = datos.compute_returns()              # logarítmicos (por defecto)
rets_simples = datos.compute_returns('simple')  # simples
```

El método devuelve un `DataFrame` de retornos y también lo guarda en `datos.rets`. Cualquier otro valor de `kind` lanza un `ValueError`. Cada llamada reemplaza lo que había en `datos.rets`.

### 9. Usar los resultados

Como todo es un `DataFrame` de pandas, puedes seguir con el análisis habitual:

```python
rets_log.mean() * 252        # retorno anualizado
rets_log.cov() * 252         # matriz de covarianzas anualizada
datos.data.reset_index()     # fecha como columna, útil para exportar o graficar
```

### Ejemplo completo

```python
from yahoo_data import YahooData

datos = YahooData(['AAPL', 'KO', 'MU'], start='2025-01-01', end='2025-12-31')
print(datos.data.tail())

rets = datos.compute_returns('log')
print(rets.describe())
```

## Créditos

El código original (`yahoo_data.py`, commit inicial de este repositorio) fue escrito por **camilocastillo** (agosto de 2025), como indica el encabezado del archivo. Las mejoras posteriores se hacen a partir de esa versión y se pueden ver en el historial de git.
