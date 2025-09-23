from __future__ import annotations
import pandas as pd
from typing import Optional, Iterable


class TimeSeriesFormatError(Exception):
    pass


def _auto_date_col(cols: Iterable[str]) -> Optional[str]:
    candidates = [c for c in cols if c.strip().lower() in {"date"}]
    return candidates[0] if candidates else None


def load_time_series(path_csv: str, date_col: Optional[str] = None) -> pd.DataFrame:
    """
    Lee el CSV de series temporales con precios diarios por ETF.
    Estructura esperada:
        date, ETF1, ETF2, ...
    - 'date' se convierte en índice datetime.
    - Columnas ETF -> float (no numéricos a NaN).
    - Fechas únicas, ordenadas asc.
    - Sin filas totalmente vacías.
    Devuelve: DataFrame 'prices' con índice DatetimeIndex y columnas = ETFs.
    """
    df = pd.read_csv(path_csv)

    # 1) Detectar columna de fecha
    date_col = date_col or _auto_date_col(df.columns)

    # 2) Parseo de fechas e índice
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce", utc=False)
    df = df.dropna(subset=[date_col])
    df = df.set_index(date_col)
    df.index.name = "date"

    # 3) Orden cronológico y unicidad
    df = df.sort_index()
    df = df[~df.index.duplicated(keep="first")]

    # 4) Convertir columnas a numéricas (coerce -> NaN si hay strings)
    value_cols = [c for c in df.columns if c is not None]
    for c in value_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # 5) Eliminar filas totalmente vacías
    df = df.dropna(how="all")

    # 6) Validaciones suaves
    if df.shape[1] == 0:
        raise TimeSeriesFormatError("No hay columnas de precios tras limpiar el CSV.")
    return df


def trading_days_from_prices(prices: pd.DataFrame) -> pd.DatetimeIndex:
    """
    Devuelve el índice de fechas (ordenado, único) como calendario maestro.
    """
    idx = pd.DatetimeIndex(prices.index).unique().sort_values()
    return idx
