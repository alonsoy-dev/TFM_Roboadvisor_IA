from __future__ import annotations
import pandas as pd
from typing import Tuple

class TimeSeriesFormatError(Exception):
    pass


def load_market_from_long_csv(path_csv: str) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Espera columnas: date, ticker, name, type, nav, shares_outstanding, aum
    Devuelve:
      prices_df : index=date, columns=ticker, values=nav
      aum_df    : index=date, columns=ticker, values=aum
      meta_df   : por ticker -> columns=['name','type']
    """
    df = pd.read_csv(path_csv)

    required = {"date", "ticker", "nav", "aum"}
    missing = required - set(df.columns)
    if missing:
        raise TimeSeriesFormatError(f"Faltan columnas requeridas: {missing}")

    # Normaliza tipos
    df["date"] = pd.to_datetime(df["date"], errors="coerce", utc=False)
    df = df.dropna(subset=["date", "ticker"])
    df["ticker"] = df["ticker"].astype(str).str.strip()

    # A numérico
    for col in ["nav", "aum"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Resolver duplicados por (date, ticker) -> nos quedamos con la última
    df = (df
          .sort_values(["date", "ticker"])
          .groupby(["date", "ticker"], as_index=False)
          .last())

    prices_df = (df.pivot(index="date", columns="ticker", values="nav").sort_index())
    aum_df = (df.pivot(index="date", columns="ticker", values="aum").sort_index())

    # Limpiezas
    prices_df = prices_df.dropna(how="all")
    aum_df = aum_df.reindex(prices_df.index)  # alinear por fechas de precios

    # ---- META por ticker (usar df completo, no subset sin 'date') ----
    meta_cols_exist = [c for c in ["name", "type"] if c in df.columns]
    if meta_cols_exist:
        meta_df = (df.sort_values(["ticker", "date"])
                     .groupby("ticker", as_index=True)
                     .last()[meta_cols_exist])
        # Asegura mismo orden/índice que las columnas de precios
        meta_df = meta_df.reindex(prices_df.columns)
    else:
        meta_df = pd.DataFrame(index=prices_df.columns)

    meta_df.index.name = "ticker"
    return prices_df, aum_df, meta_df


def trading_days_from_prices(prices: pd.DataFrame) -> pd.DatetimeIndex:
    return pd.DatetimeIndex(prices.index).unique().sort_values()
