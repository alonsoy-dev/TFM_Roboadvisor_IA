from __future__ import annotations
import pandas as pd
from typing import Tuple

class TimeSeriesFormatError(Exception):
    pass

#--- Time-series ---
def load_time_series_csv(path_csv: str) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
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

    # Pivots
    prices_df = (df.pivot(index="date", columns="ticker", values="nav").sort_index())
    aum_df = (df.pivot(index="date", columns="ticker", values="aum").sort_index())

    # Limpieza
    prices_df = prices_df.dropna(how="all")
    aum_df = aum_df.reindex(prices_df.index)  # alinear por fechas de precios

    # Meta por ticker
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
    """
    Días hábiles presentes en prices (ordenados, sin duplicados).
    """
    return pd.DatetimeIndex(prices.index).unique().sort_values()

#--- Exposiciones (country/sector) ---

_REQ_COUNTRY = ["ticker", "type", "country", "weight"]
_REQ_SECTOR  = ["ticker", "type", "sector",  "weight"]

def _ensure_cols(df: pd.DataFrame, required: list[str], fname: str):
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"{fname}: faltan columnas {missing}")

def _to_num_weights(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["weight"] = pd.to_numeric(df["weight"], errors="coerce").fillna(0.0)
    df["weight"] = df["weight"].clip(lower=0.0)
    return df

def _aggregate_normalize(df: pd.DataFrame, key_col: str) -> pd.DataFrame:
    # Suma por (ticker, key) y normaliza a 1 por ticker.
    g = df.groupby(["ticker", key_col], as_index=False)["weight"].sum()
    totals = g.groupby("ticker")["weight"].sum().rename("total")
    out = g.merge(totals, on="ticker", how="left")
    out["weight"] = out.apply(lambda r: (r["weight"] / r["total"]) if r["total"] > 0 else 0.0, axis=1)
    return out[["ticker", key_col, "weight"]].sort_values(["ticker", key_col]).reset_index(drop=True)

def _first_type_map(df: pd.DataFrame) -> dict[str, str]:
    # Si un ticker tiene varios 'type', tomamos el primero
    return df.groupby("ticker")["type"].first().to_dict()

def load_exposures_country(path_csv: str, tickers: set[str] | None = None):
    """
    Lee exposures_country.csv (cols: ticker,type,country,weight)
    Devuelve:
      country_by_ticker: dict[ticker] -> DataFrame(['country','weight'])
      type_by_ticker:    dict[ticker] -> str
    """
    df = pd.read_csv(path_csv)
    _ensure_cols(df, _REQ_COUNTRY, "exposures_country.csv")
    if tickers is not None:
        df = df[df["ticker"].astype(str).isin(tickers)]

    df = _to_num_weights(df)
    norm = _aggregate_normalize(df, key_col="country")
    type_map = _first_type_map(df)

    by_ticker: dict[str, pd.DataFrame] = {}
    for tkr, sub in norm.groupby("ticker"):
        by_ticker[tkr] = sub[["country", "weight"]].reset_index(drop=True)
    return by_ticker, type_map

def load_exposures_sector(path_csv: str, tickers: set[str] | None = None):
    """
    Lee exposures_sector.csv (cols: ticker,type,sector,weight)
    Devuelve:
      sector_by_ticker: dict[ticker] -> DataFrame(['sector','weight'])
      type_by_ticker:   dict[ticker] -> str
    """
    df = pd.read_csv(path_csv)
    _ensure_cols(df, _REQ_SECTOR, "exposures_sector.csv")
    if tickers is not None:
        df = df[df["ticker"].astype(str).isin(tickers)]

    df = _to_num_weights(df)
    norm = _aggregate_normalize(df, key_col="sector")
    type_map = _first_type_map(df)

    by_ticker: dict[str, pd.DataFrame] = {}
    for tkr, sub in norm.groupby("ticker"):
        by_ticker[tkr] = sub[["sector", "weight"]].reset_index(drop=True)
    return by_ticker, type_map

def load_all_exposures(country_path: str, sector_path: str, prices: pd.DataFrame | None = None) -> dict:
    """
    Orquestación simple:
      - Si 'prices' se pasa, filtra por sus columnas (tickers válidos del mercado).
    Return:
      {
        'country_by_ticker': {ticker -> DF[country, weight]},
        'sector_by_ticker' : {ticker -> DF[sector,  weight]},
        'type_country'     : {ticker -> type},
        'type_sector'      : {ticker -> type},
      }
    """
    tickers = set(prices.columns) if prices is not None else None

    country_by_ticker, type_country = load_exposures_country(country_path, tickers=tickers)
    sector_by_ticker,  type_sector  = load_exposures_sector(sector_path,  tickers=tickers)

    return {
        "country_by_ticker": country_by_ticker,
        "sector_by_ticker":  sector_by_ticker,
        "type_country":      type_country,
        "type_sector":       type_sector,
    }

def get_country_exposure(country_by_ticker: dict[str, pd.DataFrame], ticker: str) -> pd.DataFrame:
    return country_by_ticker.get(ticker, pd.DataFrame(columns=["country", "weight"]))

def get_sector_exposure(sector_by_ticker: dict[str, pd.DataFrame], ticker: str) -> pd.DataFrame:
    return sector_by_ticker.get(ticker, pd.DataFrame(columns=["sector", "weight"]))