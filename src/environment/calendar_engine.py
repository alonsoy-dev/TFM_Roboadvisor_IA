from __future__ import annotations
import pandas as pd
from typing import Iterable


def build_master_calendar(price_index_like: Iterable[pd.Timestamp]) -> pd.DatetimeIndex:
    """
    Crea el calendario maestro de negociación a partir de un índice de fechas
    (por ejemplo, df.index de precios diarios).
    - Ordenado
    - Sin duplicados
    - Tipo DatetimeIndex (UTC-naive)
    """
    idx = pd.DatetimeIndex(pd.to_datetime(list(price_index_like))).unique().sort_values()
    return idx


def is_first_business_day_of_month(date: pd.Timestamp, trading_days: pd.DatetimeIndex) -> bool:
    """
    Devuelve True si 'date' es el primer día hábil del mes dentro de 'trading_days'.
    """
    # Filtra los días del mismo mes/año y toma el primero
    month_days = trading_days[(trading_days.year == date.year) & (trading_days.month == date.month)]
    return len(month_days) > 0 and date == month_days[0]


def is_last_business_day_of_month(date: pd.Timestamp, trading_days: pd.DatetimeIndex) -> bool:
    """
    Devuelve True si 'date' es el último día hábil del mes dentro de 'trading_days'.
    """
    month_days = trading_days[(trading_days.year == date.year) & (trading_days.month == date.month)]
    return len(month_days) > 0 and date == month_days[-1]


def monthly_contribution_mask(trading_days: pd.DatetimeIndex) -> pd.Series:
    """
    Serie booleana indexada por 'trading_days' con True en cada primer día hábil de mes.
    """
    return pd.Series([is_first_business_day_of_month(d, trading_days) for d in trading_days],
                     index=trading_days, name="is_contribution_day")


def month_end_mask(trading_days: pd.DatetimeIndex) -> pd.Series:
    """
    Serie booleana indexada por 'trading_days' con True en cada último día hábil de mes.
    """
    return pd.Series([is_last_business_day_of_month(d, trading_days) for d in trading_days],
                     index=trading_days, name="is_month_end")
