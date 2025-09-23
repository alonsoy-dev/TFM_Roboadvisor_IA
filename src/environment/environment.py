# =========================
# environment.py
# =========================
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Dict, Any
import pandas as pd

from calendar_engine import (
    build_master_calendar,
    monthly_contribution_mask,
    month_end_mask,
    is_first_business_day_of_month,
    is_last_business_day_of_month,
)
from data_loader import trading_days_from_prices


@dataclass
class InvestorProfile:
    riesgo: int                     # 1..7
    horizonte_anios: int            # 1..15
    aportacion_mensual_usd: float   # 0 si no hay aportación
    liquidez_inicial_usd: float     # efectivo inicial


class PortfolioEnv:
    """
    Entorno base:
    - Avanza por días hábiles derivados del índice de 'prices'.
    - Aplica aportación el primer día hábil de cada mes.
    - Marca fin de mes (para informe PDF futuro).
    - Aún sin trading.
    """

    def __init__(self,
                 prices: pd.DataFrame,
                 profile: InvestorProfile):
        if not isinstance(prices.index, pd.DatetimeIndex):
            raise ValueError("Se esperaba prices.index como DatetimeIndex.")
        self.prices = prices.copy()
        self.profile = profile

        # Calendario maestro desde las fechas reales de precios
        trading_days = trading_days_from_prices(self.prices)
        self.calendar = build_master_calendar(trading_days)

        # Máscaras
        self._is_contribution_day = monthly_contribution_mask(self.calendar)
        self._is_month_end = month_end_mask(self.calendar)

        # Estado
        self._t_idx: int = 0
        self.today: Optional[pd.Timestamp] = None

        # Finanzas (solo efectivo por ahora)
        self.cash_usd: float = 0.0
        self.portfolio_value_usd: float = 0.0
        self.history: list[Dict[str, Any]] = []

    def reset(self,
              start_date: Optional[pd.Timestamp] = None,
              end_date: Optional[pd.Timestamp] = None) -> Dict[str, Any]:
        # Acotar calendario al rango
        cal = self.calendar
        if start_date is not None:
            cal = cal[cal >= pd.to_datetime(start_date)]
        if end_date is not None:
            cal = cal[cal <= pd.to_datetime(end_date)]
        if len(cal) == 0:
            raise ValueError("No hay días hábiles en el rango elegido.")

        self.calendar = cal
        self._is_contribution_day = monthly_contribution_mask(self.calendar)
        self._is_month_end = month_end_mask(self.calendar)

        self._t_idx = 0
        self.today = self.calendar[self._t_idx]
        self.cash_usd = float(self.profile.liquidez_inicial_usd)
        self.portfolio_value_usd = self.cash_usd
        self.history.clear()

        # Aportación si toca
        contrib = 0.0
        if self._is_contribution_day.loc[self.today] and self.profile.aportacion_mensual_usd > 0:
            contrib = float(self.profile.aportacion_mensual_usd)
            self.cash_usd += contrib
            self.portfolio_value_usd = self.cash_usd

        obs = self._make_observation(contribution_applied=contrib)
        self._log_day(contribution_applied=contrib)
        return obs

    def step(self) -> Dict[str, Any]:
        if self._t_idx >= len(self.calendar) - 1:
            return self._make_observation(done=True, contribution_applied=0.0)

        self._t_idx += 1
        self.today = self.calendar[self._t_idx]

        contribution_applied = 0.0
        if self._is_contribution_day.loc[self.today] and self.profile.aportacion_mensual_usd > 0:
            contribution_applied = float(self.profile.aportacion_mensual_usd)
            self.cash_usd += contribution_applied

        self.portfolio_value_usd = self.cash_usd  # sin activos todavía
        obs = self._make_observation(contribution_applied=contribution_applied)
        self._log_day(contribution_applied=contribution_applied)
        return obs

    # ---------- helpers ----------
    def _make_observation(self,
                          done: bool = False,
                          contribution_applied: float = 0.0) -> Dict[str, Any]:
        today = self.today
        return {
            "fecha": today,
            "es_primer_dia_habil_mes": bool(is_first_business_day_of_month(today, self.calendar)),
            "es_ultimo_dia_habil_mes": bool(is_last_business_day_of_month(today, self.calendar)),
            "aportacion_aplicada_usd": contribution_applied,
            "cash_usd": self.cash_usd,
            "valor_cartera_usd": self.portfolio_value_usd,
            "done": done,
        }

    def _log_day(self, contribution_applied: float = 0.0) -> None:
        self.history.append({
            "fecha": self.today,
            "aportacion_aplicada_usd": contribution_applied,
            "cash_usd": self.cash_usd,
            "valor_cartera_usd": self.portfolio_value_usd,
            "es_primer_dia_habil_mes": self._is_contribution_day.loc[self.today],
            "es_ultimo_dia_habil_mes": self._is_month_end.loc[self.today],
        })
