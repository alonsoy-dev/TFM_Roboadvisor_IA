from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Dict, Any
import pandas as pd
import numpy as np

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
    - Estructura de posiciones (participaciones por ETF).
    - Valoración diaria: cash + sum(posiciones * precio_cierre).
    - Si un ETF no tiene precio hoy, usamos el último precio conocido (ffill).
    - Sin ejecutar compras/ventas aun
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

        # Universo = columnas del DataFrame de precios
        self.universe: list[str] = [c for c in self.prices.columns if isinstance(c, str)]
        # Preparamos precios con ffill para valoración
        self._prices_ffill = self.prices.sort_index().ffill()

        # Estado
        self._t_idx: int = 0
        self.today: Optional[pd.Timestamp] = None

        # Finanzas (solo efectivo por ahora)
        self.cash_usd: float = 0.0
        # Posiciones en nº de participaciones por ETF
        self.positions_shares: Dict[str, float] = {etf: 0.0 for etf in self.universe}

        # Valores agregados
        self.portfolio_value_usd: float = 0.0
        self.history: list[Dict[str, Any]] = []

    # ---------- Ciclo de vida ----------
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

        # Estado financiero
        self.cash_usd = float(self.profile.liquidez_inicial_usd)
        self.positions_shares = {etf: 0.0 for etf in self.universe}  # sin posiciones al inicio

        # Aportación si toca (primer hábil del mes)
        contrib = 0.0
        if self._is_contribution_day.loc[self.today] and self.profile.aportacion_mensual_usd > 0:
            contrib = float(self.profile.aportacion_mensual_usd)
            self.cash_usd += contrib

        # Recalcular valor de cartera
        self._recalc_portfolio_value()

        obs = self._make_observation(contribution_applied=contrib)
        self._log_day(contribution_applied=contrib)
        return obs

    def step(self) -> Dict[str, Any]:
        # Fin de calendario
        if self._t_idx >= len(self.calendar) - 1:
            return self._make_observation(done=True, contribution_applied=0.0)

        # Avanza día
        self._t_idx += 1
        self.today = self.calendar[self._t_idx]

        # Aportación mensual si procede
        contribution_applied = 0.0
        if self._is_contribution_day.loc[self.today] and self.profile.aportacion_mensual_usd > 0:
            contribution_applied = float(self.profile.aportacion_mensual_usd)
            self.cash_usd += contribution_applied

        # Valorizar cartera con precios del día
        self._recalc_portfolio_value()

        obs = self._make_observation(contribution_applied=contribution_applied)
        self._log_day(contribution_applied=contribution_applied)
        return obs

    # ---------- Valoración ----------
    def _price_today(self, etf: str) -> float:
        """
        Devuelve el precio de cierre para valoración en self.today.
        - Si falta el dato exactamente hoy, se usa el último conocido (ffill).
        """
        d = self.today
        try:
            px = self._prices_ffill.at[d, etf]
        except KeyError:
            # Si la fecha no existe en prices (no debería pasar porque calendar viene de prices),
            # devolvemos NaN
            return np.nan
        return float(px) if pd.notna(px) else np.nan

    def _recalc_portfolio_value(self) -> None:
        """
        Recalcula el valor de la cartera = efectivo + sum(posiciones * precio_usable_hoy).
        Si algún ETF no tiene precio usable, su contribución hoy es 0.
        """
        mv = 0.0
        for etf, shares in self.positions_shares.items():
            if shares == 0.0:
                continue
            px = self._price_today(etf)
            if pd.notna(px):
                mv += shares * px
            # Si px es NaN, aportación 0 hoy (no contamos ese ETF hoy)
        self.portfolio_value_usd = self.cash_usd + mv

    # ---------- helpers ----------
    def _make_observation(self,
                          done: bool = False,
                          contribution_applied: float = 0.0) -> Dict[str, Any]:
        today = self.today
        # Snapshot simple de posiciones y valoración
        positions_copy = {k: float(v) for k, v in self.positions_shares.items() if v != 0.0}

        return {
            "fecha": today,
            "es_primer_dia_habil_mes": bool(is_first_business_day_of_month(today, self.calendar)),
            "es_ultimo_dia_habil_mes": bool(is_last_business_day_of_month(today, self.calendar)),
            "aportacion_aplicada": contribution_applied,
            "cash": self.cash_usd,
            "valor_cartera": self.portfolio_value_usd,
            "posiciones": positions_copy,
            "done": done,
        }

    def _log_day(self, contribution_applied: float = 0.0) -> None:
        self.history.append({
            "fecha": self.today,
            "aportacion_aplicada": contribution_applied,
            "cash": self.cash_usd,
            "valor_cartera": self.portfolio_value_usd,
            "es_primer_dia_habil_mes": self._is_contribution_day.loc[self.today],
            "es_ultimo_dia_habil_mes": self._is_month_end.loc[self.today],
            "posiciones": {k: float(v) for k, v in self.positions_shares.items() if v != 0.0},
        })
