from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Dict, Any
import pandas as pd
import numpy as np
from reward_engine import RewardEngine, RewardConfig

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

    def __init__(self, prices, profile, aum=None, exposures=None, reward_engine=None):

        if not isinstance(prices.index, pd.DatetimeIndex):
            raise ValueError("Se esperaba prices.index como DatetimeIndex.")
        self.prices = prices.copy()
        self.profile = profile
        self.exposures = exposures or {}
        self.reward_engine = reward_engine or RewardEngine(RewardConfig())
        self.prev_portfolio_value_usd = None

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

        # AUM por ETF
        self._aum_ffill = aum.sort_index().ffill() if aum is not None else None

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

        # Estado previo
        self.prev_portfolio_value_usd: Optional[float] = None

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

        self.prev_portfolio_value_usd = self.portfolio_value_usd  # base para el primer step
        reward = 0.0
        obs = self._make_observation(contribution_applied=contrib)
        obs["reward"] = reward
        self._log_day(contribution_applied=contrib)
        return obs

    def step(self, action: Optional[Dict[str, float]] = None) -> Dict[str, Any]:
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

        # Ejecutar la acción a cierre (si viene)
        if action:
            self._exec_at_close(target_weights=action)

        # Valorizar cartera con precios del día
        self._recalc_portfolio_value()

        score = self.reward_engine.compute(
            prev_value=self.prev_portfolio_value_usd,
            curr_value=self.portfolio_value_usd,
            net_flow=contribution_applied,
            riesgo=self.profile.riesgo,
            horizonte_anios=self.profile.horizonte_anios,
            positions_shares=self.positions_shares,
            price_today_fn=self._price_today,
            exposures=self.exposures
        )
        self.prev_portfolio_value_usd = self.portfolio_value_usd
        obs = self._make_observation(contribution_applied=contribution_applied)
        obs["reward"] = score
        self._log_day(contribution_applied=contribution_applied)
        return obs

    # Ejecución a cierre con all invertido y comisiones por ETF
    def _exec_at_close(self, target_weights: Dict[str, float]) -> None:
        """
        Rebalancea a 'target_weights' usando precios de cierre de self.today.
        - Sin cortos ni deuda; los pesos se normalizan para que sumen 1 (cero cash final).
        - Comisión por movimiento por ETF según su AUM del día.
        - Ajuste previo para que, tras pagar comisiones, quede all invertido.
        """
        # Sanitizar: [0,∞) y normalizar a suma 1
        tw = {t: max(0.0, float(target_weights.get(t, 0.0))) for t in self.universe}
        total_w = sum(tw.values())
        if total_w <= 0.0:
            raise ValueError("target_weights inválido: la suma debe ser > 0 para invertir todo.")
        tw = {t: w / total_w for t, w in tw.items()}  # suma exacta = 1

        # Precios y valor actual
        px = {t: self._price_today(t) for t in self.universe}
        self._recalc_portfolio_value()
        total_before = float(self.portfolio_value_usd)
        if not np.isfinite(total_before) or total_before <= 0:
            return

        current_usd = {}
        for t in self.universe:
            sh = self.positions_shares.get(t, 0.0)
            p = px[t]
            current_usd[t] = (sh * p) if (pd.notna(p) and sh) else 0.0

        # Fee rate por ETF (según AUM del ETF)
        fr = {t: self._fee_rate_for_ticker(t) for t in self.universe}

        # Estimación de comisiones y ajuste "neto de fees"
        delta0 = {t: tw[t] * total_before - current_usd[t] for t in self.universe}
        fees0 = sum(fr[t] * abs(delta0[t]) for t in self.universe)
        avail0 = total_before - fees0

        # Target con capital neto de fees
        delta1 = {t: tw[t] * avail0 - current_usd[t] for t in self.universe}
        fees1 = sum(fr[t] * abs(delta1[t]) for t in self.universe)
        avail1 = total_before - fees1

        # Usar la versión ajustada
        delta = {t: tw[t] * avail1 - current_usd[t] for t in self.universe}
        fees = sum(fr[t] * abs(delta[t]) for t in self.universe)

        # Operacion de ventas primero (generan cash)
        sells = {t: -min(0.0, delta[t]) for t in self.universe}  # positivos = vender
        for t, nom in sells.items():
            if nom <= 0.0: continue
            p = px[t]
            if not pd.notna(p) or p <= 0: continue
            sh_to_sell = min(self.positions_shares.get(t, 0.0), nom / p)
            if sh_to_sell > 0:
                self.positions_shares[t] -= sh_to_sell
                self.cash_usd += sh_to_sell * p

        # 5) Pagar comisiones (ventas + compras)
        self.cash_usd -= fees
        if abs(self.cash_usd) < 1e-10:  # limpieza numérica
            self.cash_usd = 0.0

        # Compras con el cash restante
        buys = {t: max(0.0, delta[t]) for t in self.universe}
        budget = self.cash_usd
        sum_buys = sum(buys.values())
        if budget > 0 and sum_buys > 0:
            scale = budget / sum_buys
            for t, nom in buys.items():
                if nom <= 0.0: continue
                p = px[t]
                if not pd.notna(p) or p <= 0: continue
                alloc = nom * scale
                sh_to_buy = alloc / p
                if sh_to_buy > 0:
                    self.positions_shares[t] = self.positions_shares.get(t, 0.0) + sh_to_buy
                    self.cash_usd -= sh_to_buy * p

        # Dejar cash exactamente a 0 si es residuo minúsculo
        if abs(self.cash_usd) < 1e-8:
            self.cash_usd = 0.0

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

    def _aum_today(self, etf: str) -> Optional[float]:
        if self._aum_ffill is None:
            return None
        try:
            val = self._aum_ffill.at[self.today, etf]
        except KeyError:
            return None
        return float(val) if pd.notna(val) else None

    def _fee_rate_for_ticker(self, etf: str) -> float:
        """
        Tabla por AUM del ETF (USD)
        """
        a = self._aum_today(etf)
        if a < 100_000_000:
            return 0.003
        if a < 500_000_000:
            return 0.002
        if a < 2_000_000_000:
            return 0.001
        return 0.0005

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
