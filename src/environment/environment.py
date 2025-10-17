from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Dict, Any
import pandas as pd
import numpy as np
from src.environment.reward_engine import RewardEngine, RewardConfig
from src.environment.observation_engine import ObservationEngine
from src.environment.calendar_engine import build_master_calendar, monthly_contribution_mask
from src.environment.data_loader import trading_days_from_prices
from src.environment.metrics_engine import MetricsEngine

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

    def __init__(self, prices, profile, aum=None, exposures=None, indicators=None, reward_engine=None):

        if not isinstance(prices.index, pd.DatetimeIndex):
            raise ValueError("Se esperaba prices.index como DatetimeIndex.")
        self.prices = prices.copy()
        self.profile = profile
        self.exposures = exposures
        self.indicators = indicators
        self.reward_engine = reward_engine or RewardEngine(RewardConfig())

        self.metrics_engine = MetricsEngine(short_window=30, long_window=252)

        # Calendario maestro desde las fechas reales de precios
        trading_days = trading_days_from_prices(self.prices)
        self.calendar = build_master_calendar(trading_days)

        # Máscaras
        self._is_contribution_day = monthly_contribution_mask(self.calendar)

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
        self._aportado_acum: float = 0.0

        # Motor de observación
        self.obs_engine = ObservationEngine(self)
        self.max_weight_per_fund: Optional[float] = None
        self._start_date_reset: Optional[pd.Timestamp] = None

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

        self._t_idx = 0
        self.today = self.calendar[self._t_idx]

        # Metrics: reset con el perfil de riesgo del episodio
        self.metrics_engine.reset(risk_profile=int(self.profile.riesgo))

        self.metrics_engine.reset(risk_profile=int(self.profile.riesgo))

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

        self._aportado_acum = float(self.cash_usd)

        reward = 0.0
        obs = self._make_observation(contribution_applied=contrib, reward=reward, done=False)
        self.prev_portfolio_value_usd = self.portfolio_value_usd
        vol0 = self._volatilidad_diaria_actual(lookback_days=120)
        self._log_day(contribution_applied=contrib, reward=0.0, r_dia=0.0, vol_diaria=vol0)
        return obs

    def step(self, action: Optional[Dict[str, float]] = None) -> Dict[str, Any]:
        # Fin de calendario
        if self._t_idx >= len(self.calendar) - 1:
            obs = self._make_observation(done=True, contribution_applied=0.0)
            obs['custom_metrics'] = {}
            return obs

        # Avanza día
        self._t_idx += 1
        self.today = self.calendar[self._t_idx]

        # Aportación mensual si procede
        contribution_applied = 0.0
        if self._is_contribution_day.loc[self.today] and self.profile.aportacion_mensual_usd > 0:
            contribution_applied = float(self.profile.aportacion_mensual_usd)
            self.cash_usd += contribution_applied
            self._aportado_acum += contribution_applied

        # Ejecutar la acción a cierre (si viene)
        if action:
            self._exec_at_close(target_weights=action)

        # Valorizar cartera con precios del día
        self._recalc_portfolio_value()

        # Volatilidad diaria basada en participaciones actuales
        vol_diaria = self._volatilidad_diaria_actual(lookback_days=120)

        # Rentabilidad diaria
        if self.prev_portfolio_value_usd and self.prev_portfolio_value_usd > 0:
            r_dia = ((self.portfolio_value_usd - contribution_applied) /
                     self.prev_portfolio_value_usd) - 1.0
        else:
            r_dia = 0.0

        score = self.reward_engine.compute(
            prev_value=self.prev_portfolio_value_usd,
            curr_value=self.portfolio_value_usd,
            net_flow=contribution_applied,
            riesgo=self.profile.riesgo,
            horizonte_anios=self.profile.horizonte_anios,
            positions_shares=self.positions_shares,
            price_today_fn=self._price_today,
            exposures=self.exposures,
            volatilidad_diaria=vol_diaria,
        )

        # Metrics: actualizar buffers y construir métricas para el paso
        self.metrics_engine.update(r_net=r_dia)
        metrics = self.metrics_engine.get_metrics()

        # LOG
        self._log_day(
            contribution_applied=contribution_applied,
            reward=score,
            r_dia=r_dia,
            vol_diaria=vol_diaria
        )

        # Actualizamos el prev para el próximo día
        self.prev_portfolio_value_usd = self.portfolio_value_usd

        # Delegar la creación de la observación
        obs = self._make_observation(
            contribution_applied=contribution_applied,
            reward=score,
            done=False,)
        # Adjuntar el diccionario de métricas a la salida del step
        obs['custom_metrics'] = metrics
        return obs

    def _log_day(self,
                 contribution_applied: float = 0.0,
                 reward: float = np.nan,
                 r_dia: float = None,
                 vol_diaria: float = None) -> None:
        """Lo que ve el humano durante el entrenamiento a modo orientativo"""

        valor_hoy = float(self.portfolio_value_usd)
        rentab_pct = float((r_dia or 0.0) * 100.0)
        balance_neto = float(valor_hoy - self._aportado_acum)
        posiciones_pct = self._posiciones_pct(valor_hoy)

        self.history.append({
            "fecha": self.today,
            "reward": float(reward),
            "valor_cartera": valor_hoy,
            "rentabilidad_diaria(%)": round(rentab_pct, 6),
            "balance_neto": round(balance_neto, 6),
            "posiciones(%)": posiciones_pct,
        })

    def _make_observation(self,
                          done: bool = False,
                          contribution_applied: float = 0.0,
                          reward: float = 0.0) -> Dict[str, Any]:
        """Lo que ve el agente para entrenarse"""
        return self.obs_engine.build(done=done, contribution_applied=contribution_applied, reward=reward)

    # ---------- Helpers ----------
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

    def _volatilidad_diaria_actual(self, lookback_days: int = 120) -> float: # Ventana de 6 meses atras
        """
        Volatilidad diaria (std) de la cartera usando:
          - participaciones actuales (self.positions_shares)
          - precios 'hasta hoy' (self._prices_ffill.loc[:self.today])
        Convierte participaciones -> pesos con los precios de 'hoy'.
        """
        # Ventana hasta hoy
        if self.today is None:
            return 0.0
        px = self._prices_ffill.loc[:self.today]
        if px.shape[0] < 2:
            return 0.0

        # Precios de hoy (última fila de la ventana)
        last = px.iloc[-1]
        vals = {}
        for t, sh in self.positions_shares.items():
            if not sh:
                continue
            p = last.get(t, np.nan)
            if pd.notna(p) and p > 0:
                vals[t] = float(sh) * float(p)

        if not vals:
            return 0.0

        total = sum(vals.values())
        if total <= 0:
            return 0.0

        w = pd.Series({t: v / total for t, v in vals.items()}, dtype=float)

        # Rentabilidades diarias de la ventana
        rets = px.pct_change().dropna()
        cols = w.index.intersection(rets.columns)
        if len(cols) == 0:
            return 0.0

        port = rets[cols] @ w[cols].values  # serie de rentabilidades diarias
        if port.empty:
            return 0.0

        tail = port.tail(lookback_days)
        if tail.shape[0] < 2:
            return 0.0

        vol = float(tail.std(ddof=1))
        return vol if np.isfinite(vol) else 0.0

    def _posiciones_pct(self, valor_hoy: float) -> dict:
        if not np.isfinite(valor_hoy) or valor_hoy <= 0.0:
            return {}
        out = {}
        last_prices = {t: self._price_today(t) for t in self.universe}
        for t, sh in self.positions_shares.items():
            if sh <= 0.0:
                continue
            p = last_prices.get(t, np.nan)
            if not pd.notna(p) or p <= 0.0:
                continue
            v = sh * p
            if v > 0.0:
                out[t] = round(100.0 * v / valor_hoy, 4)
        # Limpieza numérica: que no pase de 100 por redondeos
        s = sum(out.values())
        if 95.0 < s < 105.0 and s != 0.0:
            k = 100.0 / s
            out = {t: round(v * k, 4) for t, v in out.items()}
        return out