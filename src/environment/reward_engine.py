from dataclasses import dataclass
from typing import Optional, Dict
import numpy as np
import pandas as pd

@dataclass
class RewardConfig:
    # --- Pesos de la recompensa base ---
    w_rar: float = 0.80  # Peso para el Retorno Ajustado al Riesgo (Sharpe Score)
    w_div: float = 0.20  # Peso para la Diversificación

    # --- Parámetros de la penalización por riesgo ---
    # Volatilidad límite permitida por perfil de riesgo
    risk_limit_vol_table: Dict[int, float] = None
    # Severidad de la penalización exponencial cuando se supera el límite
    # Un valor más alto significa que la recompensa cae mucho más rápido
    penalty_severity: float = 10.0

    # --- Parámetros del sharpe score ---
    sharpe_scaler_mult: float = 5.0
    horizon_years_min: int = 1
    horizon_years_max: int = 15
    # Volatilidad objetivo diaria para horizonte corto (1 año) y largo (15 años)
    target_vol_long_horizon: float = 0.015
    target_vol_short_horizon: float = 0.002

    def __post_init__(self):
        if self.risk_limit_vol_table is None:
            self.risk_limit_vol_table = {
                1: 0.003, 2: 0.004, 3: 0.006,
                4: 0.008, 5: 0.010, 6: 0.012, 7: 0.015
            }

class RewardEngine:
    """
    Motor de recompensa con penalización exponencial por riesgo:
    Recompensa Final = base_score * risk_penalty_factor

    1. risk_penalty_factor: Factor [0, 1]. Es 1 si la volatilidad está dentro
       del límite. Si se supera, decae exponencialmente hacia 0.
    2. base_score: Recompensa por la calidad de la gestión, combinando:
       - Retorno Ajustado al Riesgo (Sharpe Score).
       - Diversificación.
    """

    def __init__(self, config: RewardConfig):
        self.cfg = config

    def compute(self,
                prev_value: Optional[float],
                curr_value: float,
                net_flow: float,
                riesgo: int,
                horizonte_anios: int,
                positions_shares: Dict[str, float],
                price_today_fn,
                exposures: Optional[dict] = None,
                volatilidad_diaria: Optional[float] = None
                ) -> float:

        r_net = ((curr_value - net_flow) / prev_value) - 1.0
        sigma_ref = float(volatilidad_diaria if volatilidad_diaria and volatilidad_diaria > 0 else 0)

        # --- Parte 1: Calcular el Factor de Penalización por Riesgo ---
        risk_limit = float(self.cfg.risk_limit_vol_table.get(int(riesgo), 0.008))
        risk_penalty_factor = self._calculate_risk_penalty_factor(sigma_ref, risk_limit)

        # --- Parte 2: Calcular la Recompensa Base ---
        rar_score = self._calculate_rar_score(r_net, sigma_ref, horizonte_anios)
        div_raw = self._diversification_score(positions_shares, price_today_fn, exposures)
        div_score = float(2.0 * div_raw - 1.0)
        base_score = (self.cfg.w_rar * rar_score + self.cfg.w_div * div_score)
        base_score = float(np.clip(base_score, -1.0, 1.0))

        # --- Parte 3: La recompensa final ---
        final_score = base_score * risk_penalty_factor
        final_score = float(np.clip(final_score, -1.0, 1.0))

        return final_score

    # ---------- Motores de Cálculo ----------
    def _calculate_risk_penalty_factor(self, current_vol: float, vol_limit: float) -> float:
        if current_vol <= vol_limit:
            return 1.0  # Sin penalización
        else:
            # El exceso de volatilidad se normaliza por el propio límite.
            # Esto hace que la penalización sea relativa al nivel de riesgo permitido.
            excess_vol_normalized = (current_vol - vol_limit) / vol_limit

            # La penalización decae exponencialmente.
            # 'severity' controla la rapidez de la caída.
            penalty_factor = np.exp(-self.cfg.penalty_severity * excess_vol_normalized)
            return float(penalty_factor)

    def _calculate_rar_score(self, r_net: float, sigma_ref: float, horizonte_anios: int) -> float:
        daily_sharpe = r_net / (sigma_ref + 1e-9)
        sigma_target_h = self._interp_horizon_target_vol(int(horizonte_anios))
        sharpe_scaler = 1.0 / (sigma_target_h * self.cfg.sharpe_scaler_mult + 1e-9)
        rar_score = np.tanh(daily_sharpe * sharpe_scaler)
        return float(rar_score)

    # ---------- helpers ----------
    def _interp_horizon_target_vol(self, h: int) -> float:
        h = int(max(self.cfg.horizon_years_min, min(self.cfg.horizon_years_max, h)))
        frac = (h - self.cfg.horizon_years_min) / (self.cfg.horizon_years_max - self.cfg.horizon_years_min)
        return float(
            self.cfg.target_vol_short_horizon +
            (self.cfg.target_vol_long_horizon - self.cfg.target_vol_short_horizon) * frac
        )

    def _weights_today(self, positions_shares: Dict[str, float], price_today_fn) -> Dict[str, float]:
        vals, total = {}, 0.0
        for t, sh in positions_shares.items():
            if not sh: continue
            px = price_today_fn(t)
            if px is None or not np.isfinite(px) or px <= 0: continue
            v = float(sh) * float(px)
            if v > 0: vals[t], total = v, total + v
        return {t: v / total for t, v in vals.items()} if total > 0 else {}

    def _portfolio_exposure(self, w_etf: Dict[str, float], by_ticker: dict, key: str) -> Optional[pd.Series]:
        if not w_etf or not by_ticker: return None
        rows = [df[[key, "weight"]].assign(weight=df["weight"] * w) for t, w in w_etf.items() if
                (df := by_ticker.get(t)) is not None and not df.empty]
        if not rows: return None
        s = pd.concat(rows).groupby(key)["weight"].sum()
        s = s[s > 0]
        tot = s.sum()
        return (s / tot) if tot > 0 else None

    def _evenness_from_series(self, s: Optional[pd.Series]) -> float:
        if s is None or s.empty: return 0.0
        w = s.values
        K = len(w)
        if K <= 1: return 0.0
        hhi = (w ** 2).sum()
        return float(np.clip((1.0 - hhi) / (1.0 - 1.0 / K), 0.0, 1.0)) if hhi < 1.0 else 0.0

    def _diversification_score(self, positions_shares: Dict[str, float], price_today_fn,
                               exposures: Optional[dict]) -> float:
        if not exposures: return 0.0
        w_etf = self._weights_today(positions_shares, price_today_fn)
        if not w_etf: return 0.0
        evens = [self._evenness_from_series(self._portfolio_exposure(w_etf, exposures.get(by), key=key)) for by, key in
                 [("country_by_ticker", "country"), ("sector_by_ticker", "sector")] if exposures.get(by)]
        return float(np.mean(evens)) if evens else 0.0