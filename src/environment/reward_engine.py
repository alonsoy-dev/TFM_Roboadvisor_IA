from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Dict
import numpy as np
import pandas as pd

@dataclass
class RewardConfig:
    w_gain: float = 0.60
    w_profile: float = 0.40
    w_risk: float = 0.50
    w_div: float = 0.15
    w_horz: float = 0.35

    # Riesgo 1..7 -> penalización creciente al downside para perfiles conservadores.
    risk_lambda_table: Dict[int, float] = None

    # Horizonte: penaliza más el downside cuando el horizonte es corto.
    horizon_lambda_min: float = 0.5   # 15 años
    horizon_lambda_max: float = 2.0   # 1 año
    horizon_years_min: int = 1
    horizon_years_max: int = 15

    def __post_init__(self):
        if self.risk_lambda_table is None:
            self.risk_lambda_table = {1: 3.0, 2: 2.5, 3: 2.0, 4: 1.5, 5: 1.0, 6: 0.7, 7: 0.5}

class RewardEngine:
    """
    Puntuación_diaria = 0.60 * Ganancia_bruta + 0.40 * Perfil_objetivo
    Ganancia_bruta = (V_t - flujo_t) / V_{t-1} - 1     # neto de aportaciones, comisiones incluidas

    Perfil_objetivo = 0.50 * Ajuste_Riesgo
                    + 0.15 * Ajuste_Diversificación
                    + 0.35 * Ajuste_Horizonte

    - Ajuste_Riesgo:  penaliza el 'downside' según perfil (1..7). Valor ≤ 0 (0 si no hay pérdidas).
    - Ajuste_Horizonte: penaliza el 'downside' más si el horizonte es corto. Valor ≤ 0 (0 si no hay pérdidas).
    - Ajuste_Diversificación: mide la “equidad” de la cartera agregada en países y sectores [0,1]
      (0 = concentración total; 1 = reparto perfecto)
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
                price_today_fn,  # callable(etf: str) -> float
                exposures: Optional[dict] = None) -> float:
        """
        Devuelve la Puntuación_diaria (float).
        """
        if prev_value is None or not np.isfinite(prev_value) or prev_value <= 0:
            return 0.0

        # Ganancia bruta (neutral a flujos de caja)
        r_net = ((curr_value - net_flow) / prev_value) - 1.0
        down = max(0.0, -r_net)

        # Ajuste de Riesgo (≤ 0)
        lam_risk = self._risk_lambda(riesgo)
        risk_adj = - lam_risk * down

        # Ajuste de Horizonte (≤ 0)
        lam_horz = self._horizon_lambda(horizonte_anios)
        horz_adj = - lam_horz * down

        # Ajuste de Diversificación [0,1]
        div_adj = self._diversification_score(positions_shares, price_today_fn, exposures)

        # Perfil objetivo
        perfil = (self.cfg.w_risk * risk_adj) + (self.cfg.w_div * div_adj) + (self.cfg.w_horz * horz_adj)

        # Puntuación final
        score = (self.cfg.w_gain * r_net) + (self.cfg.w_profile * perfil)
        return float(score)

# ---------- helpers ----------
    def _risk_lambda(self, riesgo: int) -> float:
        return self.cfg.risk_lambda_table.get(int(riesgo), 1.5)

    def _horizon_lambda(self, h: int) -> float:
        h = int(max(self.cfg.horizon_years_min, min(self.cfg.horizon_years_max, h)))
        # Interpola: 1 año -> horizon_lambda_max ; 15 años -> horizon_lambda_min
        num = (h - self.cfg.horizon_years_min) / (self.cfg.horizon_years_max - self.cfg.horizon_years_min)
        return self.cfg.horizon_lambda_max + (self.cfg.horizon_lambda_min - self.cfg.horizon_lambda_max) * num

    def _weights_today(self, positions_shares: Dict[str, float], price_today_fn) -> Dict[str, float]:
        vals, total = {}, 0.0
        for t, sh in positions_shares.items():
            if not sh: continue
            px = price_today_fn(t)
            if px is None or not np.isfinite(px) or px <= 0: continue
            v = float(sh) * float(px)
            if v > 0:
                vals[t] = v
                total += v
        if total <= 0:
            return {}
        return {t: v / total for t, v in vals.items()}

    def _portfolio_exposure(self, w_etf: Dict[str, float], by_ticker: dict, key: str) -> Optional[pd.Series]:
        if not w_etf or not by_ticker:
            return None
        rows = []
        for t, w in w_etf.items():
            df = by_ticker.get(t)
            if df is None or df.empty or key not in df.columns or "weight" not in df.columns:
                continue
            tmp = df[[key, "weight"]].copy()
            tmp["weight"] *= w
            rows.append(tmp)
        if not rows:
            return None
        s = pd.concat(rows).groupby(key)["weight"].sum()
        s = s[s > 0]
        tot = s.sum()
        return (s / tot) if tot > 0 else None

    def _evenness_from_series(self, s: Optional[pd.Series]) -> float:
        """
        Evenness [0,1]
        1 = reparto perfecto; 0 = concentración total.
        """
        if s is None or s.empty:
            return 0.0
        w = s.values.astype(float)
        K = int((w > 0).sum())
        if K <= 1:
            return 0.0
        hhi = float((w ** 2).sum())
        hhi_min = 1.0 / K
        if hhi >= 1.0:
            return 0.0
        return float((1.0 - hhi) / (1.0 - hhi_min))  # [0,1]

    def _diversification_score(self,
                               positions_shares: Dict[str, float],
                               price_today_fn,
                               exposures: Optional[dict]) -> float:
        """
        Score de diversificación ∈ [0,1] combinando país y sector.
        - 0: muy concentrado.
        - 1: perfectamente equitativo entre las categorías presentes.
        Si faltan expos o no hay posiciones, devuelve 0.0 (neutro-malo).
        """
        if not exposures:
            return 0.0

        w_etf = self._weights_today(positions_shares, price_today_fn)
        if not w_etf:
            return 0.0

        evens = []
        c_by = exposures.get("country_by_ticker")
        s_by = exposures.get("sector_by_ticker")

        if c_by is not None:
            sc = self._portfolio_exposure(w_etf, c_by, key="country")
            evens.append(self._evenness_from_series(sc))
        if s_by is not None:
            ss = self._portfolio_exposure(w_etf, s_by, key="sector")
            evens.append(self._evenness_from_series(ss))

        if not evens:
            return 0.0

        return float(np.mean(evens))  # [0,1]
