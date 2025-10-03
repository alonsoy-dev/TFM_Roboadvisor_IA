from dataclasses import dataclass
from typing import Optional, Dict
import numpy as np
import pandas as pd

@dataclass
class RewardConfig:
    # Pesos externos
    w_gain: float = 0.60
    w_profile: float = 0.40

    # Pesos internos del perfil
    w_risk: float = 0.50
    w_div: float = 0.15
    w_horz: float = 0.35

    # Si pasas volatilidad_diaria, el cap = k_cap * volatilidad_diaria
    k_cap: float = 2.0

    # Volatilidad objetivo por perfil (diaria, decimal). Ajusta a tu escala real.
    target_vol_table: Dict[int, float] = None
    # Tolerancia alrededor del objetivo (m = tol_mult * sigma_target)
    tol_mult: float = 1.0

    # Horizonte (años)
    horizon_years_min: int = 1
    horizon_years_max: int = 15
    # Vol objetivo diaria para horizonte corto (1 año) y largo (15 años)
    target_vol_max: float = 0.015  # ~1.5%/día (corto)
    target_vol_min: float = 0.002  # ~0.2%/día (largo)

    def __post_init__(self):
        if self.target_vol_table is None:
            self.target_vol_table = {
                1: 0.002, 2: 0.003, 3: 0.004,
                4: 0.006, 5: 0.008, 6: 0.010, 7: 0.012
            }

class RewardEngine:
    """
    Recompensa final [-1,1]:
        score = w_gain * gain_norm + w_profile * perfil

      gain_norm = tanh( r_net / cap ),
        r_net = (Vt - flujo_t)/Vt-1 - 1
        cap = k_cap * volatilidad_diaria

      perfil = clip( w_risk*risk_adj + w_div*div_adj + w_horz*horz_adj, [-1,1] )
        risk_adj [-1,1]: cercanía a volatilidad objetivo por perfil
        horz_adj [-1,1]: cercanía a volatilidad objetivo por horizonte
        div_adj [-1,1]: diversificación
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
        if prev_value is None or not np.isfinite(prev_value) or prev_value <= 0:
            return 0.0

        # Rendimiento neto (descontando flujos)
        r_net = ((curr_value - net_flow) / prev_value) - 1.0

        # Capping adaptativo y ganancia suavizada [-1,1]
        cap = self.cfg.k_cap * float(volatilidad_diaria)
        cap = max(1e-9, float(cap))
        gain_norm = float(np.tanh(r_net / cap))

        # Riesgo: cercanía a volatilidad objetivo por perfil [-1,1]
        sigma_target = float(self.cfg.target_vol_table.get(int(riesgo),
                                                          self.cfg.target_vol_table[4]))
        m = max(1e-9, self.cfg.tol_mult * sigma_target)
        sigma_ref = float(volatilidad_diaria) if (volatilidad_diaria and volatilidad_diaria > 0) else sigma_target
        risk_score_01 = 1.0 - abs(sigma_ref - sigma_target) / m
        risk_score_01 = float(np.clip(risk_score_01, 0.0, 1.0))
        risk_adj = 2.0 * risk_score_01 - 1.0

        # Horizonte: cercanía a volatilidad objetivo por horizonte [-1,1]
        sigma_target_h = self._interp_horizon_target_vol(int(horizonte_anios))
        m_h = max(1e-9, self.cfg.tol_mult * sigma_target_h)
        sigma_ref_h = sigma_ref
        horz_score_01 = 1.0 - abs(sigma_ref_h - sigma_target_h) / m_h
        horz_score_01 = float(np.clip(horz_score_01, 0.0, 1.0))
        horz_adj = 2.0 * horz_score_01 - 1.0

        # Diversificación [-1,1]
        div_raw = self._diversification_score(positions_shares, price_today_fn, exposures)
        div_adj = float(2.0 * div_raw - 1.0)

        # Perfil y score final
        perfil = (
            self.cfg.w_risk * risk_adj +
            self.cfg.w_div  * div_adj  +
            self.cfg.w_horz * horz_adj
        )
        perfil = float(np.clip(perfil, -1.0, 1.0))
        score = self.cfg.w_gain * gain_norm + self.cfg.w_profile * perfil
        return float(np.clip(score, -1.0, 1.0))

    # ---------- helpers ----------
    def _interp_horizon_target_vol(self, h: int) -> float:
        h = int(max(self.cfg.horizon_years_min, min(self.cfg.horizon_years_max, h)))
        num = (h - self.cfg.horizon_years_min) / (self.cfg.horizon_years_max - self.cfg.horizon_years_min)
        return float(
            self.cfg.target_vol_max +
            (self.cfg.target_vol_min - self.cfg.target_vol_max) * num
        )

    def _weights_today(self, positions_shares: Dict[str, float], price_today_fn) -> Dict[str, float]:
        vals, total = {}, 0.0
        for t, sh in positions_shares.items():
            if not sh:
                continue
            px = price_today_fn(t)
            if px is None or not np.isfinite(px) or px <= 0:
                continue
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
        return float((1.0 - hhi) / (1.0 - hhi_min))

    def _diversification_score(self,
                               positions_shares: Dict[str, float],
                               price_today_fn,
                               exposures: Optional[dict]) -> float:
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
        return float(np.mean(evens))
