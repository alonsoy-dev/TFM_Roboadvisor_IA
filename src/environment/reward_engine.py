from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Dict, Tuple
import numpy as np


@dataclass
class RewardConfig:
    # Pesos de cada término (w1..w4) y severidad gamma
    w1: float = 0.50   # retorno ajustado a riesgo y horizonte
    w2: float = 0.25   # alineación con asset allocation objetivo por perfil
    w3: float = 0.15   # concentración (HHI) objetivo por perfil
    w4: float = 0.10   # coste de trading
    gamma: float = 15.0  # penalización cuadrática por exceso de volatilidad

    # Bandas de RV por perfil: (lo, hi, target)
    equity_bands_by_risk: Dict[int, Tuple[float, float, float]] = None
    # Límite de volatilidad diaria por perfil
    risk_limit_vol_table: Dict[int, float] = None
    # Objetivo de HHI por perfil (HHI_target)
    hhi_target_by_risk: Dict[int, float] = None
    # Límite de activos por perfil
    max_assets_by_risk: Dict[int, int] = None

    # Escalador s(h): interpolación lineal en función del horizonte (años)
    horizon_years_min: int = 1
    horizon_years_max: int = 15
    s_h_short: float = 1.0   # tolerancia base para horizonte corto
    s_h_long: float = 4.0    # tolerancia mayor para horizonte largo

    # Diversos
    eps_weight: float = 1e-6

    def __post_init__(self):
        if self.equity_bands_by_risk is None:
            self.equity_bands_by_risk = {
                1:(0.05,0.25,0.15), 2:(0.15,0.35,0.25), 3:(0.30,0.50,0.40),
                4:(0.50,0.70,0.60), 5:(0.60,0.80,0.70), 6:(0.70,0.95,0.85),
                7:(0.90,1.00,0.95)
            }
        if self.risk_limit_vol_table is None:
            self.risk_limit_vol_table = {1:0.003,2:0.004,3:0.006,4:0.008,5:0.010,6:0.012,7:0.015}
        if self.hhi_target_by_risk is None:
            # más agresivo => más concentración deseada
            self.hhi_target_by_risk = {1:0.08,2:0.10,3:0.12,4:0.14,5:0.16,6:0.18,7:0.20}
        if self.max_assets_by_risk is None:
            # más agresivo => más concentración (menos activos)
            self.max_assets_by_risk = {
                1: 20, 2: 15, 3: 12, 4: 10, 5: 8, 6: 6, 7: 5
            }

    # s(h): crece linealmente con h
    def s_h(self, h_years: int) -> float:
        h = max(self.horizon_years_min, min(self.horizon_years_max, int(h_years)))
        frac = (h - self.horizon_years_min) / max(1, (self.horizon_years_max - self.horizon_years_min))
        return float(self.s_h_short + (self.s_h_long - self.s_h_short) * frac)


class RewardEngine:
    """
    Fórmula:
    R = w1 * tanh( r_net / (σ_d * s(h)) )
      + w2 * [ 1 - |w_eq - w_target(r)| / Δw_banda(r) ]
      + w3 * tanh( HHI - HHI_target(r) )
      - w4 * ( λ1 * TO + λ2 * N_trades )
      - γ * max(0, σ_d - σ_lim(r))^2
    """
    def __init__(self, config: RewardConfig):
        self.cfg = config

        # Coeficientes del coste de trading
        self.lambda_to: float = 1.0
        self.lambda_ntrades: float = 0.25

    # -------------------- API principal --------------------
    def compute(
        self,
        prev_value: Optional[float],
        curr_value: float,
        net_flow: float,
        riesgo: int,
        horizonte_anios: int,
        positions_shares: Dict[str, float],
        price_today_fn,
        exposures: Optional[dict] = None,
        volatilidad_diaria: float = None,
        turnover_L1: float = 0.0,     # si no lo calculas en el entorno, deja 0.0
        n_trades: int = 0,
        n_etfs: int = 0,
        weights_today: Optional[Dict[str, float]] = None,
    ) -> tuple[float, Dict[str, float]]:

        # 1) Retorno diario neto (sin aportaciones)
        if prev_value is None or prev_value <= 0:
            r_net = 0.0
        else:
            r_net = ((curr_value - net_flow) / prev_value) - 1.0

        sigma_d = float(volatilidad_diaria or 0.0)
        eps = self.cfg.eps_weight

        # 2) Pesos hoy (si no vienen)
        if weights_today is None:
            weights_today = self._weights_today(positions_shares, price_today_fn)

        # 3) Peso en RV actual
        w_eq, coverage = self._equity_weight(weights_today, exposures)

        # 4) RAR por horizonte: tanh( r_net / (sigma_d * s(h)) )
        rar = self._rar(r_net, sigma_d, int(horizonte_anios))

        # 5) Alineación con banda y target por perfil
        rv_align = self._rv_alignment(w_eq, int(riesgo))

        # 6) Concentración (HHI) vs objetivo por perfil
        hhi = self._hhi(weights_today)
        conc_term = float(np.tanh(hhi - self.cfg.hhi_target_by_risk.get(int(riesgo), 0.12)))

        # 7) Coste de trading: λ1*TO + λ2*N_trades
        trade_cost = self.lambda_to * float(turnover_L1) + self.lambda_ntrades * float(n_trades)

        # 8) Penalización por exceso de volatilidad: γ * max(0, σ_d - σ_lim)^2
        sigma_lim = self.cfg.risk_limit_vol_table[int(riesgo)]
        over = max(0.0, sigma_d - float(sigma_lim))
        risk_excess_pen = self.cfg.gamma * (over ** 2)

        # 9) Combinación lineal
        reward = (
            self.cfg.w1 * rar
            + self.cfg.w2 * rv_align
            + self.cfg.w3 * conc_term
            - self.cfg.w4 * trade_cost
            - risk_excess_pen
        )
        reward = float(np.clip(reward, -1.0, 1.0))

        diag = {
            "r_net": r_net,
            "sigma_d": sigma_d,
            "sigma_lim": sigma_lim,
            "risk_excess_pen": risk_excess_pen,
            "w_equity": w_eq,
            "rv_align": rv_align,
            "hhi": hhi,
            "conc_term": conc_term,
            "turnover_L1": float(turnover_L1),
            "n_trades": int(n_trades),
            "trade_cost": trade_cost,
            "rv_coverage": coverage,
        }
        return reward, diag

    # -------------------- Términos de la fórmula --------------------
    def _rar(self, r_net: float, sigma_d: float, h_years: int) -> float:
        if sigma_d <= 0:
            # sin riesgo observado: recompensa/penaliza suave por signo del retorno
            return float(np.tanh(10.0 * r_net))
        scale = self.cfg.s_h(h_years)
        x = r_net / (sigma_d * max(1e-12, scale))
        return float(np.tanh(x))

    def _rv_alignment(self, w_eq: float, riesgo: int) -> float:
        lo, hi, target = self.cfg.equity_bands_by_risk.get(int(riesgo), (0.0, 1.0, 0.5))
        bandwidth = max(1e-6, (hi - lo))  # Δw_banda(r)
        # métrica en [0,1]: 1 en el target, 0 si se desvía to el ancho de banda
        score = 1.0 - abs(w_eq - target) / bandwidth
        return float(np.clip(score, 0.0, 1.0))

    # -------------------- Helpers --------------------
    def _weights_today(self, positions_shares: Dict[str, float], price_today_fn) -> Dict[str, float]:
        vals, tot = {}, 0.0
        for t, sh in positions_shares.items():
            if sh <= 0: 
                continue
            p = price_today_fn(t)
            if p is None or not np.isfinite(p) or p <= 0:
                continue
            v = float(sh) * float(p)
            if v > 0:
                vals[t] = v
                tot += v
        return {t: v/tot for t, v in vals.items()} if tot > 0 else {}

    def _equity_weight(self, w: Dict[str, float], exposures: Optional[dict]) -> Tuple[float, float]:
        """Devuelve (peso_RV, cobertura_tickers_clasificados_en_[0,1])"""
        if not w or not exposures:
            return 0.0, 0.0
        tipos = exposures.get("type_sector", {})  # {ticker: "Equity"/"FixedIncome"/...}
        if not tipos:
            return 0.0, 0.0
        covered = sum(1 for t in w if t in tipos)
        coverage = covered / max(1, len(w))
        w_eq = float(sum(w.get(t, 0.0) for t, tp in tipos.items() if tp == "Equity"))
        return w_eq, float(coverage)

    def _hhi(self, w: Dict[str, float]) -> float:
        if not w:
            return 0.0
        ws = np.array(list(w.values()), dtype=float)
        return float(np.sum(ws * ws))  # 1/N si equiponderado
