import numpy as np
from math import sqrt, exp, log
from typing import Dict, List, Optional

class MetricsEngine:
    """
    Calcula y expone métricas de seguimiento.
    - daily_rent_net: r_net diario (neto de costes en el step).
    - daily_volatility: std diaria (ventana corta).
    - daily_sharpe: Sharpe anualizado estimado desde diarios (ventana corta).
    - anual_rent_net: retorno anual equivalente desde diarios (ventana larga).
    - anual_volatility: std anualizada (ventana larga).
    - anual_sharpe: Sharpe anualizado (ventana larga).
    - risk_profile: constante por episodio. Se fija en reset().
    """

    def __init__(self, short_window: int = 30, long_window: int = 252):
        self.short_window = int(short_window)
        self.long_window = int(long_window)
        self._ret_daily_buf: List[float] = []
        self._risk_profile: Optional[int] = None
        self._extras = {}
        self._anual_trades = 0
        self._days_elapsed = 0

    # ---------- ciclo de vida episodio ----------
    def reset(self, risk_profile: int) -> None:
        self._ret_daily_buf = []
        self._risk_profile = int(risk_profile)

    # ---------- actualización por step ----------
    def update(self, r_net: float) -> None:
        r = float(r_net)
        self._ret_daily_buf.append(r)
        if len(self._ret_daily_buf) > self.long_window:
            self._ret_daily_buf.pop(0)

    # ---------- utilidades ----------
    @staticmethod
    def _stats(x: List[float]) -> (float, float):
        n = len(x)
        if n < 2:
            return 0.0, 0.0
        mu = float(np.mean(x))
        sd = float(np.std(x, ddof=1))
        return mu, sd

    def step_day_counter(self) -> None:
        self._days_elapsed += 1

    def set_reward(self, final_reward: float) -> None:
        self._extras["final_reward"] = float(final_reward)

    def update_trading(self, n_trades: int, n_etfs: int) -> None:
        self._extras["n_trades"] = int(n_trades)
        self._extras["n_etfs"] = int(n_etfs)
        self._anual_trades += int(n_trades)
        # proyección anual simple 252*día
        if self._days_elapsed > 0:
            tpd = self._anual_trades / self._days_elapsed
            self._extras["trades_per_year"] = round(tpd * 252, 2)

    def set_risk_limits(self, risk_limit: float, hard_tol: float) -> None:
        self._extras["risk_limit"] = float(risk_limit)
        self._extras["hard_vol_tolerance"] = float(hard_tol)

    def set_risk_diagnostics(self, sigma_d: float, risk_limit: float) -> None:
        if risk_limit > 0:
            self._extras["volatility_target_ratio"] = round(float(sigma_d / risk_limit), 6)
            self._extras["excess_volatility"] = max(0.0, float(sigma_d - risk_limit))

    def set_equity_and_alignment(self, w_equity: float, rv_align: float) -> None:
        self._extras["equity_percent"] = round(100.0 * float(w_equity), 2)
        self._extras["rv_align"] = float(rv_align)

    def set_concentration(self, hhi: float) -> None:
        self._extras["hhi"] = float(hhi)

    def set_divers_country_sector(self, score: float) -> None:
        self._extras["div_country_sector"] = float(score)

    def set_trade_penalty(self, trade_penalty: float) -> None:
        self._extras["trade_penalty"] = float(trade_penalty)

    def get_short_volatility(self) -> float:
        """Devuelve la volatilidad diaria (ventana corta) ya calculada internamente a partir de r_net."""
        buf = self._ret_daily_buf
        win_short = buf[-self.short_window:] if len(buf) >= self.short_window else buf
        if len(win_short) < 2:
            return 0.0
        _, sd = self._stats(win_short)
        return float(sd)

    # ---------- consulta de métricas ----------
    def get_metrics(self) -> Dict[str, float]:
        buf = self._ret_daily_buf

        # Ventana corta (sensibilidad reciente)
        win_short = buf[-self.short_window:] if len(buf) >= self.short_window else buf
        mu_s, sd_s = self._stats(win_short)
        daily_volatility = sd_s
        daily_sharpe = (mu_s / (sd_s + 1e-12)) * sqrt(252.0) if len(win_short) >= 2 else 0.0

        # Ventana larga (anual)
        win_ann = buf
        mu_a, sd_a = self._stats(win_ann)
        anual_volatility = (sd_a * sqrt(252.0)) if len(win_ann) >= 2 else 0.0
        anual_sharpe = (mu_a / (sd_a + 1e-12)) * sqrt(252.0) if len(win_ann) >= 2 else 0.0

        # Retorno anual equivalente desde diarios
        if len(win_ann) >= 2:
            eps = 1e-12
            log_g = float(np.mean([log(1.0 + max(-0.999999, r)) for r in win_ann]))
            anual_rent_net = float(exp(log_g * 252.0) - 1.0)
        else:
            anual_rent_net = 0.0

        out = {
            "risk_profile": int(self._risk_profile) if self._risk_profile is not None else -1,
            "daily_rent_net": float(buf[-1]) if buf else 0.0,
            "daily_volatility": float(daily_volatility),
            "daily_sharpe": float(daily_sharpe),
            "anual_rent_net": float(anual_rent_net),
            "anual_volatility": float(anual_volatility),
            "anual_sharpe": float(anual_sharpe),
        }
        for k, v in self._extras.items():
            if k not in out:
                out[k] = float(v)
        return out
