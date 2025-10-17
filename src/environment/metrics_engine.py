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

        return {
            "risk_profile": int(self._risk_profile) if self._risk_profile is not None else -1,
            "daily_rent_net": float(buf[-1]) if buf else 0.0,
            "daily_volatility": float(daily_volatility),
            "daily_sharpe": float(daily_sharpe),
            "anual_rent_net": float(anual_rent_net),
            "anual_volatility": float(anual_volatility),
            "anual_sharpe": float(anual_sharpe),
        }
