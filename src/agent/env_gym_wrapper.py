# Wrapper Gym con domain randomization del InvestorProfile en cada episodio
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd

from src.environment.environment import PortfolioEnv
from src.agent.profile_sampler import ProfileSampler

class PortfolioGymWrapper(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, prices, aum, exposures, sampler: ProfileSampler,
                 indicators: dict = None,
                 start_date_limit: pd.Timestamp = None,
                 end_date_limit: pd.Timestamp = None):
        super().__init__()
        self.prices = prices
        self.aum = aum
        self.exposures = exposures
        self.sampler = sampler
        self.indicators = indicators
        self.start_date_limit = start_date_limit
        self.end_date_limit = end_date_limit

        # Para determinar el tamaño de la observación, hacemos un reset de prueba
        self.core: PortfolioEnv | None = None
        self.universe = list(prices.columns)

        vec, _ = self._reset_core()
        self._obs_dim = vec.shape[0]

        self.observation_space = spaces.Box(low=-1.0, high=1.0, shape=(self._obs_dim,), dtype=np.float32)
        self.action_space      = spaces.Box(low=0.0,  high=1.0, shape=(len(self.universe),), dtype=np.float32)

    # ---------- helpers ----------
    def _build_core(self):
        profile = self.sampler()
        self.core = PortfolioEnv(prices=self.prices, profile=profile, aum=self.aum, exposures=self.exposures, indicators=self.indicators)

    def _flatten_obs(self, obs):
        perfil = obs["perfil"]
        v_perfil = [perfil["riesgo"], perfil["horizonte_restante"], perfil["aportacion_mensual"]]

        c = obs["cartera"]
        v_cartera_num = [
            c["liquidez"], c["valor_cartera"], c["diversificacion"],
            c["concentracion_top3"], c["peso_renta_variable"],
            c["drawdown_3_meses"], c["limite_max_por_fondo"], c["rentabilidad_1d_lag"],
            c["risk_limit_target"], c["horizon_vol_target"]
        ]
        pesos_dict = {p["ticker"]: p["peso"] for p in c.get("pesos", [])}
        v_pesos = [pesos_dict.get(t, -1.0) for t in self.universe]

        mg = obs["mercado"]["global"]
        v_global = [mg["volatilidad_global"], mg["amplitud_universo"], mg["correlacion_media_3m"]]

        pa = obs["mercado"]["por_activo"]
        pa_by = {d["ticker"]: d for d in pa} if isinstance(pa, list) else {}
        v_por_activo = []
        for t in self.universe:
            d = pa_by.get(t)
            if not d:
                v_por_activo.extend([0.0] * 11)
            else:
                v_por_activo.extend([
                    d["subida_1_dia"], d["subida_5_dias"], d["subida_21_dias"], d["tendencia_6_meses"],
                    d["volatilidad_3_meses"], d["gap_ma20"], d["gap_ma120"],
                    d["rsi_14"], d["macd_hist"], d["atr_14"], d["aum"]
                ])

        vec = np.asarray(v_perfil + v_cartera_num + v_pesos + v_global + v_por_activo, dtype=np.float32)
        np.clip(vec, -1.0, 1.0, out=vec)
        return vec

    def _reset_core(self):
        self._build_core()
        obs = self.core.reset(start_date=self.start_date_limit, end_date=self.end_date_limit)
        return self._flatten_obs(obs), {}

    # ---------- API Gym ----------
    def reset(self, *, seed=None, options=None):
        return self._reset_core()

    def step(self, action):
        w = np.clip(np.asarray(action, dtype=np.float32), 0.0, None)
        s = float(w.sum())
        if s <= 0.0:
            full_obs_dict = self.core.step(action=None)
        else:
            w /= s
            action_dict = {t: float(w[i]) for i, t in enumerate(self.universe)}
            full_obs_dict = self.core.step(action=action_dict)

        vec = self._flatten_obs(full_obs_dict)
        reward = float(full_obs_dict["reward"])
        terminated = bool(full_obs_dict["done"])
        truncated = False
        info = {'reward_metrics': full_obs_dict.get('custom_metrics', {})}

        return vec, reward, terminated, truncated, info
