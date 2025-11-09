import numpy as np
from src.environment.environment import InvestorProfile

class ProfileSampler:
    def __init__(self, rng: np.random.Generator):
        self.rng = rng

    def __call__(self) -> InvestorProfile:
        riesgo = int(self.rng.integers(1, 8))
        horizonte = int(self.rng.integers(1, 16))
        liquidez = float(self.rng.uniform(1000, 50_000))
        aportacion = float(self.rng.uniform(0, 5_000))

        return InvestorProfile(
            riesgo=riesgo,
            horizonte_anios=horizonte,
            aportacion_mensual_usd=aportacion,
            liquidez_inicial_usd=liquidez
        )
