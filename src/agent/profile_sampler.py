import numpy as np
from src.environment.environment import InvestorProfile

class ProfileSampler:
    def __init__(self, seed: int = 123):
        self.rng = np.random.default_rng(seed)

    def __call__(self) -> InvestorProfile:
        # Riesgo entero 1..7
        riesgo = int(self.rng.integers(1, 8))

        # Horizonte 1..15 años
        horizonte = int(self.rng.integers(1, 16))

        # Liquidez inicial: 1k–50k
        liquidez = float(self.rng.uniform(1000, 50_000))

        # Aportación mensual: 0–5k
        aportacion = float(self.rng.uniform(0, 5_000))

        return InvestorProfile(
            riesgo=riesgo,
            horizonte_anios=horizonte,
            aportacion_mensual_usd=aportacion,
            liquidez_inicial_usd=liquidez
        )
