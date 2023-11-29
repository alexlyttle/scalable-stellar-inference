import jax.numpy as jnp
from jax.typing import ArrayLike
from jax.scipy.interpolate import RegularGridInterpolator
from ..bolometric_corrections import BolometricCorrectionsBase


class BolometricCorrections(BolometricCorrectionsBase):
    def __init__(self, bands: list):
        super(BolometricCorrections, self).__init__(bands)
        self.interpolator = RegularGridInterpolator(self.points, self.values)

    def model(self, x: ArrayLike) -> jnp.ndarray:
        x = jnp.array(x)
        return self.interpolator(x)
