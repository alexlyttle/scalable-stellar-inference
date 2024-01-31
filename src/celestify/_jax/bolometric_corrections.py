import jax.numpy as jnp
from jax.typing import ArrayLike
from jax.nn import relu
from jax.scipy.interpolate import RegularGridInterpolator
from ..bolometric_corrections import BolometricCorrectionsBase


class BolometricCorrections(BolometricCorrectionsBase):
    def __init__(self, bands: list):
        super(BolometricCorrections, self).__init__(bands)
        self.interpolator = RegularGridInterpolator(self.points, self.values)

    def interpolate(self, x: ArrayLike) -> jnp.ndarray:
        x = jnp.array(x)
        return self.interpolator(x)

    def model(self, x: ArrayLike) -> jnp.ndarray:
        x = jnp.array(x)
        x = jnp.atleast_2d(jnp.array(x))
        x -= self.minval 
        x /= (self.maxval - self.minval)
        for w, b in zip(self.weights[:-1], self.bias[:-1]):
            x = relu(jnp.matmul(x, w) + b)
        x = jnp.matmul(x, self.weights[-1]) + self.bias[-1]
        return self.loc + self.scale * x
