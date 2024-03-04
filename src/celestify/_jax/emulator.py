import jax.numpy as jnp
from jax.nn import relu
from jax.typing import ArrayLike
from ..emulator import EmulatorBase


class Emulator(EmulatorBase):

    def __init__(self):
        super(Emulator, self).__init__()
        self.weights = [jnp.array(w) for w in self.weights]
        self.offset = jnp.array(self.offset)
        self.scale = jnp.array(self.scale)

    def model(self, x: ArrayLike) -> jnp.ndarray:
        """Emulator model.

        Args:
            x (array-like): Neural network inputs.

        Returns:
            jax.numpy.ndarray: Neural network outputs.
        """
        x = jnp.array(x)
        x -= self.weights[0]
        x /= self.weights[1]**0.5
        for w, b in zip(self.weights[3:-2:2], self.weights[4:-1:2]):
            x = relu(jnp.matmul(x, w) + b)
        x = jnp.matmul(x, self.weights[-2]) + self.weights[-1]
        return self.offset + self.scale * x
