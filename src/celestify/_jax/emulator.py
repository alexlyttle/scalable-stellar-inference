import jax.numpy as jnp
from jax.nn import relu
from jax.typing import ArrayLike
from ..emulator import EmulatorBase


class Emulator(EmulatorBase):

    def __init__(self):
        super(Emulator, self).__init__()
        self.weights = [jnp.array(w) for w in self.weights]
        self.bias = [jnp.array(b) for b in self.bias]
        self.input_loc = jnp.array(self.input_loc)
        self.input_scale = jnp.array(self.input_scale)
        self.output_loc = jnp.array(self.output_loc)
        self.output_scale = jnp.array(self.output_scale)

    # def model(self, x: ArrayLike) -> jnp.ndarray:
    #     """Emulator model.

    #     Args:
    #         x (array-like): Neural network inputs.

    #     Returns:
    #         jax.numpy.ndarray: Neural network outputs.
    #     """
    #     x = jnp.array(x)
    #     x -= self.weights[0]
    #     x /= self.weights[1]**0.5
    #     for w, b in zip(self.weights[3:-2:2], self.weights[4:-1:2]):
    #         x = relu(jnp.matmul(x, w) + b)
    #     x = jnp.matmul(x, self.weights[-2]) + self.weights[-1]
    #     return self.offset + self.scale * x
    
    def model(self, x: ArrayLike) -> jnp.ndarray:
        """Emulator model.

        Args:
            x (array-like): Neural network inputs.

        Returns:
            jax.numpy.ndarray: Neural network outputs.
        """
        x = jnp.array(x)
        x -= self.input_loc
        x /= self.input_scale
        for w, b in zip(self.weights[:-1], self.bias[:-1]):
            x = relu(jnp.matmul(x, w) + b)
        x = jnp.matmul(x, self.weights[-1]) + self.bias[-1]
        return self.output_loc + self.output_scale * x
