import os
import numpy as np
from tensorflow.keras.models import load_model
# TODO: load only weights to remove tensorflow dependence

from . import PACKAGEDIR

class Emulator:
    PATH = os.path.join(PACKAGEDIR, "emulator")

    def __init__(self, backend="numpy"):
        tf_model = self._model_tf()
        self.weights = tf_model.get_weights()
        self.offset = tf_model.layers[-1].offset
        self.scale = tf_model.layers[-1].scale

        if backend == "numpy":
            self.model = self._model_numpy()
        elif backend == "jax":
            self.model = self._model_jax()
        elif backend == "pytensor":
            self.model = self._model_pytensor()
        elif backend == "tensorflow":
            self.model = self._model_tf

    def _model_tf(self):
        """Load tensorflow model."""
        return load_model(self.PATH)

    def _model_numpy(self):
        """Load numpy model."""
        def model(x):
            """Emulator model.
            
            Args:
                x (array-like): Neural network inputs.
            
            Returns:
                jax.numpy.ndarray: Neural network outputs.
            """
            x = np.array(x)
            x -= self.weights[0]
            x /= self.weights[1]**0.5
            for w, b in zip(self.weights[3:-2:2], self.weights[4:-1:2]):
                x = np.matmul(x, w) + b
                x = np.where(x >= 0, x, 0)  # relu
            x = np.matmul(x, self.weights[-2]) + self.weights[-1]
            return self.offset + self.scale * x
        return model

    def _model_jax(self):
        """Load jax model."""
        import jax.numpy as jnp
        from jax.nn import relu
        
        def model(x):
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
        return model

    def _model_pytensor(self):
        """Load pytensor model"""
        import pytensor.tensor as pt

        def model(x):
            """Emulator model.
            
            Args:
                x (array-like): Neural network inputs.
            
            Returns:
                jax.numpy.ndarray: Neural network outputs.
            """
            x = x - self.weights[0]
            x /= self.weights[1]**0.5
            for w, b in zip(self.weights[3:-2:2], self.weights[4:-1:2]):
                x = pt.matmul(x, w) + b
                x = pt.where(x >= 0, x, 0)  # relu
            x = pt.matmul(x, self.weights[-2]) + self.weights[-1]
            return self.offset + self.scale * x
        return model

    def __call__(self, x):
        """Emulator model.
        
        Args:
            x (array-like): Neural network inputs.
        
        Returns:
            jax.numpy.ndarray: Neural network outputs.
        """
        # TODO: input validation
        return self.model(x)
