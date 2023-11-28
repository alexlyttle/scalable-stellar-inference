import os
import numpy as np
from tensorflow.keras.models import load_model

from . import PACKAGEDIR

class Emulator:
    PATH = os.path.join(PACKAGEDIR, "emulator")

    def __init__(self, backend="numpy"):
        # Currently uses tensorflow to load weights
        # TODO: load only weights to remove tensorflow dependence
        model = load_model(self.PATH)
        self.weights = model.get_weights()
        self.offset = model.layers[-1].offset
        self.scale = model.layers[-1].scale

        if backend.lower() in ["numpy", "np"]:
            self.model = self._model_numpy()
        elif backend.lower() == "jax":
            self.model = self._model_jax()
        elif backend.lower() in ["pytensor", "pt"]:
            self.model = self._model_pytensor()
        elif backend.lower() in ["tensorflow", "tf"]:
            self.model = self._model_tensorflow()
        else:
            message = f"Backend value {backend!r} is invalid."
            raise ValueError(message)

    def _model_tensorflow(self):
        """Load tensorflow model."""
        import tensorflow as tf
        tf_model = load_model(self.PATH)
        def model(x):
            x = tf.constant(x)
            return tf_model(x)
        return model

    def _model_numpy(self):
        """Load numpy model."""
        def model(x):
            """Emulator model.
            
            Args:
                x (array-like): Neural network inputs.
            
            Returns:
                jax.numpy.ndarray: Neural network outputs.
            """
            x = np.atleast_2d(x)
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
        
        # convert to jax arrays
        weights = [jnp.array(w) for w in self.weights]
        offset = jnp.array(self.offset)
        scale = jnp.array(self.scale)
        
        def model(x):
            """Emulator model.
            
            Args:
                x (array-like): Neural network inputs.
            
            Returns:
                jax.numpy.ndarray: Neural network outputs.
            """
            x = jnp.atleast_2d(jnp.array(x))
            x -= weights[0]
            x /= weights[1]**0.5
            for w, b in zip(weights[3:-2:2], weights[4:-1:2]):
                x = relu(jnp.matmul(x, w) + b)
            x = jnp.matmul(x, weights[-2]) + weights[-1]
            return offset + scale * x
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
            x = pt.as_tensor(x, ndim=2)
            x -= self.weights[0]
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
