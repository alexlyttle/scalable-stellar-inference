import tensorflow as tf
from tensorflow.keras.models import load_model
from ..emulator import EmulatorBase


class Emulator(EmulatorBase):

    def __init__(self):
        self._model = load_model(self.PATH)

    def model(self, x) -> tf.Tensor:
        """Emulator model.
        
        Args:
            x (array-like): Neural network inputs.
        
        Returns:
            tensorflow.Tensor: Neural network outputs.
        """
        x = tf.constant(x)
        return self._model(x)
