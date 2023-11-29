import numpy as np
from numpy.typing import ArrayLike
from ..emulator import EmulatorBase


class Emulator(EmulatorBase):
        
    def model(self, x: ArrayLike) -> np.ndarray:
        """Emulator model.
        
        Args:
            x (array-like): Neural network inputs.
        
        Returns:
            numpy.ndarray: Neural network outputs.
        """
        x = np.atleast_2d(x)
        x -= self.weights[0]
        x /= self.weights[1]**0.5
        for w, b in zip(self.weights[3:-2:2], self.weights[4:-1:2]):
            x = np.matmul(x, w) + b
            x = np.where(x >= 0, x, 0)  # relu
        x = np.matmul(x, self.weights[-2]) + self.weights[-1]
        return self.offset + self.scale * x
