import pytensor.tensor as pt
from ..emulator import EmulatorBase


class Emulator(EmulatorBase):
        
    def model(self, x: pt.TensorLike) -> pt.TensorVariable:
        """Emulator model.
        
        Args:
            x (array-like): Neural network inputs.
        
        Returns:
            pytensor.TensorVariable: Neural network outputs.
        """
        x = pt.as_tensor(x, ndim=2)
        x -= self.weights[0]
        x /= self.weights[1]**0.5
        for w, b in zip(self.weights[3:-2:2], self.weights[4:-1:2]):
            x = pt.matmul(x, w) + b
            x = pt.where(x >= 0, x, 0)  # relu
        x = pt.matmul(x, self.weights[-2]) + self.weights[-1]
        return self.offset + self.scale * x
