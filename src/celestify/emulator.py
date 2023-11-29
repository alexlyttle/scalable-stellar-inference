import os
from tensorflow.keras.models import load_model

from . import PACKAGEDIR


class EmulatorBase:
    PATH = os.path.join(PACKAGEDIR, "emulator")

    def __init__(self):
        # Currently uses tensorflow to load weights
        # TODO: load only weights to remove tensorflow dependence
        model = load_model(self.PATH)
        self.weights = model.get_weights()
        self.offset = model.layers[-1].offset
        self.scale = model.layers[-1].scale

    def model(self, x):
        raise NotImplementedError()

    def __call__(self, x):
        """Emulator model.
        
        Args:
            x (array-like): Neural network inputs.
        
        Returns:
            array-like: Neural network outputs.
        """
        # TODO: input validation
        return self.model(x)


def Emulator(backend="numpy"):
    if backend.lower() in ["np", "numpy"]:
        from ._numpy.emulator import Emulator
    elif backend.lower() == "jax":
        from ._jax.emulator import Emulator
    elif backend.lower() in ["pt", "pytensor"]:
        from ._pytensor.emulator import Emulator
    elif backend.lower() in ["tf", "tensorflow"]:
        from ._tensorflow.emulator import Emulator
    else:
        message = f"Backend value {backend!r} is invalid."
        raise ValueError(message)
    return Emulator()
