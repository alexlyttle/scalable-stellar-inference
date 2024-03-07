import os, h5py
from . import PACKAGEDIR


class EmulatorBase:
    PATH = os.path.join(PACKAGEDIR, "data", "emulator_weights.h5")    

    def __init__(self):
        keys = [f"dense_{i}" for i in range(7, 14)]
        weights = []
        bias = []
        with h5py.File(self.PATH) as file:
            for key in keys:
                weights.append(file[key][key]["kernel:0"][()])
                bias.append(file[key][key]["bias:0"][()])
            self.input_loc = file["normalization/mean:0"][()]
            self.input_scale = file["normalization/variance:0"][()]**0.5
            self.output_loc = file["rescaling/offset"][()]
            self.output_scale = file["rescaling/scale"][()]
        self.weights = weights
        self.bias = bias

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
