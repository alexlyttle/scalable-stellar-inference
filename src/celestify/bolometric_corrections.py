import numpy as np
from isochrones.mist.bc import MISTBolometricCorrectionGrid


class BolometricCorrectionsBase:
    def __init__(self, bands: list):
        self.points, self.values = self._load_grid(bands)

    def _load_grid(self, bands: list):
        grid = MISTBolometricCorrectionGrid(bands=bands)
        points = [grid.df.index.unique(level=name).to_numpy() for name in grid.df.index.names]
        shape = [x.shape[0] for x in points]
        values = np.reshape(grid.df[bands].to_numpy(), shape + [len(bands),])
        return points, values

    def model(self, x):
        raise NotImplementedError()

    def __call__(self, x):
        return self.model(x)


def BolometricCorrections(bands: list, backend: str="jax"):
    if backend == "jax":
        from ._jax.bolometric_corrections import BolometricCorrections
    else:
        message = f"Backend value {backend!r} is invalid."
        raise ValueError(message)
    return BolometricCorrections(bands)
