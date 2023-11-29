import numpy as np
from isochrones.mist.bc import MISTBolometricCorrectionGrid

class BolometricCorrections:
    
    def __init__(self, bands: list, backend: str="jax"):
        self.points, self.values = self._load_grid(bands)

        if backend == "jax":
            self.model = self._model_jax()
        else:
            message = f"Backend value {backend!r} is invalid."
            raise ValueError(message)
    
    def _load_grid(self, bands):
        grid = MISTBolometricCorrectionGrid(bands=bands)
        points = [grid.df.index.unique(level=name).to_numpy() for name in grid.df.index.names]
        shape = [x.shape[0] for x in points]
        values = np.reshape(grid.df[bands].to_numpy(), shape + [len(bands),])
        return points, values

    def _model_jax(self):
        import jax.numpy as jnp
        from jax.scipy.interpolate import RegularGridInterpolator
        interpolator = RegularGridInterpolator(self.points, self.values)
        
        def model(x):
            x = jnp.array(x)
            return interpolator(x)
        return model

    def __call__(self, x):
        return self.model(x)
