import numpy as np
from isochrones.mist.bc import MISTBolometricCorrectionGrid


class BolometricCorrectionsBase:
    def __init__(self, bands: list):
        self.points, self.values = self._load_grid(bands)

    def _load_grid(self, bands: list):
        grid = MISTBolometricCorrectionGrid(bands=bands)
        df = grid.df

        # df = grid.df.reset_index()
        # mask = (df.Teff >= 3000) & (df.Teff <= 20000) \
        #     & (df.logg >= 2.0) & (df.logg <= 6.0) \
        #     & (df["[Fe/H]"] >= -1.0) & (df["[Fe/H]"] <= 0.5) \
        #     & (df.Av <= 4.0)
        # df = df.loc[mask]
        # df = df.set_index(['Teff', 'logg', '[Fe/H]', 'Av'])

        # df = grid.df.reset_index("Av")
        # df = df.loc[df.Av == 0.0].drop(columns="Av")
        points = [df.index.unique(level=name).to_numpy() for name in df.index.names]
        shape = [x.shape[0] for x in points]
        values = np.reshape(df[bands].to_numpy(), shape + [len(bands),])
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
