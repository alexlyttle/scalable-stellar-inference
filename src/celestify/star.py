from typing import Callable, Optional
from .emulator import Emulator
from .bolometric_corrections import BolometricCorrections
from math import log10

grav_constant = 6.6743e-8  # in cgs units


class StarBase:
    log_m_sun = log10(1.988410) + 33
    log_r_sun = log10(6.957) + 10
    log_g_sun = log10(grav_constant) + log_m_sun - 2 * log_r_sun
    log_l_sun = log10(3.828) + 33
    log_teff_sun = log10(5772.0)
    bol_mag_sun = 4.75
    log_zx_sun = log10(0.0181)

    def __init__(self, bands: Optional[list]=None, backend: str="jax"):
        self.emulator = Emulator(backend=backend)
        bolometric_corrections = None
        if bands is not None:
            bolometric_corrections = BolometricCorrections(bands, backend=backend)
        self.bolometric_corrections = bolometric_corrections

    def model(self, params: dict) -> dict:
        raise NotImplementedError()

    def __call__(self, params: dict):
        return self.model(params)


def Star(bands: Optional[list]=None, backend: str="jax"):
    if backend == "jax":
        from ._jax.star import Star
    else:
        message = f"Backend value {backend!r} is invalid."
        raise ValueError(message)
    return Star(bands, backend=backend)
