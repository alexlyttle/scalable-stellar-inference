from typing import Callable, Optional
from .emulator import Emulator
from .bolometric_corrections import BolometricCorrections


class Star:
    log_g_sun = 4.44
    log_teff_sun = 3.761326
    bol_mag_sun = 4.75
    param_names = ["evol", "log_mass", "M_H", "Y", "a_MLT", "plx", "Av"]
    determ_names = ["log_age", "log_Teff", "log_R", "log_Dnu", "log_g", "log_L"]

    def __init__(self, bands: Optional[list]=None, backend: str="jax"):
        self.emulator = Emulator(backend=backend)
        bolometric_corrections = None
        if bands is not None:
            bolometric_corrections = BolometricCorrections(bands, backend=backend)
        self.bolometric_corrections = bolometric_corrections

        if backend == "jax":
            self.model = self._model_jax()
        else:
            message = f"Backend value {backend!r} is invalid."
            raise ValueError(message)

    def _model_jax(self) -> Callable:
        import jax.numpy as jnp

        def model(params: dict) -> dict:
            """Predict from model params."""
            determs = {}  # deterministic parameters to return
            log_mass = params["log_mass"]
            mh = params["M_H"]
            
            determs["mass"] = mass = 10**log_mass
            inputs = jnp.stack(
                [params["evol"], mass, mh, params["Y"], params["a_MLT"]], 
                axis=-1
            )
            outputs = self.emulator(inputs).squeeze()

            determs["log_age"] = log_age = outputs[0] - 9  # log(age) in Gyr
            determs["log_Teff"] = log_teff = outputs[1]
            determs["log_R"] = log_rad = outputs[2]
            determs["log_Dnu"] = outputs[3]
            determs["log_g"] = log_g = log_mass - 2 * log_rad + self.log_g_sun
            determs["log_L"] = log_lum = 2 * log_rad + 4 * (log_teff - self.log_teff_sun)

            determs["Teff"] = teff = 10**log_teff
            determs["R"] = 10**log_rad
            determs["age"] = 10**log_age

            # TODO: if self.bands is not None, calculate magnitudes and assume parallax and extinciton
            if self.bolometric_corrections is None:
                return determs

            inputs = jnp.stack(
                [teff, log_g, mh, params["Av"]],
                axis=-1
            )
            bc = self.bolometric_corrections(inputs).squeeze()
            determs["bol_mag"] = bol_mag = self.bol_mag_sun - 2.5 * log_lum
            determs["abs_mag"] = abs_mag = bol_mag - bc
            determs["mag"] = abs_mag - 5 * (1 - jnp.log10(params["distance"]))
            determs["plx"] = 1 / params["distance"]

            return determs
        return model

    def __call__(self, params: dict):
        return self.model(params)
