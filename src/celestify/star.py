from .emulator import Emulator
from typing import Callable, Optional

class Star:
    log_g_sun = 4.44
    log_teff_sun = 3.761326
    param_names = ["evol", "log_mass", "M_H", "Y", "a_MLT"]

    def __init__(self, bands: Optional[list]=None, backend: str="jax"):
        self.bands = bands
        self.emulator = Emulator(backend=backend)

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

            determs["mass"] = mass = 10**log_mass
            inputs = jnp.stack(
                [params["evol"], mass, params["M_H"], params["Y"], params["a_MLT"]], 
                axis=-1
            )
            outputs = self.emulator(inputs).squeeze()

            determs["log_age"] = log_age = outputs[0] - 9  # log(age) in Gyr
            determs["log_Teff"] = log_teff = outputs[1]
            determs["log_R"] = log_rad = outputs[2]
            determs["log_Dnu"] = outputs[3]
            determs["log_g"] = log_mass - 2 * log_rad + self.log_g_sun
            determs["log_L"] = log_lum = 2 * log_rad + 4 * (log_teff - self.log_teff_sun)
            
            # TODO: if self.bands is not None, calculate magnitudes and assume parallax and extinciton

            return determs
        return model

    def __call__(self, params: dict):
        return self.model(params)
