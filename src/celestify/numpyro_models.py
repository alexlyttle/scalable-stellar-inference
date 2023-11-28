import numpyro
import numpyro.distributions as dist

import numpy as np
import jax.numpy as jnp

from .star import Star

class SingleStarModel:
    def __init__(self, const=None, bands=None):
        self.star = Star(bands=bands, backend="jax")
        self.const = self._default_const(const=const)
        self.bands = bands

    def _default_const(self, const=None):
        if const is None:
            const = {}
        const.setdefault("dof", 10)
        const.setdefault("evol", dict(concentration1=2.0, concentration0=5.0))
        const.setdefault("log_mass", dict(loc=0.0, scale=0.3))
        const.setdefault("M_H", dict(loc=0.0, scale=0.5))
        return const

    def star_prior(self):
        params = {}
        params["evol"] = numpyro.sample("evol", dist.Beta(**self.const["evol"]))
        params["log_mass"] = numpyro.sample(
            "log_mass", 
            dist.TruncatedNormal(**self.const["log_mass"], low=np.log10(0.7), high=np.log10(2.3))
        )
        params["M_H"] = numpyro.sample(
            "M_H", 
            dist.TruncatedNormal(**self.const["M_H"], low=-0.9, high=0.5)
        )
        params["Y"] = numpyro.sample("Y", dist.Uniform(low=0.22, high=0.32))
        params["a_MLT"] = numpyro.sample("a_MLT", dist.Uniform(low=1.3, high=2.7))
        return params

    def __call__(self, obs=None):
        params = self.star_prior()

        # TODO: if self.star.bands is not None, need prior on parallax and extinction
        if self.star.bands is not None:
            pass

        determs = self.star(params)
        
        for key, value in determs.items():
            numpyro.deterministic(key, value)

        if obs is None:
            return

        for key, value in obs.items():
            numpyro.sample(f"{key}_obs", dist.StudentT(self.const["dof"], determs[key], self.const[key]["scale"]), obs=value)
