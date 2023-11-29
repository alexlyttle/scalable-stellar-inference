import numpyro
import numpyro.distributions as dist

import numpy as np
import jax.numpy as jnp

from jax import vmap
from typing import Optional
from .star import Star


class SingleStarModel:
    def __init__(self, const: Optional[dict]=None, bands: Optional[list]=None):
        self.star = Star(bands=bands, backend="jax")
        self.photometry = False if bands is None else True
        self.const = self._default_const(const=const)

    def _default_const(self, const: Optional[dict]=None) -> dict:
        if const is None:
            const = {}
        const.setdefault("dof", 10)
        const.setdefault("evol", dict(concentration1=2.0, concentration0=5.0))
        const.setdefault("log_mass", dict(loc=0.0, scale=0.3))
        const.setdefault("M_H", dict(loc=0.0, scale=0.5))

        if self.photometry:
            const.setdefault("distance", dict(concentration=3.0, rate=1e-3))
            const.setdefault("Av", dict(loc=1.0, scale=1.0))  # TODO: correlate with distance?
        return const

    def sample_star(self) -> dict:
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

        if self.photometry:
            params["distance"] = numpyro.sample("distance", dist.Gamma(**self.const["distance"]))
            params["Av"] = numpyro.sample("Av", dist.TruncatedNormal(**self.const["Av"], low=0.0, high=6.0))
        
        return params

    def __call__(self, obs: Optional[dict]=None) -> None:
        params = self.sample_star()

        determs = self.star(params)
        
        for key, value in determs.items():
            numpyro.deterministic(key, value)

        if obs is None:
            return

        for key, value in obs.items():
            numpyro.sample(f"{key}_obs", dist.StudentT(self.const["dof"], determs[key], self.const[key]["scale"]), obs=value)


class MultiStarModel(SingleStarModel):
    def __init__(self, num_stars: int, const: Optional[dict]=None, bands: Optional[list]=None):
        super(MultiStarModel, self).__init__(const=const, bands=bands)
        self.num_stars = num_stars
    
    def sample_population(self):
        hyperparams = {}
        hyperparams["mu_MLT"] = numpyro.sample("mu_MLT", dist.Uniform(low=1.3, high=2.7))
        return hyperparams

    def sample_star(self, hyperparams: dict) -> dict:
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
        params["a_MLT"] = numpyro.deterministic("a_MLT", jnp.broadcast_to(hyperparams["mu_MLT"], (self.num_stars,)))

        if self.photometry:
            params["distance"] = numpyro.sample("distance", dist.Gamma(**self.const["distance"]))
            params["Av"] = numpyro.sample("Av", dist.TruncatedNormal(**self.const["Av"], low=0.0, high=6.0))
        
        return params

    def __call__(self, obs: Optional[dict]=None) -> None:
        hyperparams = self.sample_population()

        with numpyro.plate("star", self.num_stars):
            params = self.sample_star(hyperparams)

        determs = vmap(self.star)(params)
        
        for key, value in determs.items():
            numpyro.deterministic(key, value)

        if obs is None:
            return

        for key, value in obs.items():
            # TODO: add obs mask to allow some unobserved variables
            numpyro.sample(f"{key}_obs", dist.StudentT(self.const["dof"], determs[key], self.const[key]["scale"]), obs=value)
