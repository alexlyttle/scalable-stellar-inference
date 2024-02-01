import numpyro
import numpyro.distributions as dist

import numpy as np
import jax.numpy as jnp

from jax import vmap, jit
from typing import Optional
from .star import Star

def decenter(value, loc, scale):
    return (value - loc) / scale

def lognorm_from_norm(*, loc, scale):
    """Returns shape params for lognorm from mu and sigma of norm."""
    var = scale**2
    mu2 = loc**2
    return (
        jnp.log(mu2) - 0.5*jnp.log(mu2 + var),
        jnp.sqrt(jnp.log(1 + var/mu2))
    )


class SingleStarModel:
    def __init__(self, const: Optional[dict]=None, bands: Optional[list]=None):
        self.star = Star(bands=bands, backend="jax")
        self.photometry = False if bands is None else True
        self.const = self._default_const(const=const)

    def _default_const(self, const: Optional[dict]=None) -> dict:
        if const is None:
            const = {}
        const.setdefault("dof", 10)
        # const.setdefault("evol", dict(concentration1=2.0, concentration0=5.0))
        const.setdefault("log_mass", dict(loc=0.0, scale=0.3))
        const.setdefault("M_H", dict(loc=0.0, scale=0.5))
        const.setdefault("log_evol", dict(loc=-0.7, scale=0.4))
        if self.photometry:
            const.setdefault("distance", dict(concentration=3.0, rate=1e-3))
            const.setdefault("scaled_distance", dict(concentration1=3.0, concentration0=1.0))
            const.setdefault("max_distance", 500.0)
            # const.setdefault("distance", 100.)
            # const.setdefault("plx", dict(loc=0.005, scale=1e-5))
            # const.setdefault("Av", dict(loc=1.0, scale=1.0))  # TODO: correlate with distance?
            const.setdefault("Av", 0.0)

        return const

    def sample_star(self) -> dict:
        params = {}
        log_evol = numpyro.sample("log_evol", dist.TruncatedNormal(**self.const["log_evol"], high=0.0))
        params["evol"] = numpyro.deterministic("evol", 10**log_evol)
        # params["evol"] = numpyro.sample("evol", dist.Beta(**self.const["evol"]))
        params["log_mass"] = numpyro.sample(
            "log_mass", 
            dist.TruncatedNormal(**self.const["log_mass"], low=np.log10(0.7), high=np.log10(2.3))
        )
        params["M_H"] = numpyro.sample(
            "M_H", 
            dist.TruncatedNormal(**self.const["M_H"], low=-0.9, high=0.5)
        )

        # params["M_H"] = numpyro.sample("M_H", dist.Uniform(low=-0.9, high=0.5))
        params["Y"] = numpyro.sample("Y", dist.Uniform(low=0.22, high=0.32))
        params["a_MLT"] = numpyro.sample("a_MLT", dist.Uniform(low=1.3, high=2.7))

        if self.photometry:
            # params["distance"] = numpyro.sample("distance", dist.Gamma(**self.const["distance"]))
            scaled_distance = numpyro.sample("scaled_distance", dist.Beta(**self.const["scaled_distance"]))
            params["distance"] = numpyro.deterministic("distance", self.const["max_distance"] * scaled_distance)
            # params["distance"] = self.const["distance"]
            # params["plx"] = numpyro.sample("plx", dist.Normal(**self.const["plx"]))
            # params["Av"] = numpyro.sample("Av", dist.TruncatedNormal(**self.const["Av"], low=0.0, high=6.0))
            params["Av"] = self.const["Av"]
        return params

    def __call__(self, obs: Optional[dict]=None) -> None:
        params = self.sample_star()

        determs = self.star(params)

        for key, value in determs.items():
            numpyro.deterministic(key, value)

        if obs is None:
            return

        for key, value in obs.items():
            if key == "plx":
                numpyro.sample(f"{key}_obs", dist.Normal(determs[key], self.const[key]["scale"]), obs=value)
            else:
                numpyro.sample(f"{key}_obs", dist.StudentT(self.const["dof"], determs[key], self.const[key]["scale"]), obs=value)
            # numpyro.sample(f"{key}_obs", dist.Normal(determs[key], self.const[key]["scale"]), obs=value)


class MultiStarModel(SingleStarModel):
    def __init__(self, num_stars: int, const: Optional[dict]=None, bands: Optional[list]=None):
        super(MultiStarModel, self).__init__(const=const, bands=bands)
        self.num_stars = num_stars

    def sample_star(self) -> dict:
        params = {}
        log_evol = numpyro.sample("log_evol", dist.TruncatedNormal(**self.const["log_evol"], high=0.0))
        params["evol"] = numpyro.deterministic("evol", 10**log_evol)

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
            # params["distance"] = numpyro.sample("distance", dist.Gamma(**self.const["distance"]))
            scaled_distance = numpyro.sample("scaled_distance", dist.Beta(**self.const["scaled_distance"]))
            params["distance"] = numpyro.deterministic("distance", self.const["max_distance"] * scaled_distance)
            # params["distance"] = jnp.broadcast_to(self.const["distance"], (self.num_stars,))
            # params["plx"] = numpyro.sample("plx", dist.LogNormal(*lognorm_from_norm(**self.const["plx"])))
            # params["Av"] = numpyro.sample("Av", dist.TruncatedNormal(**self.const["Av"], low=0.0, high=6.0))
            params["Av"] = jnp.broadcast_to(self.const["Av"], (self.num_stars,))
            # params["Av"] = jnp.zeros((self.num_stars,))

        return params

    def __call__(self, obs: Optional[dict]=None) -> None:

        with numpyro.plate("star", self.num_stars):
            params = self.sample_star()

        determs = vmap(self.star)(params)

        for key, value in determs.items():
            numpyro.deterministic(key, value)

        if obs is None:
            return

        for key, value in obs.items():
            # TODO: add obs mask to allow some unobserved variables
            if key == "plx":
                numpyro.sample(f"{key}_obs", dist.Normal(determs[key], self.const[key]["scale"]), obs=value)
            else:
                numpyro.sample(f"{key}_obs", dist.StudentT(self.const["dof"], determs[key], self.const[key]["scale"]), obs=value)
            # numpyro.sample(f"{key}_obs", dist.Normal(determs[key], self.const[key]["scale"]), obs=value)


class HierarchicalStarModel(MultiStarModel):
    def __init__(self, num_stars: int, const: Optional[dict]=None, bands: Optional[list]=None):
        super(HierarchicalStarModel, self).__init__(num_stars, const=const, bands=bands)
    
    def _default_const(self, const: Optional[dict]=None) -> dict:
        const = super(MultiStarModel, self)._default_const(const=const)

        # Hyperparameters
        const.setdefault("mu_a", dict(low=1.5, high=2.5))

        const.setdefault("Y_0", dict(loc=0.247, scale=0.001))
        # const.setdefault("dY_dZ", dict(loc=1.5, scale=1.0))
        const.setdefault("dY_dZ", dict(low=0.0, high=3.0))
        # const.setdefault("sigma_Y", dict(scale=0.01))
        const.setdefault("sigma_Y", dict(concentration=5.0, rate=0.03))
        # const.setdefault("sigma_Y", dict(loc=-5.3, scale=0.7))

        # const.setdefault("a_1", dict(loc=2.0, scale=0.01))
        # const.setdefault("da_dM", dict(loc=-0.3, scale=0.3))
        # const.setdefault("da_dM", dict(low=-0.5, high=0.5))
        # const.setdefault("sigma_a", dict(scale=0.01))
        const.setdefault("sigma_a", dict(concentration=5.0, rate=0.3))
        # const.setdefault("sigma_a", dict(loc=-3.0, scale=0.7))
        return const

    def sample_population(self):
        hyperparams = {}

        # hyperparams["mu_Y"] = numpyro.sample("mu_Y", dist.Uniform(low=0.22, high=0.32))
        hyperparams["mu_a"] = numpyro.sample("mu_a", dist.Uniform(**self.const["mu_a"]))

        hyperparams["Y_0"] = numpyro.sample("Y_0", dist.Normal(**self.const["Y_0"]))
        # hyperparams["dY_dZ"] = numpyro.sample("dY_dZ", dist.Normal(**self.const["dY_dZ"]))
        hyperparams["dY_dZ"] = numpyro.sample("dY_dZ", dist.Uniform(**self.const["dY_dZ"]))
        # hyperparams["sigma_Y"] = numpyro.sample("sigma_Y", dist.HalfNormal(**self.const["sigma_Y"]))
        hyperparams["sigma_Y"] = numpyro.sample("sigma_Y", dist.InverseGamma(**self.const["sigma_Y"]))
        # hyperparams["sigma_Y"] = numpyro.sample("sigma_Y", dist.LogNormal(**self.const["sigma_Y"]))

        # hyperparams["a_1"] = numpyro.sample("a_1", dist.Normal(**self.const["a_1"]))
        # hyperparams["da_dM"] = numpyro.sample("da_dM", dist.Normal(**self.const["da_dM"]))
        # hyperparams["da_dM"] = numpyro.sample("da_dM", dist.Uniform(**self.const["da_dM"]))
        # hyperparams["sigma_a"] = numpyro.sample("sigma_a", dist.HalfNormal(**self.const["sigma_a"]))
        hyperparams["sigma_a"] = numpyro.sample("sigma_a", dist.InverseGamma(**self.const["sigma_a"]))
        # hyperparams["sigma_a"] = numpyro.sample("sigma_a", dist.LogNormal(**self.const["sigma_a"]))
        return hyperparams

    def sample_star(self, hyperparams: dict) -> dict:
        params = {}
        log_evol = numpyro.sample("log_evol", dist.TruncatedNormal(**self.const["log_evol"], high=0.0))
        params["evol"] = numpyro.deterministic("evol", 10**log_evol)

        params["log_mass"] = log_mass = numpyro.sample(
            "log_mass", 
            dist.TruncatedNormal(**self.const["log_mass"], low=np.log10(0.7), high=np.log10(2.3))
        )
        params["M_H"] = mh = numpyro.sample(
            "M_H", 
            dist.TruncatedNormal(**self.const["M_H"], low=-0.9, high=0.4)
        )

        # ones = jnp.ones(self.num_stars)
        # params["Y"] = numpyro.deterministic("Y", hyperparams["mu_Y"] * ones)
        # params["a_MLT"] = numpyro.deterministic("a_MLT", hyperparams["mu_a"] * ones)

        # mu_y = hyperparams["mu_Y"]
        # y0 = 0.247
        y0 = hyperparams["Y_0"]
        f = hyperparams["dY_dZ"] / (10**-(mh + self.star.log_zx_sun) + 1)
        mu_y = (y0 + f) / (1 + f)
        sigma_y = hyperparams["sigma_Y"]
        # low, high = decenter(0.22, mu_y, sigma_y), decenter(0.32, mu_y, sigma_y)
        # y_decentered = numpyro.sample("Y_decentered", dist.TruncatedNormal(low=low, high=high))
        y_decentered = numpyro.sample("Y_decentered", dist.Normal())
        params["Y"] = numpyro.deterministic("Y", mu_y + sigma_y * y_decentered)
        # TODO: reparam
        # params["Y"] = numpyro.sample("Y", dist.TruncatedNormal(mu_y, hyperparams["sigma_Y"], low=0.22, high=0.32))

        mu_a = hyperparams["mu_a"]
        # a1 = 2.0
        # a1 = hyperparams["a_1"]
        # mu_a = a1 + hyperparams["da_dM"] * (10**log_mass - 1.0)
        sigma_a = hyperparams["sigma_a"]
        # low, high = decenter(1.3, mu_a, sigma_a), decenter(2.7, mu_a, sigma_a)
        # a_decentered = numpyro.sample("a_decentered", dist.TruncatedNormal(low=low, high=high))
        a_decentered = numpyro.sample("a_decentered", dist.Normal())
        params["a_MLT"] = numpyro.deterministic("a_MLT", mu_a + sigma_a * a_decentered)
        # TODO: reparam
        # params["a_MLT"] = numpyro.sample("a_MLT", dist.TruncatedNormal(mu_a, hyperparams["sigma_a"], low=1.3, high=2.7))

        if self.photometry:
            # params["distance"] = numpyro.sample("distance", dist.Gamma(**self.const["distance"]))
            scaled_distance = numpyro.sample("scaled_distance", dist.Beta(**self.const["scaled_distance"]))
            params["distance"] = numpyro.deterministic("distance", self.const["max_distance"] * scaled_distance)
            # params["distance"] = jnp.broadcast_to(self.const["distance"], (self.num_stars,))
            # params["Av"] = numpyro.sample("Av", dist.TruncatedNormal(**self.const["Av"], low=0.0, high=6.0))
            params["Av"] = jnp.broadcast_to(self.const["Av"], (self.num_stars,))

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
            if key == "plx":
                numpyro.sample(f"{key}_obs", dist.Normal(determs[key], self.const[key]["scale"]), obs=value)
            else:
                numpyro.sample(f"{key}_obs", dist.StudentT(self.const["dof"], determs[key], self.const[key]["scale"]), obs=value)
            # numpyro.sample(f"{key}_obs", dist.Normal(determs[key], self.const[key]["scale"]), obs=value)
