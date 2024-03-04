import numpyro, os, json
import numpyro.distributions as dist

import numpy as np
import jax.numpy as jnp

from jax import vmap, jit
from jax.scipy.special import expit
from typing import Optional
from .star import Star
from .emulator import Emulator
from . import PACKAGEDIR


def decenter(value, loc, scale):
    return (value - loc) / scale

# def lognorm_from_norm(*, loc, scale):
#     """Returns shape params for lognorm from mu and sigma of norm."""
#     var = scale**2
#     mu2 = loc**2
#     return (
#         jnp.log(mu2) - 0.5*jnp.log(mu2 + var),
#         jnp.sqrt(jnp.log(1 + var/mu2))
#     )

def lognorm_from_norm(mean, variance):
    """Returns mean and variance of lognorm."""
    mean_squared = mean**2
    return (
        jnp.log(mean_squared) - 0.5*jnp.log(mean_squared + variance),
        jnp.log(1 + variance/mean_squared)
    )
    # return (
    #     jnp.log(mean),
    #     variance / mean**2,
    # )

ln10 = jnp.log(10)
grav_constant = 6.6743e-8  # in cgs units


class SingleStarModel:
    log_m_sun = np.log10(1.988410) + 33
    log_r_sun = np.log10(6.957) + 10
    log_g_sun = np.log10(grav_constant) + log_m_sun - 2 * log_r_sun
    log_l_sun = np.log10(3.828) + 33
    log_teff_sun = np.log10(5772.0)
    bol_mag_sun = 4.75
    log_zx_sun = np.log10(0.0181)
    log_numax_sun = np.log10(3090.0) 
    output_params = np.array([
        "log_age", "log_Teff", "log_R", "log_Dnu",
        "log_L", "log_g", "log_numax"
    ])

    def __init__(self, const: Optional[dict]=None, obs=None):
        # self.star = Star(bands=bands, backend="jax")
        self.emulator = Emulator(backend="jax")
        self.const = self._default_const(const=const)

        A = np.vstack(
            [
                np.eye(4),  # log(age), log(Teff), log(R), log(Dnu)
                [0.0, 4.0, 2.0, 0.0],  # log(L)
                [0.0, 0.0, -2.0, 0.0],  # log(g)
                [0.0, -0.5, -2.0, 0.0],  # log(numax)
            ]
        )

        B = np.vstack(
            [
                np.zeros((5, 7)),
                [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0],  # log(g)
                [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0],  # log(numax)
            ]
        )

        c = np.array(
            [-9.0, 0.0, 0.0, 0.0, -4.0 * self.log_teff_sun, self.log_g_sun,
             self.log_numax_sun - self.log_g_sun + 0.5 * self.log_teff_sun]
        )

        covariance = A @ self.const["precision"]["cov"] @ A.T
        variance = np.diag(covariance)

        self.A = A
        self.B = B
        self.c = c
        self.covariance = covariance
        self.variance = variance
        self.output_indices = jnp.arange(self.output_params.shape[0])

    def _emulator_precision(self):
        with open(os.path.join(PACKAGEDIR, "data/emulator_error.json"), "r") as file:
            params = json.loads(file.read())
        precision = {}
        precision["df"] = jnp.array(params["df"])
        precision["loc"] = jnp.array(params["mu"])
        precision["scale"] = scale =  jnp.array(np.sqrt(params["theta"]))
        precision["scale_tril"] = scale_tril = scale[:, None] * jnp.array(params["L_omega"])
        precision["cov"] = scale_tril @ scale_tril.T
        # precision["scale_tril"] = scale[:, None] * jnp.array(params["L_omega"])
        return precision

    def _default_const(self, const: Optional[dict]=None) -> dict:
        if const is None:
            const = {}
        # const.setdefault("dof", 4.184734344482422)
        # const.setdefault("evol", dict(concentration1=2.0, concentration0=5.0))
        const.setdefault("precision", self._emulator_precision())
        const.setdefault("log_mass", dict(loc=0.0, scale=0.3))
        const.setdefault("M_H", dict(loc=0.0, scale=0.5))
        const.setdefault("log_evol", dict(loc=-0.7, scale=0.4))
        # const.setdefault("Teff", dict(scale=0.0))
        # const.setdefault("L", dict(scale=0.0))
        return const

    def parameters(self) -> jnp.ndarray:

        log_evol = numpyro.sample("log_evol", dist.TruncatedNormal(**self.const["log_evol"], high=0.0))
        evol = numpyro.deterministic("evol", jnp.power(10.0, log_evol))

        log_mass = numpyro.sample(
            "log_mass",
            dist.TruncatedNormal(**self.const["log_mass"], low=np.log10(0.7), high=np.log10(2.3))
        )
        mass = numpyro.deterministic("mass", jnp.power(10.0, log_mass))

        mh = numpyro.sample(
            "M_H", 
            dist.TruncatedNormal(**self.const["M_H"], low=-0.9, high=0.5)
        )

        y = numpyro.sample("Y", dist.Uniform(low=0.22, high=0.32))
        a_mlt = numpyro.sample("a_MLT", dist.Uniform(low=1.3, high=2.7))

        df = self.const["precision"]["df"]
        scaled_precision = numpyro.sample("scaled_precision", dist.Gamma(df/2, df/2))

        return jnp.stack(
            [evol, mass, mh, y, a_mlt, log_mass, scaled_precision],
            axis=-1
        )

    def predict(self, params: jnp.ndarray, kind: str="diag") -> tuple[jnp.ndarray]:
        outputs = self.emulator(params[:5])
        loc = (
            jnp.matmul(self.A, outputs + self.const["precision"]["loc"])
            + jnp.matmul(self.B, params)
            + self.c
        )

        if kind == "none":
            return loc,

        if kind == "diag":
            return loc, self.variance / params[-1]

        if kind == "full":
            return loc, self.covariance / params[-1]

        raise ValueError("Kind must be one of 'none', 'diag', or 'full'.")

    def deterministics(self, y, indices):
        for i, key in enumerate(self.output_params[indices]):
            numpyro.deterministic(key, y[..., i])

    def likelihood(self, mean, covariance, obs=None, obs_indices=None, diag=None, kind="diag"):
        if obs_indices is None:
            obs_indices = self.output_indices
        
        if diag is None:
            diag = jnp.zeros(self.output_params.shape[0])

        mean = mean[..., obs_indices]
        
        if kind == "diag":
            var = covariance[..., obs_indices]
            return numpyro.sample("y", dist.Normal(mean, jnp.sqrt(var + diag)), obs=obs)
        
        if kind == "full":
            cov = covariance[..., obs_indices, obs_indices[:, None]]
            cov += diag[..., None] * jnp.identity(mean.shape[0])
            return numpyro.sample("y", dist.MultivariateNormal(mean, cov), obs=obs)

    def __call__(self, obs: Optional[jnp.ndarray]=None, obs_indices=None,
                 diag=None, kind="diag") -> None:
        
        params = self.parameters()
        mean, covariance = self.predict(params, kind)
        self.likelihood(mean, covariance, obs=obs, obs_indices=obs_indices,
                        diag=diag, kind=kind)

        # if obs is None:
            # self.deterministics(y, obs_indices)


class MultiStarModel(SingleStarModel):
    def __init__(self, num_stars: int, const: Optional[dict]=None):
        if num_stars < 2:
            raise ValueError("Variable num_stars must be greater than 1.")
        super(MultiStarModel, self).__init__(const=const)
        self.num_stars = num_stars

    def __call__(self, obs: Optional[jnp.ndarray]=None, obs_indices=None,
                 diag=None, kind="diag") -> None:
        with numpyro.plate("star", self.num_stars):
            params = self.parameters()
        mean, covariance = vmap(self.predict, in_axes=(0, None))(params, kind)
        self.likelihood(mean, covariance, obs=obs, obs_indices=obs_indices,
                        diag=diag, kind=kind)


class HierarchicalStarModel(MultiStarModel):
    def __init__(self, num_stars: int, const: Optional[dict]=None):
        super(HierarchicalStarModel, self).__init__(num_stars, const=const)
    
    def _default_const(self, const: Optional[dict]=None) -> dict:
        const = super(MultiStarModel, self)._default_const(const=const)

        # Hyperparameters
        const.setdefault("Y_0", dict(loc=0.247, scale=0.001))
        # const.setdefault("dY_dZ", dict(low=0.0, high=3.0))
        const.setdefault("dY_dZ", dict(loc=1.5, scale=0.5, low=0.0, high=3.0))
        const.setdefault("precision_Y", dict(concentration=2.0, rate=2e-4/3))

        # const.setdefault("mu_a", dict(low=1.5, high=2.5))
        const.setdefault("mu_a", dict(loc=2.0, scale=0.1, low=1.3, high=2.7))
        const.setdefault("precision_a", dict(concentration=2.0, rate=2e-2/3))
        return const

    def hyperparamters(self):
        hyperparams = {}

        hyperparams["Y_0"] = numpyro.sample("Y_0", dist.Normal(**self.const["Y_0"]))
        # hyperparams["dY_dZ"] = numpyro.sample("dY_dZ", dist.Uniform(**self.const["dY_dZ"]))
        hyperparams["dY_dZ"] = numpyro.sample("dY_dZ", dist.TruncatedNormal(**self.const["dY_dZ"]))        
        precision_Y = numpyro.sample("precision_Y", dist.Gamma(**self.const["precision_Y"]))
        hyperparams["sigma_Y"] = numpyro.deterministic("sigma_Y", precision_Y**-0.5)

        # hyperparams["mu_a"] = numpyro.sample("mu_a", dist.Uniform(**self.const["mu_a"]))
        hyperparams["mu_a"] = numpyro.sample("mu_a", dist.TruncatedNormal(**self.const["mu_a"]))
        precision_a = numpyro.sample("precision_a", dist.Gamma(**self.const["precision_a"]))
        hyperparams["sigma_a"] = numpyro.deterministic("sigma_a", precision_a**-0.5)

        return hyperparams

    def parameters(self, hyperparams: dict) -> dict:
        params = {}

        # df = self.const["precision"]["df"]
        # with numpyro.plate("outputs", 4):
            # params["scaled_precision"] = numpyro.sample("scaled_precision", dist.Gamma(df/2, df/2))
        df = self.const["precision"]["df"]
        params["scaled_precision"] = numpyro.sample("scaled_precision", dist.Gamma(df/2, df/2))

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

        y0 = hyperparams["Y_0"]
        f = hyperparams["dY_dZ"] / (10**-(mh + self.log_zx_sun) + 1)
        mu_y = (y0 + f) / (1 + f)
        sigma_y = hyperparams["sigma_Y"]
        # low, high = decenter(0.22, mu_y, sigma_y), decenter(0.32, mu_y, sigma_y)
        # y_decentered = numpyro.sample("Y_decentered", dist.TruncatedNormal(low=low, high=high))
        y_decentered = numpyro.sample("Y_decentered", dist.Normal())
        params["Y"] = numpyro.deterministic("Y", mu_y + sigma_y * y_decentered)
        # params["Y"] = numpyro.sample("Y", dist.TruncatedNormal(mu_y, hyperparams["sigma_Y"], low=0.22, high=0.32))

        mu_a = hyperparams["mu_a"]
        sigma_a = hyperparams["sigma_a"]
        # low, high = decenter(1.3, mu_a, sigma_a), decenter(2.7, mu_a, sigma_a)
        # a_decentered = numpyro.sample("a_decentered", dist.TruncatedNormal(low=low, high=high))
        a_decentered = numpyro.sample("a_decentered", dist.Normal())
        params["a_MLT"] = numpyro.deterministic("a_MLT", mu_a + sigma_a * a_decentered)
        # params["a_MLT"] = numpyro.sample("a_MLT", dist.TruncatedNormal(mu_a, hyperparams["sigma_a"], low=1.3, high=2.7))

        return params

    def __call__(self, obs: Optional[dict]=None) -> None:
        hyperparams = self.hyperparamters()
        with numpyro.plate("star", self.num_stars):
            params = self.parameters(hyperparams)
        determs = self.deterministics(params)
        self.likelihood(params, determs, obs=obs)
