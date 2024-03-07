import numpyro, os, json
import numpyro.distributions as dist
import jax.numpy as jnp

from math import log10, log
from jax import vmap
from typing import Optional
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

ln10 = log(10)
grav_constant = 6.6743e-8  # in cgs units


class SingleStarModel:
    log_m_sun = log10(1.988410) + 33
    log_r_sun = log10(6.957) + 10
    log_g_sun = log10(grav_constant) + log_m_sun - 2 * log_r_sun
    log_l_sun = log10(3.828) + 33
    log_teff_sun = log10(5772.0)
    log_zx_sun = log10(0.0181)
    log_numax_sun = log10(3090.0)
    outputs = [
        "log_age", "log_Teff", "log_R", "log_Dnu",
        "log_L", "log_g", "log_numax"
    ]

    def __init__(self, observables: list, const: Optional[dict]=None, kind: str="diag"):
        emulator = Emulator(backend="jax")
        const = self._default_const(const=const)

        if kind not in ["none", "diag", "full"]:
            raise ValueError("Kind must be one of 'none', 'diag', or 'full'.")
        

        A = jnp.vstack(
            [
                jnp.eye(4),  # log(age), log(Teff), log(R), log(Dnu)
                [0.0, 4.0, 2.0, 0.0],  # log(L)
                [0.0, 0.0, -2.0, 0.0],  # log(g)
                [0.0, -0.5, -2.0, 0.0],  # log(numax)
            ]
        )

        B = jnp.vstack(
            [
                jnp.zeros((5, 7)),
                [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0],  # log(g)
                [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0],  # log(numax)
            ]
        )

        c = jnp.array(
            [-9.0, 0.0, 0.0, 0.0, -4.0 * self.log_teff_sun, self.log_g_sun,
             self.log_numax_sun + 0.5 * self.log_teff_sun]
        )

        covariance = A @ const["delta"]["cov"] @ A.T
        variance = jnp.diag(covariance)

        observable_indices = jnp.array(
            [self.outputs.index(param) for param in observables]
        )

        self.emulator = emulator
        self.const = const
        self.kind = kind
        self.A = A
        self.B = B
        self.c = c
        self.covariance = covariance
        self.variance = variance
        self.observables = observables
        self.observable_indices = observable_indices

    def _emulator_precision(self) -> dict:
        with open(os.path.join(PACKAGEDIR, "data/emulator_error.json"), "r") as file:
            params = json.loads(file.read())
        df = jnp.array(params["df"])
        loc = jnp.array(params["mu"])
        scale = jnp.array(jnp.sqrt(jnp.array(params["theta"])))
        scale_tril = scale[:, None] * jnp.array(params["L_omega"])
        cov = scale_tril @ scale_tril.T
        return dict(df=df, loc=loc, cov=cov)

    def _default_const(self, const: Optional[dict]=None) -> dict:
        if const is None:
            const = {}
        const.setdefault("delta", self._emulator_precision())
        const.setdefault("log_evol", dict(loc=-0.7, scale=0.4, high=0.0))
        const.setdefault("log_mass", dict(loc=0.0, scale=0.3, low=log10(0.7), high=log10(2.3)))
        const.setdefault("M_H", dict(loc=0.0, scale=0.5, low=-0.85, high=0.45))
        const.setdefault("Y", dict(low=0.23, high=0.31))
        const.setdefault("a_MLT", dict(low=1.5, high=2.5))
        return const

    def parameters(self) -> jnp.ndarray:
        log_evol = numpyro.sample("log_evol", dist.TruncatedNormal(**self.const["log_evol"]))
        evol = numpyro.deterministic("evol", jnp.power(10.0, log_evol))

        log_mass = numpyro.sample(
            "log_mass",
            dist.TruncatedNormal(**self.const["log_mass"])
        )
        mass = numpyro.deterministic("mass", jnp.power(10.0, log_mass))

        mh = numpyro.sample(
            "M_H", 
            dist.TruncatedNormal(**self.const["M_H"])
        )

        y = numpyro.sample("Y", dist.Uniform(**self.const["Y"]))

        a_mlt = numpyro.sample("a_MLT", dist.Uniform(**self.const["a_MLT"]))

        df = self.const["delta"]["df"]
        scaled_precision = numpyro.sample("scaled_precision", dist.Gamma(df/2, df/2))

        return jnp.stack(
            [evol, mass, mh, y, a_mlt, log_mass, scaled_precision],
            axis=-1
        )

    def predict(self, params: jnp.ndarray) -> tuple[jnp.ndarray]:
        outputs = self.emulator(params[:5])
        loc = (
            jnp.matmul(self.A, outputs + self.const["delta"]["loc"])
            + jnp.matmul(self.B, params)
            + self.c
        )

        if self.kind == "none":
            return loc,

        if self.kind == "diag":
            return loc, self.variance / params[-1]

        if self.kind == "full":
            return loc, self.covariance / params[-1]

    def likelihood(self, mean: jnp.ndarray, covariance: jnp.ndarray,
                   obs: Optional[jnp.ndarray]=None, diag: Optional[jnp.ndarray]=None):
        
        if obs is None:
            obs_indices = jnp.arange(len(self.outputs))
        else:
            obs_indices = self.observable_indices

        if diag is None:
            diag = jnp.zeros(())

        mean = mean[..., obs_indices]

        if self.kind == "diag":
            var = covariance[..., obs_indices]
            return numpyro.sample("y", dist.Normal(mean, jnp.sqrt(var + diag)), obs=obs)

        if self.kind == "full":
            cov = covariance[..., obs_indices, obs_indices[:, None]]
            cov += diag[..., None] * jnp.identity(mean.shape[0])
            return numpyro.sample("y", dist.MultivariateNormal(mean, cov), obs=obs)

    def __call__(self, obs: Optional[jnp.ndarray]=None, diag: Optional[jnp.ndarray]=None) -> None:
        params = self.parameters()
        mean, covariance = self.predict(params)            
        y = self.likelihood(mean, covariance, obs=obs, diag=diag)


class MultiStarModel(SingleStarModel):
    def __init__(self, num_stars: int, observables: list, const: Optional[dict]=None, kind: str="diag"):
        if num_stars < 2:
            raise ValueError("Variable num_stars must be greater than 1.")
        super(MultiStarModel, self).__init__(observables, const=const, kind=kind)
        self.num_stars = num_stars

    def __call__(self, obs: Optional[jnp.ndarray]=None, diag=None) -> None:
        with numpyro.plate("star", self.num_stars):
            params = self.parameters()
        mean, covariance = vmap(self.predict)(params)
        y = self.likelihood(mean, covariance, obs=obs, diag=diag)


class HierarchicalStarModel(MultiStarModel):
    def __init__(self, num_stars: int, observables: list, const: Optional[dict]=None, kind: str="diag"):
        super(HierarchicalStarModel, self).__init__(num_stars, observables, const=const, kind=kind)

    def _default_const(self, const: Optional[dict]=None) -> dict:
        const = super(MultiStarModel, self)._default_const(const=const)

        # Hyperparameters
        const.setdefault("Y_0", dict(loc=0.247, scale=0.001))
        const.setdefault("dY_dZ", dict(loc=1.5, scale=0.5, low=0.0, high=3.0))
        const.setdefault("precision_Y", dict(concentration=2.0, rate=2e-4/3))
        const.setdefault("mu_a", dict(loc=2.0, scale=0.1, low=1.3, high=2.7))
        const.setdefault("precision_a", dict(concentration=2.0, rate=2e-2/3))
        return const

    def hyperparameters(self) -> jnp.ndarray:

        y0 = numpyro.sample("Y_0", dist.Normal(**self.const["Y_0"]))
        dydz = numpyro.sample("dY_dZ", dist.TruncatedNormal(**self.const["dY_dZ"]))        
        precision_y = numpyro.sample("precision_Y", dist.Gamma(**self.const["precision_Y"]))
        sigma_y = numpyro.deterministic("sigma_Y", precision_y**-0.5)

        mu_a = numpyro.sample("mu_a", dist.TruncatedNormal(**self.const["mu_a"]))
        precision_a = numpyro.sample("precision_a", dist.Gamma(**self.const["precision_a"]))
        sigma_a = numpyro.deterministic("sigma_a", precision_a**-0.5)

        return jnp.stack(
            [y0, dydz, sigma_y, mu_a, sigma_a],
            axis=-1
        )

    def parameters(self, hyperparams: jnp.ndarray) -> jnp.ndarray:

        df = self.const["delta"]["df"]
        scaled_precision = numpyro.sample("scaled_precision", dist.Gamma(df/2, df/2))

        log_evol = numpyro.sample("log_evol", dist.TruncatedNormal(**self.const["log_evol"]))
        evol = numpyro.deterministic("evol", 10**log_evol)

        log_mass = numpyro.sample(
            "log_mass", 
            dist.TruncatedNormal(**self.const["log_mass"])
        )
        mass = numpyro.deterministic("mass", jnp.power(10.0, log_mass))

        mh = numpyro.sample(
            "M_H", 
            dist.TruncatedNormal(**self.const["M_H"])
        )

        y0 = hyperparams[0]
        f = hyperparams[1] / (10**-(mh + self.log_zx_sun) + 1)
        mu_y = (y0 + f) / (1 + f)
        sigma_y = hyperparams[2]
        # low, high = decenter(0.22, mu_y, sigma_y), decenter(0.32, mu_y, sigma_y)
        # y_decentered = numpyro.sample("Y_decentered", dist.TruncatedNormal(low=low, high=high))
        y_decentered = numpyro.sample("Y_decentered", dist.Normal())
        y = numpyro.deterministic("Y", mu_y + sigma_y * y_decentered)
        # params["Y"] = numpyro.sample("Y", dist.TruncatedNormal(mu_y, hyperparams["sigma_Y"], low=0.22, high=0.32))

        mu_a = hyperparams[3]
        sigma_a = hyperparams[4]
        # low, high = decenter(1.3, mu_a, sigma_a), decenter(2.7, mu_a, sigma_a)
        # a_decentered = numpyro.sample("a_decentered", dist.TruncatedNormal(low=low, high=high))
        a_decentered = numpyro.sample("a_decentered", dist.Normal())
        a_mlt = numpyro.deterministic("a_MLT", mu_a + sigma_a * a_decentered)
        # params["a_MLT"] = numpyro.sample("a_MLT", dist.TruncatedNormal(mu_a, hyperparams["sigma_a"], low=1.3, high=2.7))

        return jnp.stack(
            [evol, mass, mh, y, a_mlt, log_mass, scaled_precision],
            axis=-1
        )

    def __call__(self, obs: Optional[jnp.ndarray]=None, diag: Optional[jnp.ndarray]=None) -> None:
        hyperparams = self.hyperparameters()
        with numpyro.plate("star", self.num_stars):
            params = self.parameters(hyperparams)
        mean, covariance = vmap(self.predict)(params)
        y = self.likelihood(mean, covariance, obs=obs, diag=diag)
