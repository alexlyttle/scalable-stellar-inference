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

    def __init__(self, const: Optional[dict]=None):
        # self.star = Star(bands=bands, backend="jax")
        self.emulator = Emulator(backend="jax")
        self.const = self._default_const(const=const)

        # Attempt at MV likelihood
        obs_coeff = np.array(
            [[0.0, 1.0, 0.0, 0.0],
            [0.0, 4.0, 2.0, 0.0]]
        ) * ln10  # from log10 to ln space
        self.covariance = obs_coeff @ self.const["precision"]["cov"] @ obs_coeff.T
        self.variance = jnp.diag(self.covariance)
        # scale_tril = obs_coeff @ self.const["precision"]["scale_tril"]
        # self.obs_coeff = obs_coeff
        # self.covariance = scale_tril @ scale_tril.T
        self.obs_var = jnp.stack(jnp.broadcast_arrays(
            self.const["Teff"]["scale"], 
            self.const["L"]["scale"]
            ), -1)**2
        # self.obs_cov = jnp.diag(
        #     jnp.stack([self.const["Teff"]["scale"], self.const["L"]["scale"]], -1)**2
        # )
        # input_transformation = jnp.zeros((4, 5))
        # input_transformation[2, ]
        # output_transformation = jnp.array(
        #     [
        #         [1., 0., 0., 0.],  # log(age/Gyr)
        #         [0., 4., 2., 0.],  # log(L)
        #         [0., 0., - 2., 0.],  # log(g)
        #         [0., - 0.5, - 2., ]  # log(numax)
        #     ]
        # )
        # self.output_transformation = jnp.vstack([jnp.eye(4), output_transformation])
        # transformation_offset = jnp.array(
        #     [- 9., - 4. * self.log_teff_sun, self.log_g_sun, self.log_numax_sun - self.log_g_sun + 0.5 * self.log_teff_sun]
        # )

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

    def parameters(self) -> dict:
        params = {}

        df = self.const["precision"]["df"]
        params["scaled_precision"] = numpyro.sample("scaled_precision", dist.Gamma(df/2, df/2))

        log_evol = numpyro.sample("log_evol", dist.TruncatedNormal(**self.const["log_evol"], high=0.0))
        params["evol"] = numpyro.deterministic("evol", jnp.power(10.0, log_evol))

        log_mass = numpyro.sample(
            "log_mass",
            dist.TruncatedNormal(**self.const["log_mass"], low=np.log10(0.7), high=np.log10(2.3))
        )
        params["mass"] = numpyro.deterministic("mass", jnp.power(10.0, log_mass))

        params["M_H"] = numpyro.sample(
            "M_H", 
            dist.TruncatedNormal(**self.const["M_H"], low=-0.9, high=0.5)
        )

        params["Y"] = numpyro.sample("Y", dist.Uniform(low=0.22, high=0.32))
        params["a_MLT"] = numpyro.sample("a_MLT", dist.Uniform(low=1.3, high=2.7))

        return params

    def emulate(self, params: dict) -> jnp.ndarray:
        inputs = jnp.stack(
            [params["evol"], params["mass"], params["M_H"], params["Y"], params["a_MLT"]], 
            axis=-1
        )
        outputs = self.emulator(inputs).squeeze()
        return outputs + self.const["precision"]["loc"]  # correct outputs

    # def transform_outputs(self, inputs, outputs):
    #     return jnp.matmul(self.input_transformation, inputs) \
    #         + jnp.matmul(self.output_transformation, outputs) \
    #         + self.transformation_offset

    def deterministics(self, params: dict) -> dict:
        determs = {}

        # scale = self.const["precision"]["scale"]
        # determs["variance"] = scale**2 / params["scaled_precision"].T  # variance on emulator outputs
        outputs = self.emulate(params)
        # Emulate
        # loc = self.emulate(params)
        # scale = self.const["precision"]["scale"]
        # outputs = loc + scale / params["scaled_precision"][..., None]**0.5 * params["outputs_decentered"]

        log_mass = params["log_mass"]

        log_age = numpyro.deterministic("log_age", outputs[..., 0] - 9)  # log(age) in Gyr
        determs["log_Teff"] = log_teff = numpyro.deterministic("log_Teff", outputs[..., 1])
        log_rad = numpyro.deterministic("log_R", outputs[..., 2])
        log_dnu = numpyro.deterministic("log_Dnu", outputs[..., 3])
        determs["log_L"] = log_lum = numpyro.deterministic("log_L", 2 * log_rad + 4 * (log_teff - self.log_teff_sun)) 

        determs["log_g"] = log_g = numpyro.deterministic("log_g", log_mass - 2 * log_rad + self.log_g_sun)
        determs["Teff"] = numpyro.deterministic("Teff", 10**log_teff)
        determs["L"] = numpyro.deterministic("L", 10**log_lum)
        
        log_numax = numpyro.deterministic("log_numax", self.log_numax_sun + log_g - self.log_g_sun - 0.5 * (log_teff - self.log_teff_sun))
        determs["numax"] = numpyro.deterministic("numax", 10**log_numax)
        determs["Dnu"] = numpyro.deterministic("Dnu", 10**log_dnu)

        numpyro.deterministic("mass", 10**log_mass)
        numpyro.deterministic("R", 10**log_rad)
        numpyro.deterministic("age", 10**log_age)

        return determs

    def likelihood(self, params, determs: dict, obs: Optional[dict]=None) -> None:
        # Variance of neural network outputs
        # scale = self.const["precision"]["scale"]
        
        # 1
        # covariance = self.const["precision"]["cov"] / params["scaled_precision"][..., None, None]
        # variance = jnp.diagonal(covariance, axis1=-2, axis2=-1)
        
        # 2
        # variance = scale**2 / scaled_precision  # variance on emulator outputs
        # variance = determs["variance"]

        if obs is None:
            return

        # 1
        # def sample_observable(key, var):
        #     var += self.const[key]["scale"]**2
        #     numpyro.sample(f"{key}_obs", dist.Normal(determs[key], var**0.5), obs=obs[key])

        # if (key := "Teff") in obs:
        #     sample_observable(key, variance[..., 1] * ln10**2 * determs[key]**2)

        # if (key := "L") in obs:
        #     var = (16 * variance[..., 1] + 4 * variance[..., 2] + 16 * covariance[..., 1, 2]) * ln10**2 * determs[key]**2
        #     sample_observable(key, var)
        
        # if (key := "Dnu") in obs:
        #     sample_observable(key, variance[..., 3] * ln10**2 * determs[key]**2)

        # if (key := "log_g") in obs:
        #     sample_observable(key, 4 * variance[..., 2])
        
        # if (key := "numax") in obs:
        #     var = (0.25 * variance[..., 1] + 4 * variance[..., 2] + 2 * covariance[..., 1, 2]) * ln10**2 * determs[key]**2
        #     sample_observable(key, var)

        # 3 Attempt at mutlivariate likelihood
        obs_mean, obs_variance = lognorm_from_norm(
            jnp.stack([obs["Teff"], obs["L"]], axis=-1),
            self.obs_var,
        )
        mean = ln10 * jnp.stack([determs["log_Teff"], determs["log_L"]], -1)  # natural logarithm
        
        # Full likelihood
        # covariance_matrix = self.covariance / params["scaled_precision"][..., None, None]
        # covariance_matrix += obs_variance[..., None] * jnp.identity(obs_variance.shape[-1])
        # numpyro.sample("obs", dist.MultivariateNormal(mean, covariance_matrix=covariance_matrix), obs=obs_mean)
        
        # Diagonalised likelihood
        variance = self.variance / params["scaled_precision"][..., None] + obs_variance
        numpyro.sample("obs", dist.Normal(mean, jnp.sqrt(variance)), obs=obs_mean)

        # Convert from log to exp?
        # covariance = self.covariance / params["scaled_precision"][..., None, None]
        # variance = jnp.diagonal(covariance, axis1=-1, axis2=-2)
        # mu = jnp.stack([determs["Teff"], determs["L"]], -1) * jnp.exp(0.5 * variance)
        # cov = mu[..., None] * jnp.swapaxes(mu[..., None], -1, -2) * (jnp.exp(covariance) - 1) \
        #     + self.obs_var[..., None] * jnp.eye(self.obs_var.shape[-1])
        # numpyro.sample("obs", dist.MultivariateNormal(mu, cov), obs=obs)
        # numpyro.sample("obs", dist.Normal(mu, jnp.diag(cov)**0.5), obs=obs)

    def __call__(self, obs: Optional[dict]=None) -> None:
        params = self.parameters()
        determs = self.deterministics(params)
        self.likelihood(params, determs, obs=obs)
        # for key, determ in determs.items():
            # numpyro.deterministic(key, determ)


class MultiStarModel(SingleStarModel):
    def __init__(self, num_stars: int, const: Optional[dict]=None):
        if num_stars < 2:
            raise ValueError("Variable num_stars must be greater than 1.")
        super(MultiStarModel, self).__init__(const=const)
        self.num_stars = num_stars

    def __call__(self, obs: Optional[dict]=None) -> None:
        with numpyro.plate("star", self.num_stars):
            params = self.parameters()
        determs = self.deterministics(params)
        # determs = vmap(self.deterministics)(params)
        self.likelihood(params, determs, obs=obs)
        # for key, determ in determs.items():
            # numpyro.deterministic(key, determ)


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
