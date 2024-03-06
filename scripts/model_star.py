import os, logging, argparse, numpyro
import numpy as np
import pandas as pd
import arviz as az
import xarray as xa

from corner import corner
from jax import random
from numpyro.infer import MCMC, NUTS, Predictive, init_to_median, init_to_uniform

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s %(levelname)s: %(message)s')
ch.setFormatter(formatter)
logger.addHandler(ch)

observables = ["log_Teff", "log_L"]
init_strategy = init_to_uniform
target_accept_prob = 0.8
kind = "diag"

# numpyro.enable_x64()
# numpyro.set_host_device_count(num_chains)

stat_funcs = {
    "lower": lambda x: np.quantile(x, .16),
    "median": lambda x: np.quantile(x, .5),
    "upper": lambda x: np.quantile(x, .84),
}

def validate_outdir(output_directory: str=None):
    if output_directory is None:
        output_directory = os.getcwd()
    if not os.path.exists(output_directory):
        logger.info(f"Creating directory '{output_directory}'.")
        os.makedirs(output_directory)
    elif not os.path.isdir(output_directory):
        raise ValueError("Path '{output_directory}' is not a directory.")
    return output_directory

def load_obs(path: str, star: int) -> dict:
    logger.debug(f"Loading observables from '{path}'.")
    obs = pd.read_csv(path, index_col=0)
    return obs.loc[star].to_dict()

def load_truths(path: str, star: int) -> dict:
    logger.debug(f"Loading truths from '{path}'.")
    truths = pd.read_csv(path, index_col=0)
    return truths.loc[star].to_dict()

def init_model(obs: dict, Model: object) -> callable:
    # Create dictionary of model constants
    const = {
        "M_H": dict(loc=obs["M_H"], scale=obs["sigma_M_H"], low=-0.85, high=0.45),
    }
    logger.debug(f"Initialising model with observables={observables} and const={const}.")
    return Model(observables, const=const, kind=kind)

def init_mcmc(model, num_warmup: int, num_samples: int, num_chains: int):
    logger.debug(f"Initialising NUTS with model={model} and init_strategy={init_strategy}.")
    sampler = NUTS(model, init_strategy=init_strategy, target_accept_prob=target_accept_prob, find_heuristic_step_size=True)

    logger.debug(f"Initialising NUTS with sampler={sampler}, num_warmup={num_warmup}, num_samples={num_samples}, and num_chains={num_chains}.")
    return MCMC(sampler, num_warmup=num_warmup, num_samples=num_samples, num_chains=num_chains)

def run_mcmc(key, mcmc: MCMC, obs: dict) -> MCMC:
    logger.debug(f"Creating observed y and diag arrays.")
    y = np.stack([obs[key] for key in observables], axis=-1)
    diag = np.stack([obs[f"sigma_{key}"] for key in observables], axis=-1)**2

    logger.debug(f"Random key={key}.")
    logger.debug(f"Observations y={y}.")
    logger.debug(f"Diagonal diag={diag}.")

    mcmc.run(key, obs=y, diag=diag)
    logger.info("Run complete.")
    logger.debug("Printing summary.")
    mcmc.print_summary()
    return mcmc

def inference_data(key, mcmc: MCMC, star: int) -> az.InferenceData:
    logger.debug(f"Random key={key}.")
    posterior = mcmc.get_samples(group_by_chain=True)
    sample_stats = mcmc.get_extra_fields(group_by_chain=True)
    attrs = dict(star=star, inference_library="numpyro", inference_library_version=numpyro.__version__)
    model = mcmc.sampler.model

    logger.debug("Sampling posterior predictive.")
    posterior_predictive = Predictive(model, posterior_samples=posterior, parallel=True, return_sites=["y"], batch_ndims=2)
    y = posterior_predictive(key)["y"]

    for i, key in enumerate(model.outputs):
        posterior[key] = y[..., i]

    logger.debug("Creating inference data.")
    return az.from_dict(posterior, sample_stats=sample_stats, attrs=attrs)

def summarize_data(data) -> xa.Dataset:
    logger.debug("Creating summary data.")
    return az.summary(data, fmt="xarray", stat_funcs=stat_funcs)

def save_output_data(path, **output_data):
    logger.debug("Saving output data.")
    for name, output in output_data.items():
        filepath = os.path.join(path, f"{name}.nc")
        output.to_netcdf(filepath)
        logger.info(f"Saved {name} to '{filepath}'")

def make_plots(output_directory: str, data: az.InferenceData, plot_var_names: dict, truths: dict=None):
    corner_kwargs = dict(truths=truths, show_titles=True, divergences=True, title_fmt=".3f", quantiles=[.16, .5, .84], smooth=1.)
    title = f"Star {data.attrs['star']:05}"

    for filename, var_names in plot_var_names.items():
        logger.debug(f"Making {filename} corner plot.")
        fig = corner(data, var_names=var_names, **corner_kwargs)
        fig.suptitle(title)
        filepath = os.path.join(output_directory, f"{filename}.png")
        fig.savefig(filepath)
        logger.info(f"Saved params corner plot to '{filepath}'.")

def main(args):
    from celestify.numpyro_models import SingleStarModel

    output_dir = validate_outdir(args.output_dir)
    star = args.star
    obs_filepath = args.obs_filepath
    truths_filepath = args.truths_filepath

    obs = load_obs(obs_filepath, star)
    model = init_model(obs, SingleStarModel)
    mcmc = init_mcmc(model, args.num_warmup, args.num_samples, args.num_chains)

    logger.debug("Creating random number generation key.")
    rng = random.PRNGKey(star)
    logger.debug(f"Initial RNG key for star={star} is {rng}.")

    rng, key = random.split(rng)
    mcmc = run_mcmc(key, mcmc, obs)

    rng, key = random.split(rng)
    data = inference_data(key, mcmc, star)
    summary = summarize_data(data)

    save_output_data(output_dir, data=data, summary=summary)

    if args.make_plots:
        plot_var_names = {
            "params": ["evol", "mass", "M_H", "Y", "a_MLT"],
            "outputs": SingleStarModel.outputs,
        }

        if truths_filepath is None:
            truths = None
        else:
            truths = load_truths(args.truths_filepath, star)
        make_plots(output_dir, data, plot_var_names, truths=truths)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="parse args")
    parser.add_argument(
        "star", type=int, help="star identification number"
    )
    parser.add_argument(
        "obs_filepath", type=argparse.FileType('r', encoding='UTF-8'), 
        help="filepath of observed quantities (CSV)"
    )
    parser.add_argument(
        "-t", "--truths_filepath", type=argparse.FileType('r', encoding='UTF-8'), 
        help="filepath of true quantities (CSV)"
    )
    parser.add_argument(
        "-o", "--output-dir", type=str, 
        help="directory to store outputs, defaults to current working directory"
    )
    parser.add_argument(
        "--num-samples", default=2000, type=int, help="number of samples"
    )
    parser.add_argument(
        "--num-warmup", default=2000, type=int, help="number of warmup steps"
    )
    parser.add_argument("--num-chains", default=10, type=int)
    parser.add_argument("--make-plots", action="store_true")
    parser.add_argument("--x64", action="store_true")
    parser.add_argument("--device", default="cpu", type=str, help='use "cpu" or "gpu".')

    args = parser.parse_args()
    numpyro.set_platform(args.device)
    numpyro.set_host_device_count(args.num_chains)
    if args.x64:
        numpyro.enable_x64()
    
    main(args)
