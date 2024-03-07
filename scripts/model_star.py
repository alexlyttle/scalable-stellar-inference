import os, logging, argparse, numpyro, io, celestify
import numpy as np
import pandas as pd
import arviz as az
import xarray as xa

from jax import random, Array
from numpyro.infer import MCMC, NUTS, Predictive
from corner import corner, overplot_lines, overplot_points
from typing import Optional
from collections.abc import MutableMapping
from celestify.numpyro_models import SingleStarModel

logger = logging.getLogger("model_star")
logger.setLevel(logging.DEBUG) # lowest-severity level handled by the module logger
logger.addHandler(logging.NullHandler())

init_strategy = numpyro.infer.init_to_median
target_accept_prob = 0.8
find_heuristic_step_size = True
regularize_mass_matrix = True
dense_mass = False
kind = "diag"
star_fmt = "05d"

stat_funcs = {
    "lower": lambda x: np.quantile(x, .16),
    "median": lambda x: np.quantile(x, .5),
    "upper": lambda x: np.quantile(x, .84),
}

plot_var_names = {
    "params": ["evol", "mass", "M_H", "Y", "a_MLT"],
    "outputs": SingleStarModel.outputs,
}


def flatten_dict(dictionary, parent_key='', separator='_'):
    items = []
    for key, value in dictionary.items():
        new_key = parent_key + separator + key if parent_key else key
        if isinstance(value, MutableMapping):
            items.extend(flatten_dict(value, new_key, separator=separator).items())
        else:
            items.append((new_key, value))
    return dict(items)

def init_parser():
    parser = argparse.ArgumentParser(description="parse args")
    parser.add_argument(
        "star", type=int, help="star identification number"
    )
    parser.add_argument(
        "obs_file", type=argparse.FileType('r', encoding='UTF-8'), 
        help="filepath of observed quantities (CSV)"
    )
    parser.add_argument(
        "-t", "--truths-file", type=argparse.FileType('r', encoding='UTF-8'), 
        help="filepath of true quantities (CSV)"
    )
    parser.add_argument(
        "-d", "--directory", type=str, 
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
    parser.add_argument("-l", "--log-file", type=argparse.FileType('w', encoding='UTF-8'), help='log file')
    parser.add_argument("-L", "--log-level", default="WARNING", type=str, 
                        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
                        help='log level')
    parser.add_argument("--observables", nargs='+', default=["log_Teff", "log_L"], help="names of observed quantities")
    return parser

def numpyro_config(device: str, num_chains: int, x64: bool):
    logger.debug(f"Configuring numpyro for device={device}, num_chains={num_chains}, and x64={x64}.")
    numpyro.set_platform(device)
    numpyro.set_host_device_count(num_chains)
    numpyro.enable_x64(x64)

def add_logging_handler(level: str, file: Optional[io.TextIOWrapper]=None):
    formatter = logging.Formatter('%(asctime)s %(name)s %(levelname)-8s %(message)s')
    handler = logging.StreamHandler(file)
    handler.setLevel(level)
    handler.setFormatter(formatter)
    logger.addHandler(handler)

def validate_dir(directory: str):
    if not os.path.exists(directory):
        logger.info(f"Creating directory '{directory}'.")
        os.makedirs(directory)
    elif not os.path.isdir(directory):
        raise ValueError(f"Path '{directory}' is not a directory.")
    return directory

def create_outdir(root_directory: Optional[str], star: int) -> str:
    if root_directory is None:
        root_directory = os.getcwd()
    root_directory = validate_dir(root_directory)
    output_directory = validate_dir(os.path.join(root_directory, f"{star:{star_fmt}}"))
    return output_directory

def load_obs(file: io.TextIOWrapper, star: int) -> dict:
    logger.debug(f"Loading observables from '{file.name}'.")
    obs = pd.read_csv(file, index_col=0)
    return obs.loc[star].to_dict()

def load_truths(file: Optional[io.TextIOWrapper], star: int) -> dict:
    if file is None:
        return file
    logger.debug(f"Loading truths from '{file.name}'.")
    truths = pd.read_csv(file, index_col=0)
    return truths.loc[star].to_dict()

def init_model(data: dict, observables: list) -> SingleStarModel:
    # Create dictionary of model constants
    const = {
        "M_H": dict(loc=data["M_H"], scale=data["sigma_M_H"], low=-0.85, high=0.45),
    }
    logger.debug(f"Initialising model with observables={observables} and const={const}.")
    return SingleStarModel(observables, const=const, kind=kind)

def init_mcmc(model: SingleStarModel, num_warmup: int, num_samples: int, num_chains: int):
    logger.debug(f"Initialising NUTS with model={model} and init_strategy={init_strategy}.")
    sampler = NUTS(model, init_strategy=init_strategy, target_accept_prob=target_accept_prob,
                   find_heuristic_step_size=find_heuristic_step_size, dense_mass=dense_mass,
                   regularize_mass_matrix=regularize_mass_matrix)

    logger.debug(f"Initialising MCMC with sampler={sampler}, num_warmup={num_warmup}, num_samples={num_samples}, and num_chains={num_chains}.")
    return MCMC(sampler, num_warmup=num_warmup, num_samples=num_samples, num_chains=num_chains)

def run_mcmc(key: Array, mcmc: MCMC, obs: dict) -> MCMC:
    observables = mcmc.sampler.model.observables
    logger.debug(f"Creating observed y and diag arrays.")
    y = np.stack([obs[key] for key in observables], axis=-1)
    diag = np.stack([obs[f"sigma_{key}"] for key in observables], axis=-1)**2

    logger.debug(f"Random key={key}.")
    logger.debug(f"Observations y={y}.")
    logger.debug(f"Diagonal diag={diag}.")

    logger.info("Running MCMC.")
    mcmc.run(key, obs=y, diag=diag)
    logger.info("Run complete.")
    mcmc.print_summary()
    return mcmc

def inference_data(key: Array, mcmc: MCMC, star: int) -> az.InferenceData:
    logger.debug(f"Random key={key}.")
    model = mcmc.sampler.model
    posterior = mcmc.get_samples(group_by_chain=True)
    sample_stats = mcmc.get_extra_fields(group_by_chain=True)
    attrs = dict(star=star, observables=model.observables, inference_library="numpyro", inference_library_version=numpyro.__version__)

    logger.debug("Sampling posterior predictive.")
    posterior_predictive = Predictive(model, posterior_samples=posterior, parallel=True, return_sites=["y"], batch_ndims=2)
    y = posterior_predictive(key)["y"]

    for i, key in enumerate(model.outputs):
        posterior[key] = y[..., i]

    constant_data = flatten_dict(model.const) | mcmc._kwargs
    logger.debug("Creating inference data.")
    return az.from_dict(posterior, sample_stats=sample_stats, constant_data=constant_data, attrs=attrs)

def summarize_data(data: az.InferenceData) -> xa.Dataset:
    logger.debug("Creating summary data.")
    return az.summary(data, fmt="xarray", stat_funcs=stat_funcs)

def warn_diverging(data: az.InferenceData):
    if not "diverging" in data.sample_stats:
        return
    num_diverging = data.sample_stats.diverging.sum()
    if num_diverging > 0:
        logger.warning(f"There were {num_diverging.values} divergences in this MCMC run.")

def warn_rhat(summary: xa.Dataset):
    if not "r_hat" in summary:
        return
    ...

def save_output_data(path: str, **output_data):
    logger.debug("Saving output data.")
    for name, output in output_data.items():
        filepath = os.path.join(path, f"{name}.nc")
        output.to_netcdf(filepath)
        logger.info(f"Saved {name} to '{filepath}'.")

def make_plots(output_directory: str, data: az.InferenceData, truths: dict=None):
    corner_kwargs = dict(truths=truths, show_titles=True, divergences=True, title_fmt=".3f",
                         quantiles=[.16, .5, .84], smooth=1.,
                         divergences_kwargs=dict(color="C2"))
    title = f"Star {data.attrs['star']:{star_fmt}}"
    obs = dict(zip(data.attrs["observables"], data.constant_data["obs"]))

    for filename, var_names in plot_var_names.items():
        logger.debug(f"Making {filename} corner plot.")
        fig = corner(data, var_names=var_names, **corner_kwargs)
        xs = np.array([obs.get(k) for k in var_names])
        overplot_lines(fig, xs, color="C1", ls="--")
        overplot_points(fig, xs[None, ...], color="C1", marker="o")
        fig.suptitle(title)
        filepath = os.path.join(output_directory, f"{filename}.png")
        fig.savefig(filepath)
        logger.info(f"Saved params corner plot to '{filepath}'.")

def main():
    parser = init_parser()
    args = parser.parse_args()
    add_logging_handler(args.log_level, file=args.log_file)

    numpyro_config(args.device, args.num_chains, args.x64)

    star = args.star
    obs_file = args.obs_file
    truths_file = args.truths_file
    output_dir = create_outdir(args.directory, star)

    logger.info(f"Modelling star {star} with celestify version {celestify.__version__}.")
    obs = load_obs(obs_file, star)
    model = init_model(obs, args.observables)
    mcmc = init_mcmc(model, args.num_warmup, args.num_samples, args.num_chains)

    logger.debug("Creating random number generation key.")
    rng = random.PRNGKey(star)
    logger.debug(f"Initial RNG key is {rng}.")

    rng, key = random.split(rng)
    mcmc = run_mcmc(key, mcmc, obs)

    rng, key = random.split(rng)
    data = inference_data(key, mcmc, star)
    summary = summarize_data(data)
    warn_diverging(data)
    warn_rhat(summary)
    save_output_data(output_dir, data=data, summary=summary)

    if args.make_plots:
        truths = load_truths(truths_file, star)
        make_plots(output_dir, data, truths=truths)
    
    logger.info("Complete.")

if __name__ == "__main__":
    main()
