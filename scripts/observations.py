import os
import numpy as np
import pandas as pd
import jax.numpy as jnp
from jax import random

seed = 101
observables = ["M_H", "log_Teff", "log_L", "log_Dnu"]
directory = "/mnt/data-storage/alexlyttle/scalable-stellar-inference"
input_filename = "truths-nearest-neighbour-clean.csv"
output_filename = "observables.csv"
rng = random.PRNGKey(seed)

df = pd.read_csv(os.path.join(directory, input_filename), index_col=0)

sigma = {}
sigma["M_H"] = 0.1
sigma["log_Teff"] = 75. / df["Teff"] / np.log(10)
sigma["log_L"] = 0.01 / np.log(10)
sigma["log_Dnu"] = 0.2 / df["Dnu"] / np.log(10)

mean = np.stack([df[param] for param in observables], -1)
sigma = np.stack(np.broadcast_arrays(*[sigma[param] for param in observables]), -1)

rng, key = random.split(rng)
obs = mean + sigma * random.normal(key, shape=mean.shape)

out = pd.DataFrame(
    np.concatenate([obs, sigma], axis=1),
    index=df.index,
    columns=observables + [f"sigma_{obs}" for obs in observables]
)
out.to_csv(os.path.join(directory, output_filename))
