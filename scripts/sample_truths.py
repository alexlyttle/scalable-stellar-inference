import os, json
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from jax import random
from numpyro import handlers
from celestify.numpyro_models import HierarchicalStarModel

# CONSTANTS
seed = 0
directory = "/mnt/data-storage/alexlyttle/scalable-stellar-inference"
filename = "truths.json"
plotname = "truths.png"
num_stars = 100000
truths = {
    "mu_a": 2.0,
    "sigma_a": 0.05,
    "Y_0": 0.247,
    "dY_dZ": 1.5,
    "sigma_Y": 0.005,
}
plot_params = ["evol", "mass", "M_H", "Y", "a_MLT"]

# LOAD MODEL
true_model = HierarchicalStarModel(num_stars)

# SAMPLE MODEL
key = random.PRNGKey(seed)
true_trace = handlers.trace(
    handlers.seed(handlers.substitute(true_model, truths), key)  # substitute truths
).get_trace()
truths.update({k: np.array(v["value"]) for k, v in true_trace.items()})

# SAVE TRUTHS
with open(os.path.join(directory, filename), "w") as file:
    file.write(json.dumps({k: v.tolist() for k, v in truths.items()}))

# PLOT TRUTHS
x = np.stack([truths[k] for k in plot_params], axis=-1)
samples = pd.DataFrame(x, columns=plot_params)
sns.pairplot(samples, kind="hist", diag_kind="hist")
plt.savefig(os.path.join(directory, plotname))
