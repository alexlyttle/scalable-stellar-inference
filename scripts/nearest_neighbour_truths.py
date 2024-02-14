import os, json
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.spatial import KDTree

# Constants
directory = "/mnt/data-storage/alexlyttle/scalable-stellar-inference"
input_cols = ["EEP", "star_mass", "M_H", "Yinit", "amlt"]
model_cols = ["evol", "mass", "M_H", "Y", "a_MLT"]  # equivalent model input columns
data_filename = "/mnt/data-storage/yaguangli2023/stellar-models/grid_models_surface_effect_uncorrected/dataset.h5"
data_groups = ["train", "test"]
truths_filename = "truths.json"
output_filename = "truths-nearest-neighbour.csv"
dist_filename = "nearest-neighbour-distances.png"
nearest_filename = "nearest-neighbours.png"
error_filename = "nearest-neighbour-error.png"

# Load data
tables = [pd.read_hdf(data_filename, key) for key in data_groups]
data = pd.concat(tables, axis=0, ignore_index=True)

# Load truths
with open(os.path.join(directory, truths_filename), "r") as file:
    s = file.read()
    truths = json.loads(s)

# Normalise KDTree inputs to between 0 and 1
loc = data[input_cols].min(0).to_numpy()
scale = data[input_cols].max(0).to_numpy() - loc
tree = KDTree((data[input_cols] - loc) / scale)

# Find indices and distances to truth nearest neighbours
model_cols = ["evol", "mass", "M_H", "Y", "a_MLT"]
x = np.stack([truths[k] for k in model_cols], axis=-1)
x_scaled = (x - loc) / scale
dist, indices = tree.query(x_scaled)

# Save nearest neighbours
nearest_neighbours = data.iloc[indices]
nearest_neighbours.to_csv(os.path.join(directory, output_filename))

# Number of bins
quartiles = np.quantile(dist, [.25, .75])
iqr = quartiles[1] - quartiles[0]
h = 2 * iqr / dist.shape[0]**(1/3)
bins = int((dist.max() - dist.min()) / h)

# Plot distances to nearest neighbours
fig, ax = plt.subplots()
ax.hist(dist, bins=bins)
ax.set_xlabel("distance")
ax.set_ylabel("count")
ax.set_title("Cartesian distances to nearest neighbours")
fig.savefig(os.path.join(directory, dist_filename))

# Plot nearest neighbours
splot = sns.pairplot(nearest_neighbours, vars=input_cols, kind="hist", diag_kind="hist");
fig = splot.figure
fig.suptitle("Nearest neighbours")
fig.tight_layout()
fig.savefig(os.path.join(directory, nearest_filename))

# Plot error between nearest neighbours and truths
error = nearest_neighbours[input_cols] - x
splot = sns.pairplot(error, kind="hist", diag_kind="hist")
fig = splot.figure
fig.suptitle("Nearest neighbour error")
fig.tight_layout()
fig.savefig(os.path.join(directory, error_filename))
