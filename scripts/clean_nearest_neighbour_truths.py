import os
import pandas as pd
from celestify.numpyro_models import SingleStarModel as Model

max_teff = 7500.
max_age = 14.
directory = "/mnt/data-storage/alexlyttle/scalable-stellar-inference"
input_filename = "truths-nearest-neighbour.csv"
output_filename = "truths-nearest-neighbour-clean.csv"

rename_cols = {
    "EEP": "evol",
    "star_mass": "mass",
    "log_star_mass": "log_mass",
    "Yinit": "Y",
    "amlt": "a_MLT",
    "radius": "R",
    "log_radius": "log_R",
    "Dnu_freq_o": "Dnu",
    "log_Dnu_freq_o": "log_Dnu",
}

df = pd.read_csv(os.path.join(directory, input_filename), index_col=0)
df["L"] = 10**df["log_L"]
df["log_age"] = df["log_star_age"] - 9
df["age"] = 10**df["log_age"]
df["log_numax"] = (
    df["log_g"] - Model.log_g_sun
    - 0.5 * (df["log_Teff"] - Model.log_teff_sun) + Model.log_numax_sun
)
df["numax"] = 10**df["log_numax"]

mask = (df.Teff < max_teff) & (df.age < max_age)
df = df.loc[mask].rename(rename_cols, axis=1)
df.reset_index().to_csv(os.path.join(directory, output_filename))
