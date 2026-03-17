#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 12 21:06:08 2026

@author: kylel
"""

# %% load libraries and config
import jax
import os
jax.config.update("jax_enable_x64", False)
os.environ["KERAS_BACKEND"] = "jax"
import bayesflow as bf
import keras
from src import summary_networks
from src import BYM2_simulators as bym2_sim
from src import shp_reader
from src import bayesflow_helpers
from src.spatial_covariance import scaled_CAR
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from libpysal.weights import Rook
import matplotlib.pyplot as plt
import seaborn as sns

shp_fp = "data/cb_2017_us_county_500k/cb_2017_us_county_500k.shp"
data_fp = "output/RDA/data_cleaned.csv"
model_name = "US_lungcancer"
output_dir = "output/RDA/"
beta_output_fp = Path("checkpoints") / (model_name + "_beta_net_oldv2.keras")
var_fp = Path("checkpoints") /  (model_name + "_var_net_oldv2.keras")

rng = np.random.default_rng(seed = 1130)

# %% load in shapefile and cleaned data from RDA_data_setup.py

shp, _ = shp_reader.read_US_shapefile(shp_fp)
shp['County_FIPS'] = (shp['STATEFP'] + shp['COUNTYFP']).astype(int)
all_data = pd.read_csv(data_fp)
data_shp = pd.merge(shp, all_data, on = "County_FIPS", how = "inner")
data_shp = data_shp.reset_index(drop = True)

# %% extract covariate matrix and resposne vector

pred_cols = ['total_mean_smoking', 'unemployed_2014', 'SVI_2014', 'inactivity_2014',
'uninsured_2012_2016', 'diabetes_2014', 'obesity_2014']
X = data_shp[pred_cols]
y = data_shp[['mortality2014']]
n = X.shape[0]
p = len(pred_cols)

# %% center and scale each variable 

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
y_log = np.log(y)
y_scaled = (y_log - np.mean(y_log)) / np.std(y_log)
data = {"y": np.reshape(y_scaled, (1, n, 1))}

# %% load networks

beta_approximator = keras.saving.load_model(beta_output_fp)
var_approximator = keras.saving.load_model(var_fp)

# %% chain sampling settings

n_samples = 5000
var_batch_size = 50

# %% draw posterior samples

post_draws = bayesflow_helpers.simulate_chain_samples(
    n_samples, data, X_scaled, beta_approximator, var_approximator,
    var_batch_size = var_batch_size)

# %% plot posterior samples

par_names = [rf"$\beta_{{{i}}}$" for i in range(p + 1)]
par_names += r"$\text{log}(\sigma^2)$", r"$\text{logit}(\rho)$"
f = bf.diagnostics.plots.pairs_posterior(
    estimates=post_draws, 
    targets=None,
    variable_names=par_names,
)
f.savefig(output_dir + "RDA_analysis_postsamples.png")


