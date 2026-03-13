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
beta_output_fp = Path("checkpoints") / (model_name + "_beta_net.keras")
var_fp = Path("checkpoints") /  (model_name + "_var_net.keras")

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

n_samples = 1000
var_batch_size = 50

# %% draw posterior samples

post_draws = bayesflow_helpers.simulateChainSamples(
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
#f.savefig(output_dir + "_analysis_postsamples.png")

# %% get adjacency 

W = Rook.from_dataframe(data_shp, use_index = False)
W_full = W.full()[0]
Q_scaled, Sigma_scaled, Lambda_scaled, A_scaled = scaled_CAR(W_full)
n = Q_scaled.shape[0]

# %% define generative model

lambda_rho = 0.001
prior, likelihood, _ = bym2_sim.BYM2_simulators(Lambda_scaled,
                                             A_scaled, A_scaled, lambda_rho, p,
                                             rng = rng,
                                             corrupt_residual = False,
                                             beta_noise_sd = 1.0,
                                             beta_loc = 0.0, beta_sd = 0.5,
                                             fix_X = True, X = X_scaled)
simulator = bf.simulators.SequentialSimulator([
    bf.simulators.LambdaSimulator(prior, is_batched=True),
    bf.simulators.LambdaSimulator(likelihood, is_batched=True)
])

# sample from joint distribution
simulated_data = simulator.sample(50)
print("Data:", data)
print("Shapes:", {k: v.shape for k, v in data.items()})

# %% 

# inference model breaks down when true regression coefficients are too small?
data = bym2_sim.BYM2_likelihood(1,
                                rng.normal(scale = 2, size = (1, p + 1)),
                                np.zeros((1, p + 1)),
                                np.reshape(np.array(1), shape = (1)),
                                np.reshape(np.array(-3), shape = (1)),
                                Lambda_scaled, A_scaled, A_scaled,
                                X_fixed = X_scaled)
