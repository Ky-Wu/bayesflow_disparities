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
from src import shp_reader
from src import bayesflow_helpers
from src.spatial_covariance import scaled_CAR
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
from src import disparities_helpers as disp
from libpysal.weights import Rook
from scipy.optimize import minimize_scalar

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
data_shp.to_file("output/RDA/us_mainland_data.geojson", driver = "GeoJSON")

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
# add intercept
X_scaled = np.concatenate(
    [np.ones((X_scaled.shape[0], 1), dtype=X_scaled.dtype), X_scaled], 
    axis = 1)

y_scaled = scaler.fit_transform(y)
data = {"y": np.reshape(y_scaled, (1, n, 1))}

# %% load QR decomposition

Q_x = np.load("checkpoints/US_lungcancer_Qx.npy")
R_x = np.load("checkpoints/US_lungcancer_Rx.npy")

# %% load networks

beta_approximator = keras.saving.load_model(beta_output_fp)
var_approximator = keras.saving.load_model(var_fp)

# %% chain sampling settings

n_samples = 5000
var_batch_size = 50

# %% draw posterior samples

post_draws = bayesflow_helpers.simulate_chain_samples(
    n_samples, data, X_scaled, beta_approximator, var_approximator,
    var_batch_size = var_batch_size, R_x = R_x)

# %% plot posterior samples

par_names = [rf"$\beta_{{{i}}}$" for i in range(p + 1)]
par_names += r"$\text{log}(\sigma^2)$", r"$\text{logit}(\rho)$"
f = bf.diagnostics.plots.pairs_posterior(
    estimates=post_draws, 
    targets=None,
    variable_names=par_names,
)
f.savefig(output_dir + "RDA_analysis_postsamples.png")

# %% get adjacency 

W = Rook.from_dataframe(data_shp, use_index = False)
W_full = W.full()[0]
Q_scaled, Sigma_scaled, Lambda_scaled, A_scaled = scaled_CAR(W_full)
n = Q_scaled.shape[0]
Lambda, P = np.linalg.eigh(Q_scaled)

# %% get gamma samples

beta_samps = post_draws['beta'].squeeze()
sigma2_samps = np.exp(post_draws['log_sigma2']).squeeze(0)
rho_samps = (1.0 / (1.0 + np.exp(-post_draws['logit_rho']).squeeze(0)))
gamma_draws = disp.sample_gamma_posterior(
    beta = beta_samps,
    sigma2 = sigma2_samps,
    rho = rho_samps,
    X = X_scaled,
    y = y_scaled,
    Lambda = Lambda,
    P = P,
    rng = rng)

# %% get adj list

# Use > 0 to capture all edges, including those with weight 1
edge_list = np.argwhere(np.triu(W_full, k = 1) > 0).tolist()

# %% compute std. differences

diffs = disp.compute_std_diff(gamma_draws, sigma2_samps, rho_samps, 
                              X_scaled, Lambda, P, edge_list)

# %% optimize epsilon loss criterion

loss = lambda e: disp.conditional_entropy_loss(diffs, e)
result = minimize_scalar(loss,
                         method = 'bounded',
                         bounds = (0, 5), tol = 0.001)
optim_e = result['x']
diff_prob = disp.compute_diff_prob(diffs, optim_e)
diff_prob2 = disp.compute_diff_prob(diffs, 0.857)

# %%

mcmc_res = pd.read_csv("output/RDA/mcmc_diff_prob.csv")

# %%

node1_fips = [data_shp['County_FIPS'][i] for (i, j) in edge_list]
node2_fips = [data_shp['County_FIPS'][j] for (i, j) in edge_list]
adj_df = pd.DataFrame({"node1_fips" : node1_fips,
                       "node2_fips" : node2_fips})
adj_df['network_diff_prob'] = diff_prob2
adj_df = pd.merge(adj_df, mcmc_res, on = ['node1_fips', 'node2_fips'])

# %%

np.corrcoef(adj_df['diff_prob'], adj_df['network_diff_prob'])
plt.scatter(adj_df['diff_prob'], adj_df['network_diff_prob'], alpha = 0.1)

# %% apply bayesian FDR control

delta = 0.05
cutoff_prob, fdr_estimate = disp.compute_fdr_cutoff(diff_prob, delta = delta)

# %% 

decisions = diff_prob >= cutoff_prob
cols = ['node1_fips', 'node2_fips', 'network_diff_prob']
out = adj_df[cols]
out['optim_e'] = optim_e

pd.write_csv

