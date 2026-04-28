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
from src import bayesflow_helpers as bfhelp
from src import BYM2_simulators as bym2_sim
from src.spatial_covariance import scaled_CAR
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from src import disparities_helpers as disp
from libpysal.weights import Rook


shp_fp = "data/cb_2014_us_county_500k/cb_2014_us_county_500k.shp"
data_fp = "output/RDA/data_cleaned.csv"
model_name = "US_lungcancer"
output_dir = "output/RDA/joint_network_v8/"
net_fp = Path("checkpoints") / (model_name + "_net_v8.keras")
RDET_fp = output_dir + "RDET_df.csv"
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

# %% get adjacency 

W = Rook.from_dataframe(data_shp, use_index = False)
W_full = W.full()[0]
Q_scaled, Sigma_scaled, Lambda_scaled, A_scaled = scaled_CAR(W_full)
n = Q_scaled.shape[0]
Lambda, P = np.linalg.eigh(Q_scaled)

# %% center and scale each variable 

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
# add intercept
X_scaled = np.concatenate(
    [np.ones((X_scaled.shape[0], 1), dtype=X_scaled.dtype), X_scaled], 
    axis = 1)

y_scaled = scaler.fit_transform(y)
data = {"y": np.reshape(y_scaled, (1, n, 1))}
# test computation time
#data = {"y": np.reshape(y_scaled, (1, n, 1)) + rng.standard_normal((10, n, 1))}


# %% load QR decomposition

Q_x = np.load("checkpoints/US_lungcancer_Qx.npy")
R_x = np.load("checkpoints/US_lungcancer_Rx.npy")

# %% load networks

approximator = keras.saving.load_model(net_fp)

# %% load diff prob results

diff_prob_fp = output_dir + "diff_probs.csv"
diff_prob_res = pd.read_csv(diff_prob_fp)
diff_prob = diff_prob_res['approx_diff_prob'].to_numpy()

# %% apply bayesian FDR control

delta = 0.05
cutoff_prob, fdr_estimate = disp.compute_fdr_cutoff(diff_prob, delta = delta)
disp_indx = diff_prob >= cutoff_prob

# %% initialize RDET df
"""
RDET : Residual Disparity Elimination Target
For a disparity between two counties, A and B, suppose that A has the higher 
expected spatial residual (gamma). To eliminate the disparity between A and B, 
the RDET is defined as the minimum % decrease in mortality in A, holding all
other observed covariates and mortality rates constant, such that the model
does not report a disparity between A and B.
"""
df = diff_prob_res[disp_indx]
optim_e = df.iloc[0]['optim_e']
k = len(df)
RDET_df = pd.DataFrame({
    "county1" : df['county1'],
    "county2" : df['county2'],
    "higher_gamma_mean_county" : df['higher_gamma_mean_county'],
    "approx_diff_prob" : df['approx_diff_prob'],
    "delta" : np.repeat(delta, k),
    "reduction_factor": np.repeat(-1, k),
    "RDET_percent" : np.repeat(-1, k)
})
RDET_df = RDET_df.reset_index()

# %% RDET computation settings

steps = 20
step_size = 0.005
n_samples = 3000
batch_size = 10
lower_end = 1.0 - step_size * steps
f = np.linspace(lower_end, 1.0 - step_size, num = steps)
y_np = y.to_numpy()
  

# %% compute RDET function

def compute_rf(new_y, f, subset_indx, n_samples, batch_size, optim_e, cutoff_prob):
    data = {"y": new_y_scaled}
    post_draws = approximator.sample(conditions=data, num_samples=n_samples,
                                     batch_size = batch_size)
    beta = bfhelp.backtransform_beta_samps(post_draws, R_x = R_x)
    sigma2 = np.exp(post_draws['log_sigma2'])
    rho = (1.0 / (1.0 + np.exp(-post_draws['logit_rho'])))
    gamma = disp.sample_gamma_posterior(
        beta = beta,
        sigma2 = sigma2,
        rho = rho,
        X = X_scaled,
        y = new_y,
        Lambda = Lambda,
        P = P,
        rng = rng,
        subset_indx = subset_indx)
    phi = gamma / np.sqrt(sigma2 * rho)
    d = phi[:,:,0] - phi[:,:,1]
    stds = np.std(d, axis = 1, keepdims = True)
    diffs = d / stds
    diff_prob = disp.compute_diff_prob(diffs, optim_e)
    if any(diff_prob < cutoff_prob):
        indx = np.where(diff_prob < cutoff_prob)[0][-1]
        rf = f[indx]
    else:
        rf = None
    return rf


# %% compute RDETs

for i in range(k):
    print(f"Evaluating RDET for disparity {i + 1}/{k}")
    pair_counties = df.iloc[i][['county1', 'county2']].values.astype(int)
    target_county = df.iloc[i]["higher_gamma_mean_county"]    
    r_indx = data_shp.index[data_shp['County_FIPS'] == target_county].tolist()[0]
    subset_indx = data_shp.index[data_shp['County_FIPS'].isin(pair_counties)].tolist()
    # check reduction factors in batches (0.9-0.995 first)
    rf = None
    batch_counter = 1
    while rf is None:
        lower_end = 1.0 - batch_counter * step_size * steps
        upper_end = 1.0 - step_size - (batch_counter - 1) * step_size * steps
        f = np.linspace(lower_end, upper_end, num = steps)
        new_y = np.repeat(np.array(y)[np.newaxis,...], repeats = steps, axis = 0)
        new_y[:,r_indx,:] *= f[:,np.newaxis]
        new_y_scaled = (new_y - np.mean(y_np, axis = 0)) / (np.std(y_np, axis = 0))
        rf = compute_rf(new_y_scaled, f, subset_indx,
                        n_samples, batch_size, optim_e, cutoff_prob)
        batch_counter += 1
    RDET = 100 * (1.0 - rf)
    RDET_df.loc[i, 'reduction_factor'] = rf
    RDET_df.loc[i, 'RDET_percent'] = RDET
    RDET_df.to_csv(RDET_fp, index = False)


# %%

if os.path.isfile(RDET_fp):
    RDET_df = pd.read_csv(RDET_fp)
    print("RDET data loaded successfully.")
else:
    print("RDET data file not found.")  
    
# %%

x = RDET_df['RDET_percent'].values
dp = RDET_df['approx_diff_prob'].values
x_s = x[x >= 0.0]
dp = dp[x >= 0.0]
import matplotlib.pyplot as plt
plt.scatter(dp, x_s, alpha = 0.5, s = 10)
plt.xlabel("Estimated Disparity Difference Probability")
plt.ylabel("RDET (%)")