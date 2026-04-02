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
import matplotlib.pyplot as plt
import seaborn as sns
from src import disparities_helpers as disp
from libpysal.weights import Rook
from scipy.optimize import minimize_scalar

shp_fp = "data/cb_2017_us_county_500k/cb_2017_us_county_500k.shp"
data_fp = "output/RDA/data_cleaned.csv"
model_name = "US_lungcancer"
output_dir = "output/RDA/joint_network_v1/"
net_fp = Path("checkpoints") / (model_name + "_net_v1.keras")

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
tau2_beta = np.sqrt(n / 10.0)

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

# %% define generative model

lambda_rho = 0
tau2_beta = np.sqrt(n / 10.0)
sigma2_prior_sd = np.sqrt(1.0)
theta_isotropic = True
prior, likelihood, _ = bym2_sim.BYM2_simulators(
    Lambda_scaled, A_scaled, A_scaled, lambda_rho, p,
    rng = rng,
    corrupt_residual = False,
    beta_loc = 0.0,
    tau_beta = tau2_beta,
    sigma2_sd = sigma2_prior_sd,
    fix_X = True, X = X_scaled,
    R_x = R_x, theta_isotropic = theta_isotropic)
simulator = bf.simulators.SequentialSimulator([
    bf.simulators.LambdaSimulator(prior, is_batched=True),
    bf.simulators.LambdaSimulator(likelihood, is_batched=True)
])
data = simulator.sample(5)

# %% load networks

approximator = keras.saving.load_model(net_fp)

# %%

approximator.save_weights(output_dir + "model.weights.h5")

# %% train more if needed 

summary_net = summary_networks.SummaryGNNPlusIdentity(
    adjacency_matrix = W_full,
    gnn_dim = 32,
    compress_dim = 128,
    hidden_dim = 64,
    summary_dim = 32)


depth = 8
inference_net = bf.networks.CouplingFlow(
    depth=depth,
    permutation = None,
    transform = "affine",   
    subnet_kwargs={
       "units": [256 for i in range(depth - 1)],  # Widths of the hidden layers
       "activation": "swish",
       "dropout": False,
       "dropout_prob": 0.0
   }
)

adapter = (
    bf.Adapter()
    .convert_dtype("float64", "float32")
    .concatenate(["theta", "log_sigma2", "logit_rho"], into="inference_variables")
    .concatenate(["y"], into="summary_variables")
)

new_approx =  bf.approximators.ContinuousApproximator(
    summary_network=summary_net,
    inference_network=inference_net,
    adapter=adapter,
    standardize = ["inference_variables"]
)

TARGET_LR = 1e-5   # 5e-5
EPOCHS = 50
NUM_BATCHES = 200             # steps per epoch
TOTAL_STEPS = EPOCHS * NUM_BATCHES

lr_schedule = keras.optimizers.schedules.CosineDecay(
    initial_learning_rate=TARGET_LR,  # starts at 5e-5
    decay_steps=TOTAL_STEPS,
    alpha=0.0,          # decays all the way to 0 at the end
)
optimizer = keras.optimizers.AdamW(learning_rate=1e-5, 
                                   weight_decay=1e-4,
                                   clipnorm=1.1)
new_approx.compile(optimizer = optimizer)
history = new_approx.fit(
    simulator=simulator,
    num_batches=1,
    epochs=1,
    batch_size=1
)

# %%

new_approx.load_weights(output_dir + "model.weights.h5")

# %%

history = new_approx.fit(
    epochs=50,
    num_batches=200,
    batch_size=64,
    validation_batch_size = 64,
    callbacks = [bfhelp.CleanLRLogger()],
    simulator=simulator,
)

# %% chain sampling settings

n_samples = 500
batch_size = 10

# %% draw posterior samples

post_draws = approximator.sample(conditions=data, num_samples=n_samples,
                                 batch_size = batch_size)

post_draws = {
    'beta' : bfhelp.backtransform_beta_samps(post_draws, R_x = R_x),
    'log_sigma2' : post_draws['log_sigma2'],
    'logit_rho' : post_draws['logit_rho']
    }


# %% plot posterior samples

par_names = [rf"$\beta_{{{i}}}$" for i in range(p + 1)]
par_names += r"$\text{log}(\sigma^2)$", r"$\text{logit}(\rho)$"
f = bf.diagnostics.plots.pairs_posterior(
    estimates=post_draws, 
    targets=None,
    variable_names=par_names,
)
f.savefig(output_dir + "RDA_analysis_postsamples.png")


# %% get gamma samples

beta = post_draws['beta']
sigma2 = np.exp(post_draws['log_sigma2'])
rho = (1.0 / (1.0 + np.exp(-post_draws['logit_rho'])))
gamma = disp.sample_gamma_posterior(
    beta = beta,
    sigma2 = sigma2,
    rho = rho,
    X = X_scaled,
    y = y_scaled,
    Lambda = Lambda,
    P = P,
    rng = rng)

# %% get adj list

# Use > 0 to capture all edges, including those with weight 1
edge_list = np.argwhere(np.triu(W_full, k = 1) > 0).tolist()

# %% compute std. differences

diffs = disp.gamma_diffs_marginalvar(gamma, sigma2, rho, edge_list)

# %%

diffs = disp.compute_std_diff(gamma, sigma2, rho, 
                                         X_scaled, Lambda, P,
                                         edge_list)

# %% optimize epsilon loss criterion

loss = lambda e: disp.conditional_entropy_loss(diffs, e)
result = minimize_scalar(loss,
                         method = 'bounded',
                         bounds = (0, 5), tol = 0.001)
optim_e = result['x']
diff_prob = disp.compute_diff_prob(diffs, optim_e)


# %% apply bayesian FDR control

delta = 0.05
cutoff_prob, fdr_estimate = disp.compute_fdr_cutoff(diff_prob, delta = delta)

# %% 

decisions = diff_prob >= cutoff_prob

# %%
import rdata

# Parses the file into a Python object representation
with open('output/RDA/mcmc_samps.rds', 'rb') as f:
    parsed_data = rdata.read_rds(f)
beta_mcmc = parsed_data['beta'][np.newaxis,:,:]
n_s = beta_mcmc.shape[0]
sigma2_mcmc = parsed_data['sigma2'][np.newaxis,:,np.newaxis]
rho_mcmc = parsed_data['rho'][np.newaxis,:,np.newaxis]
gamma_mcmc = disp.sample_gamma_posterior(
    beta = beta_mcmc,
    sigma2 = sigma2_mcmc,
    rho = rho_mcmc,
    X = X_scaled,
    y = y_scaled,
    Lambda = Lambda,
    P = P,
    rng = rng)

# %%

mcmc_diffs = disp.gamma_diffs_marginalvar(gamma_mcmc, sigma2_mcmc, rho_mcmc, 
                                          edge_list)

# %% optimize epsilon loss criterion

loss = lambda e: disp.conditional_entropy_loss(mcmc_diffs, e)
result = minimize_scalar(loss,
                         method = 'bounded',
                         bounds = (0, 5), tol = 0.001)
mcmc_optim_e = result['x']
mcmc_diff_prob = disp.compute_diff_prob(mcmc_diffs, optim_e)


# %% compare mcmc and network diff probs

plt.scatter(diff_prob, mcmc_diff_prob, alpha = 0.05)
print("Correlation: " + str(np.corrcoef(diff_prob, mcmc_diff_prob)[0,1]))

# %% compare beta (smoking)

# Sort the samples
j = 1
beta1 = np.sort(beta_samps[:,j])
plt.hist(beta1)
beta2 = np.sort(beta_mcmc[:,j])
plt.hist(beta2, color="red")

# Create the plot
plt.figure(figsize=(6, 6))
plt.plot(beta1, beta2, ls="", marker="o") # "ls=" removes the line, "marker=o" adds points

# Add a 45-degree reference line
max_val = max(beta1[-1], beta2[-1])
min_val = min(beta1[0], beta2[0])
plt.plot([min_val, max_val], [min_val, max_val], 'k-', lw=1) # 'k-' for a black solid line

plt.xlabel("Quantiles of Sample 1")
plt.ylabel("Quantiles of Sample 2")
plt.title("Q-Q Plot of Two Samples")
plt.show()
