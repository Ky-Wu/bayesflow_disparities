#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

@author: kylel
"""

# %% load libraries and config
import jax
import os
jax.config.update("jax_enable_x64", False)
os.environ["KERAS_BACKEND"] = "jax"
cwd = os.getcwd()
print("Current working directory:" + cwd)
import bayesflow as bf
import keras
from src import summary_networks
from src import BYM2_simulators as bym2_sim
from src import shp_reader
from src import bayesflow_helpers as bfhelp
from src.spatial_covariance import scaled_CAR
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from libpysal.weights import Rook
                         
shp_fp = os.path.join(cwd, "data", "cb_2014_us_county_500k", "cb_2014_us_county_500k.shp")
data_fp = os.path.join(cwd, "output", "RDA", "data_cleaned.csv")
model_name = "US_lungcancer"
output_dir = os.path.join(cwd, "output", "RDA", "joint_network_test/")
output_fp = os.path.join(cwd, "checkpoints", (model_name + "_net_test.keras"))
rng = np.random.default_rng(seed = 1130)


# %% load in shapefile and cleaned data from RDA_data_setup.py

shp, _ = shp_reader.read_US_shapefile(shp_fp)
shp['County_FIPS'] = (shp['STATEFP'] + shp['COUNTYFP']).astype(int)
all_data = pd.read_csv(data_fp)
data_shp = pd.merge(shp, all_data, on = "County_FIPS", how = "inner")
data_shp = data_shp.reset_index(drop = True)

# %% extract covariate matrix and response vector

pred_cols = ['total_mean_smoking', 'unemployed_2014', 'SVI_2014', 'inactivity_2014',
'uninsured_2012_2016', 'diabetes_2014', 'obesity_2014']
p = len(pred_cols)
X = data_shp[pred_cols]
y = data_shp[['mortality2014']]

# %% center and scale each variable 

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
#y_scaled = scaler.fit_transform(y)
X_scaled = np.concatenate(
    [np.ones((X_scaled.shape[0], 1), dtype=X_scaled.dtype), X_scaled], 
    axis = 1)

# %% QR decomposition

Q_x, R_x = np.linalg.qr(X_scaled, mode = 'reduced')
np.save("checkpoints/US_lungcancer_Qx.npy", Q_x)
np.save("checkpoints/US_lungcancer_Rx.npy", R_x)

# %% get adjacency 

W = Rook.from_dataframe(data_shp, use_index = False)
W_full = W.full()[0]
Q_scaled, Sigma_scaled, Lambda_scaled, A_scaled = scaled_CAR(W_full)
n = Q_scaled.shape[0]

# %% prior hyperparameters

# zellner-g prior settings
tau_beta = np.sqrt(n / 5.0)
sigma2_prior_sd = np.sqrt(1.0)
theta_isotropic = True

# %% define generative model

lambda_rho = 0
prior, likelihood, _ = bym2_sim.BYM2_simulators(
    Lambda_scaled, A_scaled, A_scaled, lambda_rho, p,
    rng = rng,
    corrupt_residual = False,
    beta_loc = 0.0,
    tau_beta = tau_beta,
    sigma2_sd = sigma2_prior_sd,
    fix_X = True, X = X_scaled,
    R_x = R_x, theta_isotropic = theta_isotropic)
simulator = bf.simulators.SequentialSimulator([
    bf.simulators.LambdaSimulator(prior, is_batched=True),
    bf.simulators.LambdaSimulator(likelihood, is_batched=True)
])

# sample from joint distribution
data = simulator.sample(50)
#print("Data:", data)
print("Shapes:", {k: v.shape for k, v in data.items()})

# %% define summary network and adapter for fixed effects

adapter = (
    bf.Adapter()
    .convert_dtype("float64", "float32")
    .concatenate(["theta", "log_sigma2", "logit_rho"], into="inference_variables")
    .concatenate(["y"], into="summary_variables")
)

# Custom summary network (spatial data not row-exchangeable)

summary_net = summary_networks.ResidualSummary(
    adjacency_matrix = W_full,
    X = X_scaled,
    gnn_dim = 32,
    compress_dim = 256,
    hidden_dim = 128,
    summary_dim = 64)

#summary_net = summary_networks.SummaryIdentity()

# %% define inference network and amortizer

depth = 8
inference_net = bf.networks.CouplingFlow(
    depth=depth,
    transform = "affine",   
    subnet_kwargs={
       "widths": [128, 128],  # Widths of the hidden layers
       "activation": "swish",
       "dropout": False,
       "dropout_prob": 0.0
   }
)


# %% define workflow, combining beta summary and inference network into approximator

workflow = bf.BasicWorkflow(
    simulator=simulator,
    adapter=adapter,
    inference_network=inference_net,
    summary_network=summary_net,
    standardize=["inference_variables"]
)

# %% start training!

history = workflow.fit_online(epochs = 50, 
                              batch_size = 64,
                              num_batches_per_epoch = 100,
                              validation_data = 64,
                              callbacks = [bfhelp.CleanLRLogger()],
                              verbose = 2)
bf.diagnostics.plots.loss(history)
