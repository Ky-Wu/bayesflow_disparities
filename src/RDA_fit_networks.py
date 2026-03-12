#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  9 16:39:36 2026

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
                         
shp_fp = "data/cb_2017_us_county_500k/cb_2017_us_county_500k.shp"
data_fp = "output/RDA/data_cleaned.csv"
model_name = "US_lungcancer"
output_dir = "output/RDA/"

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
p = len(pred_cols)
X = data_shp[pred_cols]
y = data_shp[['mortality2014']]

# %% center and scale each variable 

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
#y_scaled = scaler.fit_transform(y)

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
                                             beta_loc = 0.0, beta_sd = 10.0,
                                             fix_X = True, X = X_scaled)
simulator = bf.simulators.SequentialSimulator([
    bf.simulators.LambdaSimulator(prior, is_batched=True),
    bf.simulators.LambdaSimulator(likelihood, is_batched=True)
])

# sample from joint distribution
data = simulator.sample(50)
print("Data:", data)
print("Shapes:", {k: v.shape for k, v in data.items()})


# %% define summary network and adapter for fixed effects

beta_adapter = (
    bf.Adapter()
    .convert_dtype("float64", "float32")
    .concatenate(["beta"], into="inference_variables")
    .concatenate(["y"], into="summary_variables")
)

# Graph neural network as summary network (spatial data not row-exchangeable)

"""
beta_summary_net = summary_networks.SummaryGNN(
    adjacency_matrix = W_full,
    gnn_dim = 32,
    compress_dim = 256,
    hidden_dim = 64,
    summary_dim = 16)
"""
beta_summary_net = summary_networks.SummaryIdentity()

# %% define inference network and amortizer for fixed effects

depth = 8
beta_inference_net = bf.networks.CouplingFlow(
    depth=depth,
    permutation = None,
    transform = "affine",   
    subnet_kwargs={
       "units": [256 for i in range(depth)],  # Widths of the hidden layers
       "activation": "swish",
       "dropout": False,
       "dropout_prob": 0.0
   }
)


# %% define workflow, combining beta summary and inference network into approximator

beta_workflow = bf.BasicWorkflow(
    simulator=simulator,
    adapter=beta_adapter,
    inference_network=beta_inference_net,
    summary_network=beta_summary_net,
    standardize=["inference_variables"]
)

# %% define beta checkpoint

output_fp = Path("checkpoints") / (model_name + "_beta_net.keras")

beta_checkpoint = keras.callbacks.ModelCheckpoint(
    filepath='checkpoints/' + model_name + '_beta.weights.h5',
    monitor='val_loss',
    save_best_only=True,
    save_weights_only=True
)

# %% start training!

history = beta_workflow.fit_online(epochs = 300, 
                              batch_size = 64,
                              iterations_per_epoch = 1000,
                              validation_data = 64,
                              callbacks = [bayesflow_helpers.CleanLRLogger(),
                                           beta_checkpoint],
                              checkpoint_path = "checkpoints")
bf.diagnostics.plots.loss(history)

# %% save beta network

output_fp.parent.mkdir(exist_ok=True)
beta_workflow.approximator.save(filepath=output_fp)

# %% load beta network

beta_approximator = keras.saving.load_model(output_fp)

"""
# %%

beta_lr_schedule = keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate=2.5e-5,
    decay_steps=250,
    decay_rate=0.9,
    staircase=True # If True, LR drops in "steps"; if False, it's a smooth curve
)
beta_optimizer = keras.optimizers.Adam(learning_rate = beta_lr_schedule)
beta_workflow = bf.BasicWorkflow(
    simulator=simulator,
    adapter=beta_adapter,
    inference_network=beta_approximator.inference_network,
    optimizer = beta_optimizer,
    summary_network=beta_approximator.summary_network,
    standardize=["inference_variables"]
)
beta_workflow.approximator.compile(optimizer = beta_optimizer)

# %%

history = beta_workflow.fit_online(
    epochs = 50, 
    batch_size = 64,
    iterations_per_epoch = 1000,
    validation_data = 64,
    callbacks = [bayesflow_helpers.CleanLRLogger()],
    standardize = ['inference_variables']
)

"""

# %% check diagnostics

num_samples = 5000
val_sims = simulator.sample(200)
post_draws = beta_approximator.sample(conditions=val_sims, num_samples=num_samples)
post_draws.keys()

par_names = [rf"$\beta_{{{i}}}$" for i in range(p + 1)]
f = bf.diagnostics.plots.pairs_posterior(
    estimates=post_draws, 
    targets=val_sims,
    dataset_id=0,
    variable_names=par_names,
)

f = bf.diagnostics.plots.recovery(
    estimates=post_draws, 
    targets=val_sims,
    variable_names=par_names
)
f.savefig(output_dir + "beta_recovery.png")

f = bf.diagnostics.plots.calibration_histogram(
    estimates=post_draws, 
    targets=val_sims,
    variable_names=par_names
)
f.savefig(output_dir + "beta_calibration.png")

# %%

f = bf.diagnostics.plots.calibration_ecdf(
    estimates=post_draws, 
    targets=val_sims,
    variable_names=par_names,
    difference=True,
    rank_type="distance"
)


# %% incorporate beta network post sd into simulated corrupted residuals

beta_postsd = np.std(post_draws['beta'], axis = 1)
beta_avgpostsd = np.mean(beta_postsd, axis = 0)
beta_noise_sd = beta_avgpostsd * 1.2

prior, likelihood, _ = bym2_sim.BYM2_simulators(Lambda_scaled,
                                             A_scaled, A_scaled, lambda_rho, p,
                                             rng = rng,
                                             corrupt_residual = True,
                                             beta_loc = 0.0, beta_sd = 10.0,
                                             beta_noise_sd = beta_noise_sd,
                                             fix_X = True, X = X_scaled)
simulator = bf.simulators.SequentialSimulator([
    bf.simulators.LambdaSimulator(prior, is_batched=True),
    bf.simulators.LambdaSimulator(likelihood, is_batched=True)
])

# %% define summary network and adapter for variance parameters

var_adapter = (
    bf.Adapter()
    .convert_dtype("float64", "float32")
    .concatenate(["log_sigma2", "logit_rho"], into="inference_variables")
    .concatenate(["r"], into="summary_variables")
    # .concatenate(["X", "y", "X_mask", "y_mask"], into = "summary_variables")
)

# Graph neural network as summary network (spatial data not row-exchangeable)
var_summary_net = summary_networks.SummaryGNN(
    adjacency_matrix = W_full,
    gnn_dim = 32,
    compress_dim = 128,
    hidden_dim = 64,
    summary_dim = 32)

# %% define var inference network and amortizer

depth = 6
var_inference_net = bf.networks.CouplingFlow(
    depth=depth,
    permutation = None,
    transform = "affine", 
    subnet_kwargs={
       "units": [256 for i in range(depth)],  # Widths of the hidden layers
       "activation": "swish",
       "dropout": False,
       "dropout_prob": 0.0
   }
)

# %% define var workflow, combining summary and inference network into approximator

var_workflow = bf.BasicWorkflow(
    simulator=simulator,
    adapter=var_adapter,
    inference_network=var_inference_net,
    summary_network=var_summary_net,
    standardize=["inference_variables"]
)

# %% define var optimizer

# Define a schedule: Start at 5e-4, reduce by 10% every 1000 steps
var_lr_schedule = keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate=5e-4, 
    decay_steps=250, 
    decay_rate=0.99,
    staircase=True
)

var_checkpoint = keras.callbacks.ModelCheckpoint(
    filepath= 'checkpoints/' + model_name + '_var.weights.h5',
    monitor='val_loss',
    save_best_only=True,
    save_weights_only=True
)

var_optimizer = keras.optimizers.Adam(learning_rate=var_lr_schedule)

# %% start training variance network!

history = var_workflow.fit_online(epochs = 150, 
                                  batch_size = 64,
                                  iterations_per_epoch = 1000,
                                  validation_data = 64,
                                  optimizer = var_optimizer,
                                  callbacks = [bayesflow_helpers.CleanLRLogger(),
                                               var_checkpoint],
                                  checkpoint_path = "checkpoints")

# %% save variance approximator

var_fp = Path("checkpoints") /  (model_name + "_var_net.keras")
var_fp.parent.mkdir(exist_ok=True)
var_workflow.approximator.save(filepath=var_fp)

# %% load variance network

var_approximator = keras.saving.load_model(var_fp)

# %% diagnostic plots

bf.diagnostics.plots.loss(history)

# %% draw validation set

num_samples = 5000
val_sims = simulator.sample(200)
post_draws = var_approximator.sample(conditions=val_sims, num_samples=num_samples)
post_draws.keys()

# %% plot posterior samples for first dataset

par_names = [r"$\text{log}(\sigma^2)$", r"$\text{logit}(\rho)$"]
f = bf.diagnostics.plots.pairs_posterior(
    estimates=post_draws, 
    targets=val_sims,
    dataset_id=0,
    variable_names=par_names,
)

# %% recovery and SBC plot for variance network

f = bf.diagnostics.plots.recovery(
    estimates=post_draws, 
    targets=val_sims,
    variable_names=par_names
)
f.savefig(output_dir + "var_recovery.png")

f = bf.diagnostics.plots.calibration_ecdf(
    estimates=post_draws, 
    targets=val_sims,
    variable_names=par_names,
    difference=True,
    rank_type="distance"
)

f.savefig(output_dir + "var_calibration.png")

# %% sampling settings

n_samples = 1000
data = simulator.sample(100)

# %% chain samples

post_draws = bayesflow_helpers.simulateChainSamples(
    n_samples, data, X_scaled, beta_approximator, var_approximator,
    var_batch_size = 10)

# %% plot posterior samples

par_names = [rf"$\beta_{{{i}}}$" for i in range(p + 1)]
par_names += r"$\text{log}(\sigma^2)$", r"$\text{logit}(\rho)$"
f = bf.diagnostics.plots.pairs_posterior(
    estimates=post_draws, 
    targets=data,
    dataset_id=0,
    variable_names=par_names,
)
f.savefig(output_dir + "chained_postsamples.png")

# %% check calibration plots

f = bf.diagnostics.plots.calibration_ecdf(
    estimates=post_draws, 
    targets=data,
    variable_names=par_names,
    difference=True,
    rank_type="distance"
)
f.savefig(output_dir + "chained_calibration.png")

# %% recovery plot for chained samples (should be from joint posterior)

f = bf.diagnostics.plots.recovery(
    estimates=post_draws, 
    targets=data,
    variable_names=par_names
)

f.savefig(output_dir + "chained_recovery.png")
