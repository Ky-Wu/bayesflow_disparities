#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar  3 18:38:42 2026

@author: kylel
"""

# %% load libraries and config
import sys
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

def str_to_bool(s):
    s = s.strip().lower() # Remove spaces and convert to lowercase
    if s in ('yes', 'true', 't', '1', 'on'): # Define "truthiness"
        return True
    elif s in ('no', 'false', 'f', '0', 'off'): # Define "falsiness"
        return False
    else:
        raise ValueError(f"'{s}' is not a valid boolean string")
                         
fp = sys.argv[1]
region = sys.argv[2]
p = int(sys.argv[3])
fix_X = str_to_bool(sys.argv[4])
model_name = sys.argv[5]
lambda_rho = float(sys.argv[6])
corrupt_residual = str_to_bool(sys.argv[7])
output_dir = sys.argv[8]

rng = np.random.default_rng(seed = 1130)

# %% read in shape file

if region == "CA":
    shp, W = shp_reader.read_CA_shapefile(fp)
elif region == "CAORWA":
    shp, W = shp_reader.read_CAORWA_shapefile(fp)
elif region == "US":
    shp, W = shp_reader.read_US_shapefile(fp)
else:
    raise Exception("region argument not in ['CA', 'CAORWA', 'US']")


# %% read in shapefile and compute scaled CAR covariance

W_full = W.full()[0]
Q_scaled, Sigma_scaled, Lambda_scaled, A_scaled = scaled_CAR(W_full)
n = Q_scaled.shape[0]

# %% define generative model

prior, likelihood, X_fixed = bym2_sim.BYM2_simulators(Lambda_scaled,
                                             A_scaled, A_scaled, lambda_rho, p,
                                             rng = rng,
                                             corrupt_residual = corrupt_residual,
                                             beta_noise_sd = 1.0,
                                             beta_loc = 0.0, beta_sd = 5.0,
                                             fix_X = fix_X)
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
#beta_summary_net = summary_networks.SummaryGNN(W_full, 32, 32, 16, 16)
beta_summary_net = summary_networks.SummaryIdentity()

# %% define inference network and amortizer for fixed effects

depth = 6
beta_inference_net = bf.networks.CouplingFlow(
    depth=depth,
    permutation = None,
    transform = "affine"
)

# %% define workflow, combining beta summary and inference network into approximator

beta_workflow = bf.BasicWorkflow(
    simulator=simulator,
    adapter=beta_adapter,
    inference_network=beta_inference_net,
    summary_network=beta_summary_net,
    #standardize=["inference_variables"]
)

# %% define beta optimizer

# Define a schedule: Start at 5e-4, reduce by 10% every 1000 steps
beta_lr_schedule = keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate=5e-4, 
    decay_steps=250, 
    decay_rate=0.99,
    staircase=True
)
        
beta_checkpoint = keras.callbacks.ModelCheckpoint(
    filepath='checkpoints/' + model_name + '_beta.weights.h5',
    monitor='val_loss',
    save_best_only=True,
    save_weights_only=True
)

beta_optimizer = keras.optimizers.Adam(learning_rate=beta_lr_schedule)

# %% start training!

history = beta_workflow.fit_online(epochs = 150, 
                              batch_size = 64,
                              iterations_per_epoch = 1000,
                              validation_data = 64,
                              optimizer = beta_optimizer,
                              callbacks = [bayesflow_helpers.CleanLRLogger(),
                                           beta_checkpoint],
                              checkpoint_path = "checkpoints")

# %% save beta network

output_fp = Path("checkpoints") / (model_name + "_beta_net.keras")
output_fp.parent.mkdir(exist_ok=True)
beta_workflow.approximator.save(filepath=output_fp)

# %% load beta network

beta_approximator = keras.saving.load_model(output_fp)

# %% save covariate matrix if fixed

X_fixed_outputfp = Path("checkpoints") / (model_name + "_X.npy")
np.save(X_fixed_outputfp, X_fixed)

# %% check diagnostics

bf.diagnostics.plots.loss(history)

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

# %% incorporate beta network post sd into simulated corrupted residuals

beta_postsd = np.std(post_draws['beta'], axis = 1)
beta_avgpostsd = np.mean(beta_postsd, axis = 0)
beta_noise_sd = beta_avgpostsd * 1.2

prior, likelihood, X_fixed = bym2_sim.BYM2_simulators(Lambda_scaled,
                                             A_scaled, A_scaled, lambda_rho, p,
                                             rng = rng,
                                             corrupt_residual = True,
                                             beta_loc = 0.0, beta_sd = 5.0,
                                             beta_noise_sd = beta_noise_sd,
                                             fix_X = True, X = X_fixed)
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
var_summary_net = summary_networks.SummaryGNN(W_full, 32, 32, 16, 8)
#var_summary_net = summary_networks.SummaryIdentity()

# %% define var inference network and amortizer

var_inference_net = bf.networks.CouplingFlow(
    depth=6,
    permutation = None,
    transform = "affine"
)

# %% define var workflow, combining summary and inference network into approximator

var_workflow = bf.BasicWorkflow(
    simulator=simulator,
    adapter=var_adapter,
    inference_network=var_inference_net,
    summary_network=var_summary_net,
    #standardize=["inference_variables"]
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

history = var_workflow.fit_online(epochs = 200, 
                                  batch_size = 64,
                                  iterations_per_epoch = 1000,
                                  validation_data = 64,
                                  #optimizer = var_optimizer,
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
post_draws = var_workflow.sample(conditions=val_sims, num_samples=num_samples)
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

f = bf.diagnostics.plots.calibration_histogram(
    estimates=post_draws, 
    targets=val_sims,
    variable_names=par_names
)
f.savefig(output_dir + "var_calibration.png")

# %% sampling settings

n_samples = 500
data = simulator.sample(200)

# %% chain samples

post_draws = bayesflow_helpers.simulateChainSamples(
    n_samples, data, X_fixed, beta_approximator, var_approximator,
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
    targets=val_sims,
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
