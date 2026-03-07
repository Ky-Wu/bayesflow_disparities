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
import bayesflow as bf
import keras
from src import summary_networks
from src import BYM2_simulators as bym2_sim
from src import shp_reader
from src import bayesflow_helpers
from src.spatial_covariance import scaled_CAR
from pathlib import Path
import numpy as np

# US county shapefile
fp = "data/cb_2017_us_county_500k/cb_2017_us_county_500k.shp"
# define main run parameters
p = 7
fix_X = True
model_name = "ca_fixedXp7"

# %% read in shapefile and compute scaled CAR covariance

ca_shp, W = shp_reader.read_CA_shapefile(fp)
W_full = W.full()[0]
Q_scaled, Sigma_scaled, Lambda_scaled, A_scaled = scaled_CAR(W_full)
n = Q_scaled.shape[0]

# %% define generative model

# prior hyperparameters
lambda_rho = 0.03

prior, likelihood, X_fixed = bym2_sim.BYM2_simulators(Lambda_scaled,
                                             A_scaled, A_scaled, lambda_rho, p,
                                             simulate_missing=True,
                                             missing_covariates_prob=0.0,
                                             missing_response_prob=0.0,
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
beta_summary_net = summary_networks.SummaryGNN(W_full, 1, 32, 32, 16)

# %% define inference network and amortizer for fixed effects

beta_inference_net = bf.networks.CouplingFlow(
    num_params = p + 1,
    num_coupling_layers=6,
    coupling_settings={
        "dense_args": {'kernel_regularizer' : None,
                       'units' : 256}, 
        "dropout": False}
)

# %% define workflow, combining beta summary and inference network into approximator

beta_workflow = bf.BasicWorkflow(
    simulator=simulator,
    adapter=beta_adapter,
    inference_network=beta_inference_net,
    summary_network=beta_summary_net,
    standardize=["inference_variables"]
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

history = beta_workflow.fit_online(epochs = 100, 
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

f = bf.diagnostics.plots.calibration_histogram(
    estimates=post_draws, 
    targets=val_sims,
    variable_names=par_names
)


# %% define summary network and adapter for variance parameters

var_adapter = (
    bf.Adapter()
    .convert_dtype("float64", "float32")
    .concatenate(["log_sigma2", "logit_rho"], into="inference_variables")
    .concatenate(["y", "mu"], into="summary_variables")
    # .concatenate(["X", "y", "X_mask", "y_mask"], into = "summary_variables")
)

# Graph neural network as summary network (spatial data not row-exchangeable)
var_summary_net = summary_networks.SummaryGNN(W_full, 2, 64, 64, 32)

# %% define var inference network and amortizer

var_inference_net = bf.networks.CouplingFlow(
    num_params = 2,
    num_coupling_layers=8,
    coupling_settings={
        "dense_args": {'kernel_regularizer' : None,
                       'units' : 256}, 
        "dropout": False}
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

history = var_workflow.fit_online(epochs = 100, 
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

f = bf.diagnostics.plots.calibration_histogram(
    estimates=post_draws, 
    targets=val_sims,
    variable_names=par_names
)

# %% chain networks together

num_samples = 10000
data = simulator.sample(3)
samples = bayesflow_helpers.simulateChainSamples(
    num_samples, data, X_fixed, beta_approximator, var_approximator)

# %% plot draws for sigma2

logsigma2_draws = samples["log_sigma2"][0,:,:].squeeze()
bayesflow_helpers.plot_sim_samples(logsigma2_draws, data["log_sigma2"][0],
                 np.exp, label = r"$\sigma^2$")

# %% plot draws for rho

logitrho_draws = samples["logit_rho"][0,:,:].squeeze()
bayesflow_helpers.plot_sim_samples(logitrho_draws, data["logit_rho"][0],
                                   lambda x: 1.0 / (1.0 + np.exp(-x)),
                                   label = r"$\rho$")

