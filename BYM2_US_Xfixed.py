#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar  3 20:58:33 2026

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
from pathlib import Path
import numpy as np
import os
import jax

# US county shapefile
fp = "data/cb_2017_us_county_500k/cb_2017_us_county_500k.shp"

# %% read in shapefile and compute scaled CAR covariance

ca_shp, W = shp_reader.read_US_shapefile(fp)

W_full = W.full()[0]
D = np.diag(np.sum(W_full, axis=1))
Q = D - 0.99 * W_full
n = Q.shape[0]
# Eigendecomposition (Q = P * diag(Lambda) * P^{T})
Lambda, P = np.linalg.eig(Q)
# A = P * Lambda^{-1/2}
# Q^{-1} = AA^{T}
A = P @ np.diag(np.pow(Lambda, -0.5))
Sigma = A @ A.transpose()
scaling_factor = np.exp(np.mean(np.log(Sigma.diagonal())))
Q_scaled = scaling_factor * Q
Sigma_scaled = (1.0 / scaling_factor) * Sigma
A_scaled = np.pow(scaling_factor, -0.5) * A
Lambda_scaled = Lambda * scaling_factor

del W, D, Q, Lambda, A, Sigma

# %% define generative model

# prior hyperparameters
lambda_rho = 0.0335
# number of covariates (predictors)
p = 7

prior, likelihood, X_fixed = bym2_sim.BYM2_simulators(Lambda_scaled,
                                             A_scaled, A_scaled, lambda_rho, p,
                                             simulate_missing=True,
                                             missing_covariates_prob=0.0,
                                             missing_response_prob=0.0,
                                             beta_loc = 0.0, beta_sd = 5.0,
                                             fix_X = True)
simulator = bf.simulators.SequentialSimulator([
    bf.simulators.LambdaSimulator(prior, is_batched=True),
    bf.simulators.LambdaSimulator(likelihood, is_batched=True)
])

# sample from joint distribution
data = simulator.sample(50)
print("Data:", data)
print("Shapes:", {k: v.shape for k, v in data.items()})

# %% define summary network and adapter

adapter = (
    bf.Adapter()
    # .as_set(["X", "y", "X_mask", "y_mask"])
    .as_set(["y"])
    .convert_dtype("float64", "float32")
    .concatenate(["beta", "log_sigma2", "logit_rho"], into="inference_variables")
    .concatenate(["y"], into="summary_variables")
    # .concatenate(["X", "y", "X_mask", "y_mask"], into = "summary_variables")
)

# Graph neural network as summary network (spatial data not row-exchangeable)
summary_net = summary_networks.PoolSummaryGNN(
    adjacency_matrix = W_full,
    n_features = 1, 
    gnn_dim = 32,
    hidden_dim = 128,
    summary_dim = 32)

summary_net = summary_networks.SummaryGNN(W_full, 1, compress_dim = 64,
                                          gnn_dim = 32, summary_dim = 16)
# %% define inference network and amortizer

inference_net = bf.networks.CouplingFlow(
    num_params=p + 3,
    num_coupling_layers=10,
    coupling_settings={
        "dense_args": {'kernel_regularizer' : None,
                       'units' : 256}, 
        "dropout": False}
)

# %% define workflow, combining summary and inference network into approximator

workflow = bf.BasicWorkflow(
    simulator=simulator,
    adapter=adapter,
    inference_network=inference_net,
    summary_network=summary_net,
    standardize=["inference_variables"]
)

# %% define optimizer

# Define a schedule: Start at 5e-4, reduce by 10% every 1000 steps
lr_schedule = keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate=5e-4, 
    decay_steps=250, 
    decay_rate=0.99,
    staircase=True
)

class CleanLRLogger(keras.callbacks.Callback):
    def on_epoch_begin(self, epoch, logs=None):
        # 1. Get the current step
        step = self.model.optimizer.iterations
        lr_obj = self.model.optimizer.learning_rate
        
        # 2. Extract value safely
        if hasattr(lr_obj, '__call__'):
            current_lr = lr_obj(step)
        else:
            current_lr = lr_obj
            
        # 3. Inject into the logs dictionary so Keras prints it for us
        # We use float() to ensure compatibility with the progress bar
        if logs is not None:
            logs['lr'] = float(current_lr)
            
    def on_epoch_end(self, epoch, logs=None):
        # Repeat for the end of the epoch to ensure the final summary is correct
        lr_obj = self.model.optimizer.learning_rate
        if hasattr(lr_obj, '__call__'):
            current_lr = lr_obj(self.model.optimizer.iterations)
        else:
            current_lr = lr_obj
            
        if logs is not None:
            logs['lr'] = float(current_lr)
        
        
checkpoint = keras.callbacks.ModelCheckpoint(
    filepath='checkpoints/bym2_us_p7_Xfixed.weights.h5',
    monitor='val_loss',
    save_best_only=True,
    save_weights_only=True
)

optimizer = keras.optimizers.Adam(learning_rate=lr_schedule)

# %% start training!

history = workflow.fit_online(epochs = 300, 
                              batch_size = 64,
                              iterations_per_epoch = 1000,
                              validation_data = 64,
                              optimizer = optimizer,
                              callbacks = [CleanLRLogger(), checkpoint])

"""
# %% keep training...

lr_schedule = keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate=1e-6,
    decay_steps=1000, 
    decay_rate=0.9,
    staircase=True
)

optimizer = keras.optimizers.Adam(learning_rate=lr_schedule)

history = workflow.fit_online(epochs=500, batch_size=64,
                              iterations_per_epoch=1000,
                              validation_data = 64,
                              optimizer = optimizer)
"""

# %% plot loss

bf.diagnostics.plots.loss(history)

# %% draw validation set

num_samples = 1000
val_sims = simulator.sample(200)
post_draws = workflow.sample(conditions=val_sims, num_samples=num_samples)
post_draws.keys()

# %% plot posterior samples for first dataset

par_names = [r"$\beta_0$", r"$\beta_1$", r"$\beta_2$", r"$\beta_3$",
             r"$\beta_4$", r"$\beta_5$", r"$\beta_6$", r"$\beta_7$",
             r"$\text{log}(\sigma^2)$", r"$\text{logit}(\rho)$"]
#par_names = [r"$\beta_0$", r"$\beta_1$", 
#             r"$\text{log}(\sigma^2)$", r"$\text{logit}(\rho)$"]
#par_names = [r"$\beta_0$", r"$\beta_1$", r"$\beta_2$", r"$\beta_3$",
#             r"$\text{log}(\sigma^2)$", r"$\text{logit}(\rho)$"]
f = bf.diagnostics.plots.pairs_posterior(
    estimates=post_draws, 
    targets=val_sims,
    dataset_id=0,
    variable_names=par_names,
)

# %% recovery and SBC plot

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

# %% save bayesflow approximator

output_fp = Path("checkpoints") / "bym2_spatialgnn_p3_fixedX.keras"
output_fp.parent.mkdir(exist_ok=True)
workflow.approximator.save(filepath=output_fp)

# %% load in

history = workflow.fit_online(epochs=100, batch_size=64,
                              iterations_per_epoch=250,
                              validation_data = 64)