#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  2 11:32:40 2026

@author: kylel
"""

# %% load libraries and config

import bayesflow as bf
import keras
from src import flattening_net
from src import BYM2_simulators as bym2_sim
from src import shp_reader
from pathlib import Path
import numpy as np
import os
import jax
jax.config.update("jax_enable_x64", False)
os.environ["KERAS_BACKEND"] = "jax"


# US county shapefile
fp = "data/cb_2017_us_county_500k/cb_2017_us_county_500k.shp"

# %% read in shapefile and compute scaled CAR covariance

us_mainland, W = shp_reader.read_US_shapefile(fp)

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

del W, W_full, D, Q, Lambda, A, Sigma

# %% define generative model

# prior hyperparameters
lambda_rho = 0.00335
# number of covariates (predictors)
p = 6

prior, likelihood = bym2_sim.BYM2_simulators(Lambda_scaled,
                                             A_scaled, A_scaled, lambda_rho, p,
                                             simulate_missing=True,
                                             missing_covariates_prob=0.0,
                                             missing_response_prob=0.0)
simulator = bf.simulators.SequentialSimulator([
    bf.simulators.LambdaSimulator(prior, is_batched=True),
    bf.simulators.LambdaSimulator(likelihood, is_batched=True)
])

# sample from joint distribution
data = simulator.sample(2)
print("Data:", data)
print("Shapes:", {k: v.shape for k, v in data.items()})

# %% define flattening summary network and adapter

adapter = (
    bf.Adapter()
    # .as_set(["X", "y", "X_mask", "y_mask"])
    .as_set(["X", "y"])
    .constrain("sigma2", lower=0)
    .constrain("rho", lower=0, upper=1.0)
    .convert_dtype("float64", "float32")
    .concatenate(["beta", "sigma2", "rho"], into="inference_variables")
    .concatenate(["X", "y"], into="summary_variables")
    # .concatenate(["X", "y", "X_mask", "y_mask"], into = "summary_variables")
)

# Total length = 2(p + 1)n
summary_net = flattening_net.FlatteningNet((n, (p + 1)))

# %% define inference network and amortizer

inference_net = bf.networks.CouplingFlow(
    num_params=p + 3,
    num_coupling_layers=12,
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
    standardize=["inference_variables", "summary_variables"]
)

lr_schedule = keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate=1e-4,
    decay_steps=10000,
    decay_rate=0.95
)


# %% start training!

history = workflow.fit_online(epochs=200, batch_size=64,
                              iterations_per_epoch=1000,
                              validation_data = 64,
                              optimizer=keras.optimizers.Adam(learning_rate=lr_schedule))

# %% 

bf.diagnostics.plots.loss(history)

# %% draw validation set

num_samples = 1000
val_sims = simulator.sample(200)
post_draws = workflow.sample(conditions=val_sims, num_samples=num_samples)
post_draws.keys()

# %% plot posterior samples for first dataset

par_names = [r"$\beta_0$", r"$\beta_1$", r"$\beta_2$", r"$\beta_3$",
             r"$\beta_4$", r"$\beta_5$", r"$\beta_6$", r"$\sigma^2$", r"$\rho$"]
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

output_fp = Path("checkpoints") / "bym2_example.keras"
output_fp.parent.mkdir(exist_ok=True)
workflow.approximator.save(filepath=output_fp)

# %% train more