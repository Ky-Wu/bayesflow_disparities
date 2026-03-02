#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 24 13:20:37 2026

@author: kylel
"""

import jax
jax.config.update("jax_enable_x64", False)
import os
os.environ["KERAS_BACKEND"] = "jax"
import numpy as np
from pathlib import Path

import keras
import bayesflow as bf

# avoid scientific notation for outputs
np.set_printoptions(suppress=True)

def likelihood(beta, sigma, N):
    # x: predictor variable
    x = np.random.normal(0, 1, size=N)
    # y: response variable
    y = np.random.normal(beta[0] + beta[1] * x, sigma, size=N)
    return dict(y=y, x=x)

# %% Sample from likelihood

data_draws = likelihood(beta = [2, 1], sigma = 1, N = 3)
print(data_draws["y"].shape)
print(data_draws["y"])

# %% prior definition

def prior():
    # beta: regression coefficients (intercept, slope)
    beta = np.random.normal([2, 0], [3, 1])
    # sigma: residual standard deviation
    sigma = np.random.gamma(1, 1)
    return dict(beta=beta, sigma=sigma)

# %% sample from prior

prior_draws = prior()
print(prior_draws["beta"].shape)
print(prior_draws["beta"])

# %% simulator for N

def meta():
    # N: number of observation in a dataset
    N = np.random.randint(5, 15)
    return dict(N=N)

meta_draws = meta()
print(meta_draws["N"])

# %% Simulator composed of prior, likelihood, and meta function

simulator = bf.simulators.make_simulator([prior, likelihood], meta_fn=meta)

# Generate a batch of three training samples
sim_draws = simulator.sample(500)
print(sim_draws["N"])
print(sim_draws["beta"].shape)
print(sim_draws["sigma"].shape)
print(sim_draws["x"].shape)
print(sim_draws["y"].shape)

par_keys = ["beta", "sigma"]
par_names = [r"$\beta_0$", r"$\beta_1$", r"$\sigma$"]

f = bf.diagnostics.plots.pairs_samples(
    samples=sim_draws,
    variable_keys=par_keys,
    variable_names=par_names
)

# %% Adapter definition

adapter = (
    bf.Adapter()
    .broadcast("N", to="x")
    .as_set(["x", "y"])
    .constrain("sigma", lower=0)
    .sqrt("N")
    .convert_dtype("float64", "float32")
    .concatenate(["beta", "sigma"], into="inference_variables")
    .concatenate(["x", "y"], into="summary_variables")
    .rename("N", "inference_conditions")
)

processed_draws = adapter(sim_draws)
print(processed_draws["summary_variables"].shape)
print(processed_draws["inference_conditions"].shape)
print(processed_draws["inference_variables"].shape)

# %% Summary and inference networks

summary_network = bf.networks.SetTransformer(summary_dim=10)
inference_network = bf.networks.CouplingFlow(transform="spline")

# %% Workflow, start training

workflow = bf.BasicWorkflow(
    simulator=simulator,
    adapter=adapter,
    inference_network=inference_network,
    summary_network=summary_network,
    standardize=["inference_variables", "summary_variables"]
)

history = workflow.fit_online(epochs=50, batch_size=64, num_batches_per_epoch=200)

f = bf.diagnostics.plots.loss(history)

# %% Diagnostics 

# Set the number of posterior draws you want to get
num_samples = 1000

# Simulate validation data (unseen during training)
val_sims = simulator.sample(200)

# Obtain num_samples samples of the parameter posterior for every validation dataset
post_draws = workflow.sample(conditions=val_sims, num_samples=num_samples)

# post_draws is a dictionary of draws with one element per named parameters
post_draws.keys()


# %%

print(post_draws["beta"].shape)
post_draws["sigma"].min()
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

# %% SBC Diagnostics using joint log likelihood

def joint_log_likelihood(data):
    beta = data["beta"]
    sigma = data["sigma"]
    x = data["x"]
    y = data["y"]
    mean = beta[:,0][:,None] + beta[:,1][:,None] * x
    log_lik = np.sum(- (y - mean)**2 / (2 * sigma**2) - 0.5 * np.log(2 * np.pi) - np.log(sigma), axis=-1)
    return log_lik

joint_log_likelihood(val_sims).shape

val_sims_conditions = val_sims.copy()
val_sims_conditions.pop("beta")
val_sims_conditions.pop("sigma")
# N is only necessary for simulating data. It is not necessary to keep it for our test quantity.
val_sims_conditions.pop("N")
val_sims_conditions = keras.tree.map_structure(
    lambda tensor: np.repeat(np.expand_dims(tensor, axis=1), num_samples, axis=1), 
    val_sims_conditions
)
post_draws_conditions = post_draws | val_sims_conditions

# %% Check custom SBC using joint log likelihood

f = bf.diagnostics.plots.calibration_ecdf(
    estimates=post_draws_conditions, 
    targets=val_sims,
    test_quantities={  # this argument is new
        r"$\beta_0 + \beta_1$": lambda data: np.sum(data["beta"], axis=-1),
        r"$\log p(x|\beta,\sigma)$": joint_log_likelihood,
    },
    variable_keys=[],  # we just want to show SBC for the custom test quantities
    difference=True,
    rank_type="distance"
)

# %% Check posterior contraction

f = bf.diagnostics.plots.z_score_contraction(
    estimates=post_draws, 
    targets=val_sims,
    variable_keys=["beta"],
    variable_names=par_names[0:2]
)

# %% Saving and Loading Trained Network

# Recommended - full serialization (checkpoints folder must exist)
filepath = Path("checkpoints") / "regression.keras"
filepath.parent.mkdir(exist_ok=True)
workflow.approximator.save(filepath=filepath)

# Not recommended due to adapter mismatches - weights only
# approximator.save_weights(filepath="checkpoints/regression.h5")

# Load approximator
approximator = keras.saving.load_model(filepath)

# Again, obtain num_samples samples of the parameter posterior for every validation dataset
post_draws = approximator.sample(conditions=val_sims, num_samples=num_samples)

f = bf.diagnostics.plots.recovery(
    estimates=post_draws, 
    targets=val_sims,
    variable_names=par_names
)