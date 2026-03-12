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
output_dir = sys.argv[7]

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
                                             corrupt_residual = False,
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
#summary_net = summary_networks.SummaryGNN(W_full, 32, 32, 16, 16)
summary_net = summary_networks.SummaryIdentity()

# %% define inference network and amortizer

depth = 8
inference_net = bf.networks.CouplingFlow(
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

# %% define workflow, combining summary and inference network into approximator

workflow = bf.BasicWorkflow(
    simulator=simulator,
    adapter=adapter,
    inference_network=inference_net,
    summary_network=summary_net,
    standardize=["inference_variables", "summary_variables"]
)

# %% define optimizer

# Define a schedule: Start at 5e-4, reduce by 10% every 1000 steps
lr_schedule = keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate=5e-4, 
    decay_steps=250, 
    decay_rate=0.99,
    staircase=True
)

checkpoint = keras.callbacks.ModelCheckpoint(
    filepath='checkpoints/' + model_name + "_total.weights.h5",
    monitor='val_loss',
    save_best_only=True,
    save_weights_only=True
)

optimizer = keras.optimizers.Adam(learning_rate=lr_schedule)

# %% start training!

history = workflow.fit_online(epochs = 200, 
                              batch_size = 64,
                              iterations_per_epoch = 1000,
                              validation_data = 64,
                              optimizer = optimizer,
                              callbacks = [bayesflow_helpers.CleanLRLogger(),
                                           checkpoint],
                              checkpoint_path = "checkpoints")


 # %%  diagnostic plots

bf.diagnostics.plots.loss(history)

# %% draw validation set

num_samples = 5000
val_sims = simulator.sample(200)
post_draws = workflow.sample(conditions=val_sims, num_samples=num_samples)
post_draws.keys()

# %% plot posterior samples for first dataset

par_names = [rf"$\beta_{{{i}}}$" for i in range(p + 1)]
par_names += r"$\text{log}(\sigma^2)$", r"$\text{logit}(\rho)$"
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

f.savefig(output_dir + "recovery.png")

f = bf.diagnostics.plots.calibration_ecdf(
    estimates=post_draws, 
    targets=val_sims,
    variable_names=par_names,
    difference=True,
    rank_type="distance"
)
f.savefig(output_dir + "calibration.png")

# %% save bayesflow approximator

output_fp = Path("checkpoints") / (model_name + "_total.keras")
output_fp.parent.mkdir(exist_ok=True)
workflow.approximator.save(filepath=output_fp)