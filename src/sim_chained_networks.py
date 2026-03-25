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
os.environ['JAX_PLATFORMS'] = 'cpu'
import bayesflow as bf
import keras
from src import summary_networks
from src import BYM2_simulators as bym2_sim
from src import shp_reader
from src import bayesflow_helpers as bfhelp
from src.spatial_covariance import scaled_CAR
from pathlib import Path
import numpy as np

rng = np.random.default_rng(seed = 1130)

# %% read command args

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
theta_isotropic = str_to_bool(sys.argv[8])
output_dir = sys.argv[9]


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

# %% prior hyperparameters

tau_beta = np.sqrt(n)
sigma2_prior_sd = np.sqrt(1)

# %% sample covariates and apply QR decomposition

X_fixed_outputfp = "checkpoints/" + model_name
if fix_X:
    X = bym2_sim.generate_CAR_covariates(1, n, p, A_scaled, rng).squeeze()
    Q_x, R_x = np.linalg.qr(X, mode = 'reduced')
    np.save(X_fixed_outputfp + "_X.npy", X)
    np.save(X_fixed_outputfp + "_Qx.npy", Q_x)
    np.save(X_fixed_outputfp + "_Rx.npy", R_x)
else:
    X = None
    Q_x = None
    R_x = None

# %% define generative model

prior, likelihood, _ = bym2_sim.BYM2_simulators(
    Lambda_scaled, A_scaled, A_scaled, lambda_rho, p,
    rng = rng,
    corrupt_residual = corrupt_residual,
    beta_loc = 0.0,
    tau_beta = tau_beta,
    sigma2_sd = sigma2_prior_sd,
    fix_X = fix_X, X = X, R_x = R_x,
    theta_isotropic = theta_isotropic)
simulator = bf.simulators.SequentialSimulator([
    bf.simulators.LambdaSimulator(prior, is_batched=True),
    bf.simulators.LambdaSimulator(likelihood, is_batched=True)
])

# sample from joint distribution
data = simulator.sample(50)
#print("Data:", data)
print("Data shapes:", {k: v.shape for k, v in data.items()})


# %% define summary network and adapter for fixed effects

beta_adapter = (
    bf.Adapter()
    .convert_dtype("float64", "float32")
    .concatenate(["theta"], into="inference_variables")
    .concatenate(["y"], into="summary_variables")
)

# Graph neural network as summary network (spatial data not row-exchangeable)
#beta_summary_net = summary_networks.SummaryGNN(W_full, 32, 32, 16, 16)
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

# compile approximator with optimizer if using custom learning rate schedule
"""
# %% define beta optimizer

# Define a schedule: Start at 5e-4, reduce by 10% every 1000 steps
beta_lr_schedule = keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate=5e-4, 
    decay_steps=250, 
    decay_rate=0.99,
    staircase=True
)
beta_optimizer = keras.optimizers.Adam(learning_rate=beta_lr_schedule)

"""
beta_checkpoint = keras.callbacks.ModelCheckpoint(
    filepath='checkpoints/' + model_name + '_beta.weights.h5',
    monitor='val_loss',
    save_best_only=True,
    save_weights_only=True
)

# %% start training!

history = beta_workflow.fit_online(epochs = 200, 
                              batch_size = 64,
                              num_batches_per_epoch = 100,
                              validation_data = 64,
                              callbacks = [bfhelp.CleanLRLogger(),
                                           beta_checkpoint],
                              checkpoint_path = "checkpoints")
bf.diagnostics.plots.loss(history)

# %% save beta network

output_fp = Path("checkpoints") / (model_name + "_beta_net.keras")
output_fp.parent.mkdir(exist_ok=True)
beta_workflow.approximator.save(filepath=output_fp)

# %% load beta network

beta_approximator = keras.saving.load_model(output_fp)

# %% check diagnostics

num_samples = 5000
val_sims = simulator.sample(500)
post_draws = beta_approximator.sample(conditions=val_sims,
                                      num_samples=num_samples,
                                      batch_size = 10)
post_draws.keys()

par_names = [rf"$\theta_{{{i}}}$" for i in range(p + 1)]
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
f.savefig(output_dir + "theta_recovery.png")

f = bf.diagnostics.plots.calibration_ecdf(
    estimates=post_draws, 
    targets=val_sims,
    variable_names=par_names,
    difference = True,
    rank_type="distance"
)
f.savefig(output_dir + "theta_calibration.png")

# %% mitigate exposure bias via ancestral network sampling

# too slow...
"""
var_simulator = bfhelp.ancestral_residual_simulator(
    prior = prior, likelihood = likelihood, beta_approx = beta_approximator)
"""

beta_draws = bfhelp.backtransform_beta_samps(post_draws, R_x = R_x)
beta_postcov = np.stack([np.cov(beta_draws[b].T) for b in range(beta_draws.shape[0])])
beta_noise_cov = beta_postcov.mean(axis=0)
beta_noise_R = np.linalg.cholesky(beta_noise_cov, upper = False)

"""
beta_draws = np.sinh(post_draws['beta_arcsinh'])
beta_postsd = np.std(beta_draws, axis = 1)
beta_avgpostsd = np.mean(beta_postsd, axis = 0)
beta_noise_sd = beta_avgpostsd
"""

prior, likelihood, _ = bym2_sim.BYM2_simulators(
    Lambda_scaled, A_scaled, A_scaled, lambda_rho, p,
    rng = rng,
    corrupt_residual = True,
    beta_loc = 0.0,
    tau_beta = tau_beta,
    beta_noise_R = beta_noise_R,
    sigma2_sd = sigma2_prior_sd,
    fix_X = True, X = X, R_x = R_x,
    theta_isotropic = theta_isotropic)
var_simulator = bf.simulators.SequentialSimulator([
    bf.simulators.LambdaSimulator(prior, is_batched=True),
    bf.simulators.LambdaSimulator(likelihood, is_batched=True)
])

# sample from joint distribution
data = var_simulator.sample(50)
#print("Data:", data)
print("Data shapes:", {k: v.shape for k, v in data.items()})

# %% define summary network and adapter for variance parameters

var_adapter = (
    bf.Adapter()
    .convert_dtype("float64", "float32")
    .concatenate(["log_sigma2", "logit_rho"], into="inference_variables")
    .concatenate(["r"], into="summary_variables")
)

# Graph neural network as summary network (spatial data not row-exchangeable)
#var_summary_net = summary_networks.SummaryGNN(W_full, 32, 32, 16, 8)
var_summary_net = summary_networks.SummaryIdentity()

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
    simulator=var_simulator,
    adapter=var_adapter,
    inference_network=var_inference_net,
    summary_network=var_summary_net,
    standardize=["inference_variables"]
)

# %% define var optimizer

var_checkpoint = keras.callbacks.ModelCheckpoint(
    filepath= 'checkpoints/' + model_name + '_var.weights.h5',
    monitor='val_loss',
    save_best_only=True,
    save_weights_only=True
)

# %% start training variance network!

history = var_workflow.fit_online(epochs = 200, 
                                  batch_size = 64,
                                  num_batches_per_epoch = 100,
                                  validation_data = 64,
                                  callbacks = [bfhelp.CleanLRLogger(),
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
val_sims = var_simulator.sample(200)
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

f = bf.diagnostics.plots.calibration_ecdf(
    estimates=post_draws, 
    targets=val_sims,
    variable_names=par_names,
    difference = True,
    rank_type="distance"
)
f.savefig(output_dir + "var_calibration.png")

# %% sampling settings

n_samples = 500
data = simulator.sample(200)

# %% chain samples

post_draws = bfhelp.simulate_chain_samples(
    n_samples, data, X, beta_approximator, var_approximator,
    R_x = R_x,
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

# %% assess coverage

f = bf.diagnostics.plots.coverage(
    estimates = post_draws,
    targets = data,
    variable_names = par_names,
    difference = True
)

f.savefig(output_dir + "chained_coverage.png")

# %% test for small fixed effects

prior, likelihood, _ = bym2_sim.BYM2_simulators(
    Lambda_scaled, A_scaled, A_scaled, lambda_rho, p,
    rng = rng,
    corrupt_residual = True,
    beta_loc = 0.0,
    tau_beta = 0.5,
    beta_noise_R = beta_noise_R,
    sigma2_sd = 1.0,
    fix_X = True, X = X, R_x = R_x,
    theta_isotropic = theta_isotropic)
small_simulator = bf.simulators.SequentialSimulator([
    bf.simulators.LambdaSimulator(prior, is_batched=True),
    bf.simulators.LambdaSimulator(likelihood, is_batched=True)
])

# sample from joint distribution
data = small_simulator.sample(300)

post_draws = bfhelp.simulate_chain_samples(
    n_samples, data, X, beta_approximator, var_approximator,
    R_x = R_x,
    var_batch_size = 10)

# %% plot small effect recovery

f = bf.diagnostics.plots.recovery(
    estimates=post_draws, 
    targets=data,
    variable_names=par_names
)

f.savefig(output_dir + "chained_recovery_smallbeta.png")

# %% assess coverage

f = bf.diagnostics.plots.coverage(
    estimates = post_draws,
    targets = data,
    variable_names = par_names,
    difference = True
)

f.savefig(output_dir + "chained_coverage_smallbeta.png")

# %% check small-beta calibration plots

f = bf.diagnostics.plots.calibration_ecdf(
    estimates=post_draws, 
    targets=data,
    variable_names=par_names,
    difference=True,
    rank_type="distance"
)
f.savefig(output_dir + "chained_calibration_smallbeta.png")