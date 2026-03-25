#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import keras
import numpy as np
import bayesflow as bf
import matplotlib.pyplot as plt
from typing import Callable
import tqdm
from scipy.linalg import solve_triangular

def signed_log(x):
    out = np.sign(x) * np.log1p(np.abs(x))
    return out

def inverse_signed_log(x):
    out = np.sign(x) * (np.exp(np.abs(x)) - 1.0)
    return out

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

# for silencing internal bayesflow sampling
class SilentSampling:
    def __enter__(self):
        self.original_init = tqdm.tqdm.__init__
        # Force the 'disable' argument to True for every new tqdm bar
        tqdm.tqdm.__init__ = lambda *args, **kwargs: self.original_init(*args, **{**kwargs, 'disable': True})
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        # Restore tqdm so your training progress bar still works
        tqdm.tqdm.__init__ = self.original_init
        
def plot_sim_samples(draws, truth, backtransform, label="parameter"):
    """
    

    Parameters
    ----------
    draws : 1D array
        draws of parameter
    truth : scalary
        true parameter value
    backtransform : function
        backtransform function for samples and truth value
    parameter : string
        label of backtransformed parameters

    Returns
    -------
    plot of samples

    """
    draws_transformed = backtransform(draws)
    truth_transformed = np.round(backtransform(truth), 4).item()
    plt.figure()
    plt.hist(draws_transformed)
    plt.xlabel(label)
    plt.ylabel("Frequency")
    plt.title(rf"Posterior samples of {label}, true value: {truth_transformed}")
    ci95 = np.quantile(draws_transformed, [0.025, 0.975])
    plt.axvline(x = truth_transformed, color = "r",
                linestyle='--', label='Truth')
    plt.axvline(x = ci95[0].item(), color = "b",
                linestyle='--', label='95% CI (lower)')
    plt.axvline(x = ci95[1].item(), color = "b",
                linestyle='--', label='95% CI (upper)')
    plt.legend()
    
def backtransform_beta_samps(samps: dict, R_x = None):
    if 'beta' in samps:
        beta_draws = samps['beta']
    elif 'beta_signlog' in samps:
        beta_draws = inverse_signed_log(samps['beta_signlog'])
    elif 'beta_arcsinh' in samps:
        beta_draws = np.sinh(samps['beta_arcsinh'])
    elif 'theta' in samps:
        if R_x is None:
            raise ValueError("R_x must be supplied for backtransformation of theta")
        batch_size, n_samples, p_plus_1 = samps['theta'].shape
        theta = samps['theta'].reshape(-1, p_plus_1).T
        beta_draws = solve_triangular(R_x, theta, lower = False).T.reshape(
            batch_size, n_samples, p_plus_1)
    else:
        raise AttributeError("beta samples from approximator not recognized")
    return beta_draws
    
def simulate_network_residuals(n_samples: int,
                               y: np.array,
                               X: np.array,
                               beta_approx: bf.ContinuousApproximator,
                               R_x = None,
                               **kwargs):
    """
    
    Uses a trained fixed effects posterior approximator to draw posterior
    samples of the residuals y - X$\beta$.
    
    Parameters
    ----------
    n_samples : int
        number of samples to draw for each dataset
    y : np.array
        respopnse data of shape (batch_size, n, 1)
    X : np.array
        covariate data of shape (batch_size, n, p)
    beta_approx : ContinuousApproximator
        trained Bayesflow approximator, takes as input a dictionary containing
        'y' and samples from posterior distribution of $\beta$ given y. 
        Supported transformations: identity, signed log transform, arcsinh

    Raises
    ------
    AttributeError
        If beta_approx samples do not contain 'beta', 'beta_signlog', or
        'beta_arcsinh', the beta samples are not recognized and an error 
        is raised.

    Returns
    -------
    dict
        dictionary containing 'r', sampled residuals

    """    
    with SilentSampling():
        beta_samps = beta_approx.sample(conditions = dict(y = y),
                                        num_samples = n_samples)
    beta_draws = backtransform_beta_samps(beta_samps, R_x = R_x)
    # beta_draws shape : (batch_size, n_samples, p + 1)
    # X shape : (batch_size, n, p + 1)
    batch_size, n, _ = X.shape
    #mu_draws = np.einsum("bnp,bsp->bns", X, beta_draws)
    mu_draws = np.matmul(X, beta_draws.swapaxes(-1, -2)) 
    return dict(r_pred = y - mu_draws)
    
# warning: very slow
def ancestral_residual_simulator(prior: Callable,
                                 likelihood: Callable,
                                 beta_approx: bf.ContinuousApproximator
                                 ) -> Callable:
    def residual_sim(batch_size, y, X):
        return simulate_network_residuals(
            n_samples = batch_size[0], y = y, X = X, beta_approx = beta_approx)

    return bf.simulators.SequentialSimulator([
        bf.simulators.LambdaSimulator(prior, is_batched=True),
        bf.simulators.LambdaSimulator(likelihood, is_batched=True),
        bf.simulators.LambdaSimulator(residual_sim, is_batched=True)
    ])
    
def simulate_chain_samples(n_samples, data, X, beta_approx, var_approx,
                         var_batch_size = 10, R_x = None):
    """
    Function for obtaining samples from chained BYM2 networks.
    Can be quite memory-intensive if n is large and number of batches is large!

    Parameters
    ----------
    n_samples : int
        number of samples to draw
    data : dict
        dict containing {y: (batch_size, n, 1) np.array}
    X : np.array
        covariate matrix (n, p)
    beta_approx : ContinuousApproximator
        a Bayesflow approximator for BYM2 fixed effects
    var_approx : ContinuousApproximator
        a Bayesflow approximator for BYM2 variance parameters conditional on X\beta

    Returns
    -------
    dict{
        'y': (batch_size, n, 1) array
            original response data
        "beta": (batch_size, n_samples, p) array
            fixed effects posterior samples
        "mu": (batch_size, n_samples, n) array
            predictive mean posterior samples
        "log_sigma2": (batch_size, n_samples, 1) array
            log total error variance posterior samples
        "logit_rho": (batch_size, n_samples, 1) array
            logit spatial proportion of variance posterior samples
        }

    """
    batches = data["y"].shape[0]
    # (batch_size, n_samples, p + 1)
    beta_samps = beta_approx.sample(conditions = data, num_samples = n_samples)
    beta_draws = backtransform_beta_samps(beta_samps, R_x = R_x)
    # (batch_size, n_samples, n)
    y_flat = np.repeat(data['y'], repeats = n_samples, axis = 0)
    mu_flat = (beta_draws @ X.T).reshape(batches * n_samples, X.shape[0], 1)
    var_data = {
        "y" : y_flat,
        "mu": mu_flat,
        "r" : y_flat - mu_flat
    }
    var_draws = var_approx.sample(conditions = var_data, num_samples = 1,
                                  batch_size = var_batch_size)
    return {
        "beta" : beta_draws,
        "log_sigma2" : var_draws["log_sigma2"].reshape(batches, n_samples, 1),
        "logit_rho" : var_draws["logit_rho"].reshape(batches, n_samples, 1),
    }

if __name__ == "__main__":
    fp = "data/cb_2017_us_county_500k/cb_2017_us_county_500k.shp"
    region = "CA"
    p = 7
    fix_X = True
    model_name = "ca_fixedXp7"
    lambda_rho = 0.001
    corrupt_residual = True
    output_dir = "output/CA_chained_sim/"
    rng = np.random.default_rng(seed = 1130)

    beta_prior_sd = np.sqrt(10)
    sigma2_prior_sd = np.sqrt(10)
