#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 28 20:26:54 2026

@author: 
"""

import numpy as np

def CAR_prior(n_samples, A, rng):
    # A: 2D square matrix where Q^{-1} = AA^{\T}
    # rng: output from np.random.default_rng(seed = seed)
    if A.shape[0] != A.shape[1]:
        raise AttributeError("RInv must be square")
    x = rng.normal(size = (n_samples, A.shape[0]))
    x = x @ A.transpose()
    return x

def rho_KLD(rho, Lambda):
    # rho: spatial proportion of variance in BYM2 model
    # Lambda: 1D numpy array of eigenvalues of spatial precision matrix
    n = Lambda.size
    kld = -n * rho / 2.0
    kld += np.sum(rho / Lambda) / 2.0
    kld -= np.sum(np.log(rho / Lambda + (1 - rho))) / 2.0
    return kld
    

def PC_prior(n_samples, lambda_rho, Lambda, rng):
    #rng = np.random.default_rng(seed = seed)
    rho_samples = np.zeros(n_samples)
    proposed_samples = np.zeros(n_samples)
    for i in range(0, n_samples):
        accept = False
        rho_star = rng.uniform()
        current_samples = 1
        while accept == False:
            kld = rho_KLD(rho_star, Lambda)
            d = np.sqrt(2.0 * kld)
            accept_logprob = -lambda_rho * d
            logu = np.log(rng.uniform())
            if logu <= accept_logprob:
                accept = True
            else: 
                rho_star = rng.uniform()
                current_samples += 1
        rho_samples[i] = rho_star
        proposed_samples[i] = current_samples
    return rho_samples, proposed_samples
    
def BYM2_prior(n_samples, p, Lambda, lambda_rho, 
               beta_loc = 0.0, beta_sd = 10.0,
               rng = np.random.default_rng()):
    # Lambda: 1D array of eigenvalues of spatial precision matrix
    # lambda_rho: PC penalty parameter for spatial variance proportion
    # p: number of predictors (beta will include p + 1 columns with intercept)
    beta = rng.normal(loc = beta_loc, scale = beta_sd, 
                      size = (n_samples, p + 1))
    # IG prior on sigma2
    #sigma2 = np.reciprocal(rng.gamma(shape = a0, scale = 1.0 / b0))
    # half normal prior on sigma2
    sigma2 = np.abs(rng.normal(loc = 0.0, scale = 10.0, size = n_samples))
    rho, proposed_samples = PC_prior(n_samples, lambda_rho, Lambda, rng)
    return dict(beta = beta, log_sigma2 = np.log(sigma2),
                logit_rho = np.log(rho / (1 - rho)))

def generate_CAR_covariates(n_samples, n, p, A_x, rng):
    X = np.zeros((n_samples, n, p))
    # generate covariates using conditional regressions
    for i in range(0, p):
        if (i == 0):
            linear_combo = np.zeros(shape = [n_samples, n])
        else:
            weights = rng.normal(loc = 0.0, scale = 1.0, size = [n_samples, i])    
            linear_combo = np.einsum('bij,bj->bi', X[:,:,:i], weights)
        noise = CAR_prior(n_samples, A_x, rng)
        X[:,:,i] = linear_combo + noise
    return X
    
def BYM2_likelihood(n_samples, beta, log_sigma2, logit_rho, Lambda, A_y, A_x,
                    X_fixed = None, # fixes X for all samples if input
                    rng = np.random.default_rng(),
                    simulate_missing = True,
                    missing_covariates_prob = 0.0,
                    missing_response_prob = 0.0):
    # beta: 1D array of regression coefficients (including intercept)
    # sigma2: total error variance
    # rho: spatial proportion of variance
    # Lambda: 1D array of eigenvalues of CAR precision matrix Q
    # A_y: 2D matrix, factor of response spatial covariance Q^{-1} = A_y A^{t}_y
    # A_x: 2D matrix, factor of response spatial covariance Q^{-1} = A_x A^{t}_x
    sigma2 = np.exp(log_sigma2)
    rho = 1.0 / (1.0 + np.exp(-logit_rho))
    n = Lambda.size
    p = beta.shape[-1] - 1
    if (X_fixed is None):
        X = generate_CAR_covariates(n_samples, n, p, A_x, rng)
    else:
        X = np.repeat(X_fixed[np.newaxis, ...], repeats = n_samples, axis = 0)
    # first compute mu = E[y | X, beta]
    mu = np.repeat(beta[:,0][:, np.newaxis], n, axis = 1)
    mu += mu + np.einsum('bij,bj->bi', X, beta[:,1:])
    # compute latent spatial effects gamma with CAR prior
    gamma = CAR_prior(n_samples, A_y, rng)
    gamma = np.einsum('bi,b->bi', gamma, np.sqrt(sigma2 * rho))
    # compute non-spatial error term
    eta = rng.normal(loc = 0.0, scale = 1.0, size = (n_samples, n))
    eta = np.einsum('bi,b->bi', eta, np.sqrt(sigma2 * (1 - rho)))
    # add in error terms
    y = mu + gamma + eta
    # reshape to 2D array
    y = y[:,:,np.newaxis]
    mu = mu[:,:,np.newaxis]
    # mask covariates and response randomly
    X_mask = rng.uniform(size = X.shape) < missing_covariates_prob
    X[X_mask] = -9001
    y_mask = rng.uniform(size = y.shape) < missing_response_prob
    y[y_mask] = -9001
    # convert masks to integer indices
    X_mask = X_mask.astype('int')
    y_mask = y_mask.astype('int')
    return dict(X = X, X_mask = X_mask, y = y, y_mask = y_mask, mu = mu)

def BYM2_simulators(Lambda, A_y, A_x, lambda_rho, p,
                    rng = np.random.default_rng(),
                    simulate_missing = True,
                    missing_covariates_prob = 0.0,
                    missing_response_prob = 0.0,
                    beta_loc = 0.0, beta_sd = 10.0, fix_X = False):
    n = Lambda.size
    if fix_X:
        X = generate_CAR_covariates(1, n, p, A_x, rng).squeeze()
    else:
        X = None
    def prior(batch_size):
        res = BYM2_prior(batch_size[0], p, Lambda, lambda_rho,
                         beta_loc = beta_loc, beta_sd = beta_sd,
                         rng = rng)
        return res
    def likelihood(batch_size, beta, log_sigma2, logit_rho):
        res = BYM2_likelihood(batch_size[0], beta, log_sigma2, logit_rho,
                              Lambda, A_y, A_x,
                              rng = rng, X_fixed = X,
                              simulate_missing = True,
                              missing_covariates_prob = missing_covariates_prob,
                              missing_response_prob = missing_response_prob)
        return res
    return prior, likelihood, X

if __name__ == "__main__":
    import shp_reader
    import matplotlib.pyplot as plt
    # set filepath to US county shapefile
    print("Reading in US county shapefile...")
    fp = "../data/cb_2017_us_county_500k/cb_2017_us_county_500k.shp"
    # read in shapefile and adjancency matrix of mainland US counties
    us_mainland, W = shp_reader.read_US_shapefile(fp)
    print("Done! Constructing CAR covariance matrix using US county map...")
    # convert contiguity object to dense matrix
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
    print("Done! Generating sample from CAR prior...")
    rng = np.random.default_rng(seed = 11301)
    z = CAR_prior(1, A_scaled, rng)
    us_mainland["z"] = z.squeeze()
    us_mainland.plot(column = "z", legend = True)
    plt.title("Simulated CAR Random Effects")
    print("Done! Generating 10 samples from PC prior on",
          "spatial variance proportion...")
    n_samples = (10,)
    rho, proposed_samples = PC_prior(n_samples = n_samples[0], lambda_rho = 0.005, 
                                     Lambda = Lambda_scaled, rng = rng)
    print("Sampled rho:", rho, "| Proposed samples:", proposed_samples)
    print("Simulating from BYM2 prior with 6 covariates...")
    p = 6
    lambda_rho = 0.003
    prior, likelihood, fix_X = BYM2_simulators(Lambda_scaled, A_scaled, 
                                               A_scaled, lambda_rho, p,
                                               rng, True, 0.0, 0.0)
    params = prior(n_samples)
    print(params)
    print("Simulating from BYM2 likelihood using sampled parameters...")
    data = likelihood(n_samples,
                      params["beta"], params["log_sigma2"], params["logit_rho"])
    print("Proportion of missing covariates:", np.mean(data["X_mask"]))
    print("Proportion of missing responses:", np.mean(data["y_mask"]))
    print(data)
    # plot first dataset
    y_masked = data["y"][1,:].copy()
    y_masked[data["y_mask"][1,:].astype("bool")] = np.nan
    us_mainland["y"] = y_masked
    us_mainland.plot(column = "y", legend = True)
    plt.title("Simulated Response")
    
    
