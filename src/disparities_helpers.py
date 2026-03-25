#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: kylel
"""

import numpy as np
from scipy.linalg import solve_triangular

def sample_gamma_posterior(beta: np.array,
                           sigma2: np.array,
                           rho: np.array,
                           X: np.array,
                           y: np.array,
                           Lambda: np.array,
                           P: np.array,
                           rng = None):
    """
    

    Parameters
    ----------
    beta : np.array
        (n_s, p + 1)
    sigma2 : np.array
        (n_s, 1)
    rho : np.array
        (n_s, 1)
    X : np.array
        (N, p + 1)
    y : np.array
        (N, 1)
    Lambda : np.array
        (N,) eigenvalues of CAR precision matrix
    P : np.array
        (N, N) eigenvectors of CAR precision matrix

    Returns
    -------
    gamma : np.array
        (n_s, N)

    """
    N = Lambda.shape[0]
    n_s = beta.shape[0]
    gamma = y - X @ beta.T
    gamma = P.T @ gamma
    gamma = (1.0 / ((1.0 - rho) / rho * Lambda + 1.0)).T * gamma
    
    noise = (Lambda / rho)
    noise += (1.0 / (1.0 - rho))
    noise = np.pow(noise, -0.5)
    noise = noise * rng.standard_normal((n_s, N))
    noise = np.sqrt(sigma2) * noise
    
    gamma = P @ (gamma + noise.T)
    
    return gamma.T

def compute_std_diff(gamma, sigma2, rho, X, Lambda, P, edge_list):
    N = X.shape[0]
    Q_neghalf = P @ np.diag(np.pow(Lambda, -0.5)) @ P.T
    H = np.linalg.cholesky(X.T @ X, upper = False)
    H = solve_triangular(H, X.T, lower = True)
    H = H.T @ H
    I_H = np.eye(N) - H
    B = Q_neghalf @ I_H @ Q_neghalf
    D, O = np.linalg.eigh(B)
    U = Q_neghalf @ O
    phi = gamma / np.sqrt(sigma2 * rho)
    squared_contrasts = [np.pow(U[:,i] - U[:,j], 2) for (i, j) in edge_list]
    squared_contrasts = np.stack(squared_contrasts, axis = 1)
    var_core = 1.0 / (1.0 + rho / (1.0 - rho) * D)
    sds = np.sqrt(var_core @ squared_contrasts)
    diffs = [(phi[:,i] - phi[:,j]) for (i, j) in edge_list]
    diffs = np.stack(diffs, axis = 1)
    return diffs / sds
    

def compute_diff_prob(d, epsilon):
    return np.mean(np.abs(d) > epsilon, axis = 0)

def conditional_entropy_loss(d, epsilon):
    v = compute_diff_prob(d, epsilon)
    neg_entropy = np.where((v == 0.0) | (v == 1.0),
                       0.0, v * np.log(v) + (1.0 - v) * np.log(1.0 - v))
    return np.sum(neg_entropy)

def FDR_estimate(v, t):
    v_s = v[v >= t]
    return np.sum(1.0 - v_s) / np.max([len(v_s), 1])

def compute_fdr_cutoff(v, delta):
    t_seq = np.sort(np.unique(v), axis = None)
    t_FDR = np.array([FDR_estimate(v, t) for t in t_seq])
    optim_t = np.min(np.append(t_seq[t_FDR <= delta], 1.0))
    fdr_estimate = FDR_estimate(v, optim_t)
    return optim_t, fdr_estimate