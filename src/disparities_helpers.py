#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: kylel
"""

import numpy as np
from numpy.typing import NDArray
from scipy.linalg import solve_triangular
from tqdm import tqdm

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
        (batch_size, n_s, p + 1)
    sigma2 : np.array
        (batch_size, n_s, 1)
    rho : np.array
        (batch_size, n_s, 1)
    X : np.array
        (N, p + 1)
    y : np.array
        (batch_size, N, 1)
    Lambda : np.array
        (N,) eigenvalues of CAR precision matrix
    P : np.array
        (N, N) eigenvectors of CAR precision matrix

    Returns
    -------
    gamma : np.array
        (batch_size, n_s, N)

    """
    batch_size, n_s, _ = beta.shape
    N = Lambda.shape[0]
    p = X.shape[1] - 1
    
    gamma = y - X @ beta.transpose((0,2,1))
    gamma = P.T @ gamma
    gamma = (1.0 / ((1.0 - rho) / rho * Lambda + 1.0)).transpose((0,2,1)) * gamma
    
    noise = (Lambda / rho)
    noise += (1.0 / (1.0 - rho))
    noise = np.pow(noise, -0.5)
    noise = noise * rng.standard_normal((batch_size, n_s, N))
    noise = np.sqrt(sigma2) * noise
    
    gamma = P @ (gamma + noise.transpose((0, 2, 1)))
    
    return gamma.transpose((0, 2, 1))

def compute_std_diff(gamma: NDArray[tuple[int, int, int]],
                     sigma2: NDArray[tuple[int, int, 1]],
                     rho: NDArray[tuple[int, int, 1]],
                     X: NDArray[tuple[int, int]],
                     Lambda: NDArray[tuple[int]],
                     P: NDArray[tuple[int, int]],
                     edge_list: list[list[int]]):
    """
    

    Parameters
    ----------
    gamma : NDArray[tuple[int, int, int]]
        (batch_size, n_s, N) array of random effects
    sigma2 : NDArray[tuple[int, int, 1]]
        (batch_size, n_s, 1) array of sigma2 (total error variance) samples
    rho : NDArray[tuple[int, int, 1]]
        (batch_size, n_s, 1) array of rho (spatial proprtion of variance) samples
    X : NDArray[tuple[int, int]]
        (N, p + 1) matrix of covariates with intercept
    Lambda : NDArray[tuple[int]]
        (N,) array of eigenvalues of CAR precision matrix
    P : NDArray[tuple[int, int]]
        (N, N) array of eigenvectors of CAR precision matrix
    edge_list : list[list[int]]
        k-length list of graph edges, indexed by rows of X

    Returns
    -------
    ()
        samples of standardized differences, defined as absolute differences in
        random effects, under a non-informative prior on beta
        (|\gamma_i - \gamma_j|) / \Var(|\gamma_i - \gamma_j| | y, sigma^2, rho)

    """
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
    diffs = [(phi[:,:,i] - phi[:,:,j]) for (i, j) in edge_list]
    diffs = np.stack(diffs, axis = 2)
    return diffs / sds

def gamma_diffs_marginalvar(gamma: NDArray[tuple[int, int, int]],
                            sigma2: NDArray[tuple[int, int, 1]],
                            rho: NDArray[tuple[int, int, 1]],
                            edge_list: list[list[int]]):
    phi = gamma / np.sqrt(sigma2 * rho)
    diffs = [(phi[:,:,i] - phi[:,:,j]) for (i, j) in edge_list]
    diffs = np.stack(diffs, axis = 2)
    stds = np.std(diffs, axis = 1, keepdims = True)
    return diffs / stds

def conjugate_prior_gamma_diffs(gamma: NDArray[tuple[int, int, int]],
                                sigma2: NDArray[tuple[int, int, 1]],
                                rho: NDArray[tuple[int, int, 1]],
                                X: NDArray[tuple[int, int]],
                                Lambda: NDArray[tuple[int]],
                                P: NDArray[tuple[int, int]],
                                tau2_beta: float,
                                edge_list: list[list[int]]):
    """
    

    Parameters
    ----------
    gamma : NDArray[tuple[int, int, int]]
        (batch_size, n_s, N) array of random effects
    sigma2 : NDArray[tuple[int, int, 1]]
        (batch_size, n_s, 1) array of sigma2 (total error variance) samples
    rho : NDArray[tuple[int, int, 1]]
        (batch_size, n_s, 1) array of rho (spatial proprtion of variance) samples
    X : NDArray[tuple[int, int]]
        (N, p + 1) matrix of covariates with intercept
    Lambda : NDArray[tuple[int]]
        (N,) array of eigenvalues of CAR precision matrix
    P : NDArray[tuple[int, int]]
        (N, N) array of eigenvectors of CAR precision matrix
    edge_list : list[list[int]]
        k-length list of graph edges, indexed by rows of X

    Returns
    -------
    ()
        samples of standardized differences, defined as absolute differences in
        random effects, under a conditionally conjugate normal prior on beta
        (|\gamma_i - \gamma_j|) / \Var(|\gamma_i - \gamma_j| | y, sigma^2, rho)

    """
    
    # compute diffs
    batch_size, n_s, N = gamma.shape
    p = X.shape[1] - 1
    phi = gamma / np.sqrt(sigma2 * rho)
    diffs = [(phi[:,:,i] - phi[:,:,j]) for (i, j) in edge_list]
    diffs = np.stack(diffs, axis = 2)
    
    ### Compute Variance of Differences ###
    
    ## First Variance Term ##
    # Step 1: Precompute W2 = P.T @ V -> shape (N, k)
    W = np.stack([(P[i,:] - P[j,:]) for (i, j) in edge_list], axis = 1)
    W2 = W ** 2
    
    # Step 2: Square the diagonals
    d = (Lambda + rho / (1.0 - rho))
    d2 = d ** 2           # (batch_size, n_S, N)

    # Step 3: Single matmul gives all results
    variance = d2 @ W2      # (batch_size, n_S, N) @ (batch_size, N, k) → (batch_size, n_S, k)
    
    ## Second Variance Term ##
    PtX = P.T @ X
    XtX = np.broadcast_to(X.T @ X, (n_s, p + 1, p + 1)).astype(np.float32)
    

    for b in tqdm(range(batch_size), desc="Computing variance batches"):
        c1 = ((1.0 - rho[b,:,:]) / rho[b,:,:]) * (1.0 + (1.0 - rho[b,:,:]) / tau2_beta)
        A1 = c1[:,:,np.newaxis] * XtX
        d = np.pow(Lambda + (rho[b,:, :] / (1.0 - rho[b,:,:])), -0.5)
        K1neghalfX = d[:,:,np.newaxis] * PtX
        XtK1invX = K1neghalfX.transpose((0,2,1)) @ K1neghalfX
        K2 = A1 - XtK1invX
        L = np.linalg.cholesky(K2, upper = False)
        LinvXtP = solve_triangular(L, PtX.T, lower = True, check_finite = False)
        V = (LinvXtP * d[:,np.newaxis,:])
        variance[b,:,:] += np.linalg.norm(V @ W, axis = 1) ** 2
    
    return diffs / np.sqrt(variance)
        

def compute_diff_prob(d: NDArray[tuple[int, int, int]],
                      epsilon: float):
    return np.mean(np.abs(d) > epsilon, axis = 1)

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

