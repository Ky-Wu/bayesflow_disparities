#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 28 20:26:54 2026

@author: 
"""

import src.bayesflow_helpers as bfhelp
import numpy as np
from scipy.special import expit, logit
from scipy.linalg import solve_triangular
from scipy import sparse

def CAR_prior(n_samples: int,
              A: np.array,
              rng: np.random.default_rng = np.random.default_rng()) -> np.array:
    """

    Parameters
    ----------
    n_samples : int
        number of samples to draw from CAR distribution
    A : np.array
        asymmetric matrix factor A such that precision matrix Q^{-1} = AA^{T}
    rng : np.random.default_rng, optional
        Random number generator. The default is np.random.default_rng().

    Raises
    ------
    AttributeError
        Matrix factor A must be square

    Returns
    -------
    x : np.array
        array of samples

    """
    if A.shape[0] != A.shape[1]:
        raise AttributeError("A must be square")
    return rng.standard_normal(size = (n_samples, A.shape[0])) @ A.T

def rho_KLD(rho: float,
            Lambda: np.array,
            check_rho: bool = False) -> float:
    """
    

    Parameters
    ----------
    rho : float
        Spatial proportion of noise in [0,1]
    Lambda : np.array
        Array of eigenvalues of spatial precision matrix
    check_rho : bool, optional
        Whether to check rho is in [0,1]. The default is False.

    Raises
    ------
    ValueError
        rho, spatial proportion of variance, must lie in interval [0,1]

    Returns
    -------
    float
        Kullback-Leiber divergence term for penalized-complexity prior on rho.

    """
    if check_rho and not (0 <= rho <= 1):
            raise ValueError("rho must be in [0,1]")
    n = Lambda.size
    rho_over_Lambda = rho / Lambda
    kld = -n * rho
    kld += rho_over_Lambda.sum()
    kld -= np.sum(np.log(rho_over_Lambda + (1.0 - rho)))
    return kld * 0.5
    

def PC_prior(n_samples, lambda_rho, Lambda, rng = None):
    if rng is None:
        rng = np.random.default_rng()
    rho_samples = np.empty(n_samples)
    proposed_samples = np.empty(n_samples)
    for i in range(0, n_samples):
        n_proposed = 1
        while True:
            rho_star = rng.uniform()
            d = np.sqrt(2.0 * rho_KLD(rho_star, Lambda))
            accept_logprob = -lambda_rho * d
            logu = np.log(rng.uniform())
            if logu <= accept_logprob:
                break
            n_proposed += 1
        rho_samples[i] = rho_star
        proposed_samples[i] = n_proposed
    return rho_samples, proposed_samples
    
def BYM2_prior(n_samples: int,
               p: int,
               Lambda: np.array,
               lambda_rho: float, 
               beta_loc: float = 0.0,
               tau_beta: float = 1.0,
               beta_noise_R: np.array = None,
               sigma2_sd: float = 1.0,
               R_x: np.array = None,
               theta_isotropic: bool = False,
               rng = None):
    # Lambda: 1D array of eigenvalues of spatial precision matrix
    # lambda_rho: PC penalty parameter for spatial variance proportion
    # p: number of predictors (beta will include p + 1 columns with intercept)
    if rng is None:
        rng = np.random.default_rng()
    if R_x is None:
        R_x = np.eye(p + 1)
    # IG prior on sigma2
    #sigma2 = np.reciprocal(rng.gamma(shape = a0, scale = 1.0 / b0))
    # half normal prior on sigma2
    sigma2 = np.maximum(np.abs(rng.normal(loc = 0.0, scale = sigma2_sd, size = n_samples)),
                        1e-10)
    if theta_isotropic:
        # beta ~ N_p(Rm_0, tau^2 sigma^2(X^{\T}X)^{-1})
        theta = rng.normal(loc = beta_loc,
                          scale = tau_beta * np.sqrt(sigma2[:, np.newaxis]), 
                          size = (n_samples, p + 1))
        beta = solve_triangular(R_x, theta.T, lower = False).T
    else:
        # beta ~ N_p(m_0, tau^2 sigma^2)
        beta = rng.normal(loc = beta_loc,
                          scale = tau_beta * np.sqrt(sigma2[:, np.newaxis]), 
                          size = (n_samples, p + 1))
        theta = beta @ R_x.T
    beta_signlog = bfhelp.signed_log(beta)
    beta_arcsinh = np.arcsinh(beta)
    if beta_noise_R is None:
        beta_corrupted = beta + rng.standard_normal(size = beta.shape)
    else:
        beta_corrupted = beta + rng.standard_normal(size = beta.shape) @ beta_noise_R

    rho, proposed_samples = PC_prior(n_samples, lambda_rho, Lambda, rng)
    return dict(beta = beta,
                theta = theta,
                beta_corrupted = beta_corrupted,
                sigma2 = sigma2,
                log_sigma2 = np.log(sigma2), 
                beta_signlog = beta_signlog,
                beta_arcsinh = beta_arcsinh,
                rho = rho,
                logit_rho = logit(rho))

def generate_CAR_covariates(n_samples, n, p, A_x, rng):
    X = np.empty((n_samples, n, p + 1))
    X[:,:,0] = 1.0
    # generate covariates using conditional regressions
    for i in range(1, p + 1):
        if (i == 1):
            linear_combo = 0.0
        else:
            weights = rng.standard_normal(size = [n_samples, i - 1])    
            linear_combo = (X[:,:,1:i] @ weights[:,:, np.newaxis]).squeeze(axis = -1)
        X[:,:,i] = linear_combo + CAR_prior(n_samples, A_x, rng)
    return X
    
def BYM2_likelihood(n_samples, beta, beta_corrupted,
                    log_sigma2, logit_rho, Lambda, A_y, A_x,
                    X_fixed = None, # fixes X for all samples if input
                    corrupt_residual = True,
                    rng = None, **kwargs):
    # beta: 1D array of regression coefficients (including intercept)
    # sigma2: total error variance
    # rho: spatial proportion of variance
    # Lambda: 1D array of eigenvalues of CAR precision matrix Q
    # A_y: 2D matrix, factor of response spatial covariance Q^{-1} = A_y A^{t}_y
    # A_x: 2D matrix, factor of response spatial covariance Q^{-1} = A_x A^{t}_x
    if rng is None:
        rng = np.random.default_rng()
    sigma2 = np.exp(log_sigma2)
    rho = expit(logit_rho)
    n = Lambda.size
    p = beta.shape[-1] - 1
    if (X_fixed is None):
        # (batch_size, n, p + 1)
        X = generate_CAR_covariates(n_samples, n, p, A_x, rng)
        # (batch_size, n)
        mu = (X @ beta[:,:,np.newaxis]).squeeze(-1)
        if corrupt_residual:
            mu_corrupted = (X @ beta_corrupted[:, :, np.newaxis]).squeeze(-1)
    else:      
        # X has shape (n, p + 1)
        X = np.broadcast_to(X_fixed, (n_samples, *X_fixed.shape))
        # (batch_size, n)
        mu = beta @ X_fixed.T
        if corrupt_residual:
            mu_corrupted = beta_corrupted @ X_fixed.T
    # Scale factors — (B,) broadcast over (batch_size, n) via newaxis
    scale_s = np.sqrt(sigma2 * rho)[:, np.newaxis]          # (B, 1)
    scale_e = np.sqrt(sigma2 * (1.0 - rho))[:, np.newaxis]  # (B, 1)
    # compute latent spatial effects gamma with CAR prior
    gamma = CAR_prior(n_samples, A_y, rng) * scale_s        # (B, N)
    eta   = rng.standard_normal((n_samples, n)) * scale_e   # (B, N)
    # add in error terms and reshape to 3D array
    y = (mu + gamma + eta)[:,:,np.newaxis]                  # (B, N, 1)
    if corrupt_residual:
        r = y - mu_corrupted[:,:,np.newaxis]
    else:
        r = y - mu[:,:,np.newaxis]
    return dict(X = X, y = y, r = r, gamma = gamma)

def BYM2_simulators(Lambda, A_y, A_x, lambda_rho, p,
                    rng = np.random.default_rng(),
                    corrupt_residual = False,
                    beta_noise_R = None,
                    beta_loc = 0.0,
                    tau_beta = 10.0,
                    sigma2_sd = 10.0,
                    fix_X = False,
                    X = None,
                    R_x = None,
                    theta_isotropic = False):
    n = Lambda.size
    if fix_X and X is None:
        X = generate_CAR_covariates(1, n, p, A_x, rng).squeeze(axis=0)        
    def prior(batch_size):
        return BYM2_prior(batch_size[0], p, Lambda, lambda_rho,
                          beta_loc = beta_loc, tau_beta = tau_beta,
                          beta_noise_R = beta_noise_R,
                          sigma2_sd = sigma2_sd,
                          R_x = R_x, theta_isotropic = theta_isotropic,
                          rng = rng)

    def likelihood(batch_size, beta, beta_corrupted, log_sigma2, logit_rho):
        return BYM2_likelihood(batch_size[0], beta, beta_corrupted,
                               log_sigma2, logit_rho,
                               Lambda, A_y, A_x,
                               X_fixed = X, 
                               corrupt_residual = corrupt_residual,
                               rng = rng)
        
    return prior, likelihood, X

def compute_spatial_operators(W: np.array) -> dict:
    """
    Precompute the spatial filtering operator derived from a (fixed)
    adjacency matrix W: the symmetrically-normalized adjacency (with
    self-loops), stored as a sparse CSR matrix.
 
    This depends only on W, not on any sampled data, so it should be
    computed once per adjacency matrix rather than inside a per-sample
    function like local_spatial_statistics.
 
    A sparse representation matters a lot here: for typical spatial
    adjacency (each region touching a handful of neighbors), a dense
    (n, n) matmul costs O(n^2 * batch) while the sparse version costs
    O(nnz * batch) -- often a 100-1000x reduction for n in the thousands.
 
    Parameters
    ----------
    W : np.array
        Adjacency matrix, shape (n, n).
 
    Returns
    -------
    dict
        A_norm - symmetrically-normalized adjacency (with self-loops),
                 as a scipy.sparse.csr_matrix of shape (n, n).
    """
    W = np.asarray(W, dtype=float)
    n = W.shape[0]
 
    A_tilde = W + np.eye(n)
    d = A_tilde.sum(axis=1)
    d_inv_sqrt = np.power(d, -0.5)
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    D_inv_sqrt = np.diag(d_inv_sqrt)
    A_norm = D_inv_sqrt @ A_tilde @ D_inv_sqrt
 
    A_norm_sparse = sparse.csr_matrix(A_norm.astype(np.float32))
 
    return dict(A_norm=A_norm_sparse)
 
 
def local_spatial_statistics(batch_size,
                              y: np.array,
                              A_norm) -> dict:
    """
    Compute closed-form local (per-node) spatial summary statistics directly
    from observed data y and a precomputed spatial operator - without
    routing through any learned/MLP layers.
 
    Batched version: y is expected with a leading batch dimension, shape
    (batch, n, 1) (or (batch, n)). A_norm is a sparse (n, n) operator
    shared across the whole batch. Rather than relying on dense broadcasting
    (which would redo an O(n^2) matmul per batch), y is reshaped once to
    (n, batch) so the whole batch is filtered in a single sparse-dense
    matrix multiply.
 
    Only spatial_lag and its cheap (elementwise) derivatives are computed
    here - the Laplacian-based statistics (laplacian_filter, local_roughness,
    local_rayleigh) have been dropped since they required a second,
    equally expensive full matmul (L @ y) for comparatively little extra
    information beyond what spatial_lag/high_freq/local_moran/local_nugget
    already capture. Re-add compute_spatial_operators' L and this
    function's original Laplacian-derived terms if you need them and can
    afford the extra matmul.
 
    Note: this operates on y directly rather than the OLS residual
    (I - H) y. Since X is fixed/broadcast identically across samples in
    this simulator, residualizing per sample would be wasted work for an
    identical result; using y avoids that cost entirely. If you later use
    a simulator where X genuinely varies per sample, residualize against X
    once per unique X (or upstream, in the likelihood step) rather than
    inside this function.
 
    Parameters
    ----------
    batch_size : tuple
        Batch size passed by LambdaSimulator when is_batched=True. Unused -
        the shape of y already determines the batch dimension.
    y : np.array
        Response array, shape (batch, n, 1) or (batch, n).
    A_norm : scipy.sparse.csr_matrix
        Precomputed symmetrically-normalized adjacency matrix, shape (n, n).
 
    Returns
    -------
    dict
        Dictionary of per-node (batch, n, 1) arrays:
        spatial_lag  - normalized-adjacency-smoothed y
        high_freq    - y minus its spatial lag
        local_moran  - local Moran's I / LISA statistic
        local_nugget - squared high-frequency component
    """
    y = np.asarray(y, dtype=np.float32)
    if y.ndim == 1:
        y = y[np.newaxis, :, np.newaxis]   # (n,) -> (1, n, 1)
    elif y.ndim == 2:
        y = y[..., np.newaxis]             # (batch, n) -> (batch, n, 1)
 
    batch, n, _ = y.shape
 
    # --- One sparse-dense matmul for the whole batch at once ---
    # (n, n) sparse @ (n, batch) dense -> (n, batch), instead of looping
    # or broadcasting a dense (n, n) matrix batch_size times.
    y_2d = y.reshape(batch, n).T            # (n, batch)
    y_spatial_2d = A_norm.dot(y_2d)         # (n, batch)
    y_spatial = y_spatial_2d.T[..., np.newaxis]  # (batch, n, 1)
 
    y_highfreq = y - y_spatial
 
    # --- Closed-form local (per-node) statistics: elementwise, no matmul ---
    local_moran = y * y_spatial
    local_nugget = np.square(y_highfreq)
 
    return dict(
        spatial_lag=y_spatial,
        high_freq=y_highfreq,
        local_moran=local_moran,
        local_nugget=local_nugget,
    )
 
 
def make_local_spatial_statistics_fn(W: np.array):
    """
    Bind a fixed adjacency matrix's derived spatial operator into a
    ready-to-use batched simulator function, so W/A_norm never need to
    be produced by (or flow through) the simulator pipeline itself.
 
    Parameters
    ----------
    W : np.array
        Adjacency matrix, shape (n, n). Computed once, here, not per batch.
 
    Returns
    -------
    callable
        A function `fn(batch_size, y)` suitable for
        `bf.simulators.LambdaSimulator(fn, is_batched=True)`.
    """
    spatial_ops = compute_spatial_operators(W)  # {"A_norm": <sparse csr>}
 
    def _fn(batch_size, y):
        return local_spatial_statistics(batch_size, y, **spatial_ops)
 
    return _fn

if __name__ == "__main__":
    pass

    
