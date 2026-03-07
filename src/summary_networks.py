#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  2 20:21:03 2026

@author: kylel
"""

import os
import jax
from jax.experimental import sparse
import jax.numpy as jnp
os.environ["KERAS_BACKEND"] = "jax"
jax.config.update("jax_enable_x64", False)
import keras
import bayesflow as bf
from keras import layers, ops
import numpy as np
import tensorflow as tf

class FlatteningNet(bf.networks.SummaryNetwork):
    def __init__(self, input_shape, **kwargs):
        """
        input_shape: tensor of shape (n_regions, n_features_total)
        """
        super().__init__(**kwargs)
        
        # Define the expected input shape (e.g., (total_dim,))
        self.input_layer = layers.Input(shape=input_shape)
        # This layer collapses (n_regions, n_features_total) -> n_regions * n_features_total
        self.flatten_layer = layers.Flatten()
        self.bn_input = layers.BatchNormalization()
        # Batch normalization
        
        # Compress down to 256
        self.bottleneck = layers.Dense(256)
        self.bn_bottleneck = layers.BatchNormalization()
        self.activation = layers.Activation('leaky_relu')
        
        # Build an internal model to handle shape inference for BayesFlow
        # Even if input is already 1D, Flatten ensures consistency.
        x = self.flatten_layer(self.input_layer)
        x = self.bn_input(x)
        x = self.bottleneck(x)
        x = self.bn_bottleneck(x)
        output = self.activation(x)
        self.internal_model = keras.Model(inputs=self.input_layer, outputs=output)

    def call(self, inputs):
        """
        Receives the pre-concatenated array from the adapter.
        """
        return self.internal_model(inputs)

    @property
    def output_shape(self):
        # Informs the InferenceNetwork of the condition's dimensionality
        return self.internal_model.output_shape


@keras.saving.register_keras_serializable(package="BayesflowSpatialSummary")
class SummaryGNN(bf.networks.SummaryNetwork):
    def __init__(self, adjacency_matrix, n_features, compress_dim=64, 
                 gnn_dim=64, summary_dim=16, **kwargs):
        super().__init__(**kwargs)
        
        # save input arguments for recovery in config
        self.adjacency_matrix = np.asarray(adjacency_matrix)
        self.n_features = n_features
        self.compress_dim = compress_dim
        self.gnn_dim = gnn_dim
        self.summary_dim = summary_dim
        
        # --- 1. Precompute Constants ---
        A_tilde = adjacency_matrix + np.eye(adjacency_matrix.shape[0])
        d = np.array(A_tilde.sum(1))
        d_inv_sqrt = np.power(d, -0.5).flatten()
        d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
        D_inv_sqrt = np.diag(d_inv_sqrt)
        
        # Use keras.ops for backend-agnostic constants
        self.A_norm = ops.cast(D_inv_sqrt @ A_tilde @ D_inv_sqrt, dtype="float32")
        L = np.diag(np.array(adjacency_matrix.sum(1))) - adjacency_matrix
        self.L_tensor = ops.cast(L, dtype="float32")
        
        # --- 2. Layers ---
        self.gcn_1 = layers.Dense(gnn_dim)
        self.gcn_2 = layers.Dense(gnn_dim)
        self.smooth_compress = layers.Dense(compress_dim)
        self.resid_compress = layers.Dense(compress_dim)
        self.rough_compress = layers.Dense(compress_dim)
        self.fc_out = layers.Dense(summary_dim)

    def call(self, x, **kwargs):
        # Use keras.activations and keras.ops for JAX compatibility
        # Step 1: Feature Extraction
        h = keras.activations.swish(self.gcn_1(x))
        h = keras.activations.swish(self.gcn_2(h))
        
        # Step 2: Spatial Filtering (Using ops.matmul)
        h_spatial = ops.matmul(self.A_norm, h)
        h_rough = ops.matmul(self.L_tensor, h)
        h_residual = h - h_spatial
        
        # Step 3: Branch Compression
        batch_size = ops.shape(x)[0]
        
        # Reshape + Compress
        f_smooth = keras.activations.swish(
            self.smooth_compress(ops.reshape(h_spatial, (batch_size, -1)))
        )
        f_resid = keras.activations.swish(
            self.resid_compress(ops.reshape(h_residual, (batch_size, -1)))
        )
        f_rough = keras.activations.swish(
            self.rough_compress(ops.reshape(h_rough, (batch_size, -1)))
        )
        
        # Step 4: Global Stats (Backend-agnostic)
        y_scale = ops.std(x[:, :, -1:], axis=1)
        sigma_nugget = ops.std(h_residual, axis=1)
        tau_spatial = ops.std(h_rough, axis=1)
        
        # Log-ratio
        spatial_logratio = ops.log(tau_spatial + 1e-8) - \
            ops.log(sigma_nugget + 1e-8)
        
        # Step 5: Additional Spatial Statistics
        
        # 5a. Geary-like contrast (Local Contrast / Squared Differences)
        # Measures local 'jitter'. High values = Low spatial proportion.
        geary_stat = ops.mean(ops.square(h - h_spatial), axis=1)

        # 5b. Moran-like correlation (Global Smoothness)
        # Centering h first makes this a proper correlation measure
        moran_stat = ops.mean(h * h_spatial, axis = 1)

        # 5c. The Rayleigh Quotient (Graph Frequency)
        # Formula: (h^T L h) / (h^T h)
        # numerator: h * (L @ h) -> then sum over nodes
        # denominator: h * h -> then sum over nodes
        # Shape: (batch, 64)
        rayleigh_stat = ops.sum(h * h_rough, axis=1) \
            / (ops.sum(ops.square(h), axis=1) + 1e-8)
        
        # Step 5: Concatenate and Project
        summary = ops.concatenate([
            y_scale, f_smooth, f_resid, f_rough, 
            sigma_nugget, tau_spatial, spatial_logratio,
            geary_stat, moran_stat, rayleigh_stat
        ], axis=-1)
        
        return self.fc_out(summary)
    
    def get_config(self):
        config = super().get_config()
        config.update({
            "adjacency_matrix": self.adjacency_matrix.tolist(),
            "n_features": self.n_features,
            "compress_dim": self.compress_dim,
            "gnn_dim": self.gnn_dim,
            "summary_dim": self.summary_dim
        })
        return config
    
    @classmethod
    def from_config(cls, config):
        """
        Explicitly convert the serialized list back to a numpy array 
        before passing it to the constructor.
        """
        if "adjacency_matrix" in config:
            config["adjacency_matrix"] = np.array(config["adjacency_matrix"])
        return cls(**config)
    
    
""" 
class PoolSummaryGNN(bf.networks.SummaryNetwork):
    def __init__(self, adjacency_matrix, n_features, gnn_dim=128, 
                 hidden_dim = 256, summary_dim=64, **kwargs):
        super().__init__(**kwargs)
        
        # --- 1. Graph Setup (Stays the same, but cast to float32 early) ---
        # Pre-compute A_norm and L_tensor as before...
        A_tilde = adjacency_matrix + np.eye(adjacency_matrix.shape[0])
        d = np.array(A_tilde.sum(1))
        d_inv_sqrt = np.power(d, -0.5).flatten()
        d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
        D_inv_sqrt = np.diag(d_inv_sqrt)
        
        # Use keras.ops for backend-agnostic constants
        def to_jax_sparse(dense_matrix):
            # Convert a numpy/dense matrix to JAX BCOO format
            return sparse.bcoo_fromdense(jnp.array(dense_matrix, dtype=jnp.float32))
        
        self.A_norm = to_jax_sparse(D_inv_sqrt @ A_tilde @ D_inv_sqrt)
        #L = np.diag(np.array(adjacency_matrix.sum(1))) - adjacency_matrix
        #self.L_tensor = to_jax_sparse(L)

        # --- 2. GNN for larger Receptive Field ---
        self.gcn_1 = layers.Dense(gnn_dim)
        self.gcn_2 = layers.Dense(gnn_dim)
        
        # Final bottlenecks
        self.fc_hidden = layers.Dense(hidden_dim)
        self.fc_out = layers.Dense(summary_dim)

    def call(self, x, **kwargs):
        
        def sp_matmul(A, B):
            # vmap over the batch dimension (axis 0 of B) to safely apply the sparse matmul.
            single_matmul = sparse.sparsify(jnp.matmul)
            return jax.vmap(single_matmul, in_axes=(None, 0))(A, B)

        # Step 1: Deep GCN Layers
        # Step-wise message passing
        h1 = ops.swish(self.gcn_1(sp_matmul(self.A_norm, x))) 
        h2 = ops.swish(self.gcn_2(sp_matmul(self.A_norm, h1))) + h1
        
        # Step 2: Spatial Filtering 
        h_spatial = sp_matmul(self.A_norm, h2)
        #h_rough = sp_matmul(self.L_tensor, h2)
        h_residual = h2 - h_spatial
        
        # --- STEP 3: GLOBAL POOLING ---
        # We take the mean and max across ALL nodes (axis 1)
        # This reduces (batch, n, dim) -> (batch, dim)
        def pool(tensor):
            m = ops.mean(tensor, axis=1)
            v = ops.std(tensor, axis=1)
            return ops.concatenate([m, v], axis=-1)

        pooled_smooth = pool(h_spatial) # The 'blobs'
        pooled_nugget = pool(h_residual)  # The 'white noise'
        #pooled_rough  = pool(h_rough)   # The 'jagged edges'
        
        # Step 4: Global Stats (Backend-agnostic)
        y_scale = ops.std(x[:, :, -1:], axis=1)
        
        # Step 5: High-order Stats (Moran/Rayleigh) stay as (batch, gnn_dim)
        # 5a. Geary-like contrast (Local Contrast / Squared Differences)
        # Measures local 'jitter'. High values = Low spatial proportion.
        geary_stat = ops.mean(ops.square(h2 - h_spatial), axis=1)

        # 5b. Moran-like correlation (Global Smoothness)
        # Centering h first makes this a proper correlation measure
        h2_mean = ops.mean(h2, axis=1, keepdims=True)
        h2_centered = h2 - h2_mean
        h_spatial_centered = h_spatial - ops.mean(h_spatial, axis = 1, keepdims = True)
        moran_stat = ops.mean(h2_centered * h_spatial_centered, axis = 1)

        # 5c. The Rayleigh Quotient (Graph Frequency)
        # Formula: (h^T L h) / (h^T h)
        # numerator: h * (L @ h) -> then sum over nodes
        # denominator: h * h -> then sum over nodes
        # Shape: (batch, 64)
        lograyleigh_stat = ops.log(ops.mean(h2 * h_residual, axis=1) + 1e-10) \
            - ops.log(ops.mean(ops.square(h2), axis=1) + 1e-10)
            

            
        # Step 6: Final Summary Vector
        summary = ops.concatenate([
            y_scale,                # (batch, 1)
            pooled_smooth,          # (batch, gnn_dim * 2)
            pooled_nugget,          # (batch, gnn_dim * 2)
        #    pooled_rough,           # (batch, gnn_dim * 2)
            geary_stat,             # (batch, gnn_dim)
            moran_stat,             # (batch, gnn_dim)
            lograyleigh_stat        # (batch, gnn_dim)
        ], axis=-1)
        
        # Step 7. Final Interpretation Block (Deep enough for unnormalized data)
        summary_hidden = ops.swish(self.fc_hidden(summary))
        return self.fc_out(summary_hidden)
    
"""