#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  2 20:21:03 2026

@author: kylel
"""

import os
import jax
os.environ["KERAS_BACKEND"] = "jax"
jax.config.update("jax_enable_x64", False)
import keras
import bayesflow as bf
from keras import layers, ops
import numpy as np

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
class SummaryIdentity(bf.networks.SummaryNetwork):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.flatten_layer = layers.Flatten()
    
    def call(self, x, **kwargs):
        return self.flatten_layer(x)
    
    def get_config(self):
        config = super().get_config()
        return config
    
    @classmethod
    def from_config(cls, config):
        return cls(**config)


@keras.saving.register_keras_serializable(package="BayesflowSpatialSummary")
class SummaryGNN(bf.networks.SummaryNetwork):
    def __init__(self, adjacency_matrix,
                 gnn_dim=64,
                 compress_dim=64,
                 hidden_dim=32,
                 summary_dim=16, **kwargs):
        super().__init__(**kwargs)
        
        # save input arguments for recovery in config
        self.adjacency_matrix = np.asarray(adjacency_matrix)
        self.compress_dim = compress_dim
        self.gnn_dim = gnn_dim
        self.hidden_dim = hidden_dim
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
        self.fc_1 = layers.Dense(gnn_dim)
        self.fc_2 = layers.Dense(gnn_dim)
        self.compress = layers.Dense(compress_dim)
        self.smooth_compress = layers.Dense(compress_dim)
        #self.resid_compress = layers.Dense(compress_dim)
        #self.rough_compress = layers.Dense(compress_dim)
        self.fc_hidden = layers.Dense(hidden_dim)
        self.fc_out = layers.Dense(summary_dim)

    def call(self, x, **kwargs):
        # Use keras.activations and keras.ops for JAX compatibility
        # Step 1: Feature Extraction
        h = keras.activations.swish(self.fc_1(x))
        h = keras.activations.swish(self.fc_2(h))
        
        # Step 2: Spatial Filtering (Using ops.matmul)
        h_spatial = ops.matmul(self.A_norm, h)
        h_rough = ops.matmul(self.L_tensor, h)
        h_residual = h - h_spatial
        
        # Step 3: Branch Compression
        batch_size = ops.shape(x)[0]
        
        # Reshape + Compress
        f = keras.activations.swish(
            self.compress(ops.reshape(h, (batch_size, -1)))
        )
        f_smooth = keras.activations.swish(
            self.smooth_compress(ops.reshape(h_spatial, (batch_size, -1)))
        )
        """
        f_resid = keras.activations.swish(
            self.resid_compress(ops.reshape(h_residual, (batch_size, -1)))
        )
        f_rough = keras.activations.swish(
            self.rough_compress(ops.reshape(h_rough, (batch_size, -1)))
        )
        """
        
        # Step 4: Global Stats (Backend-agnostic)
        #y_scale_log = ops.log(ops.std(x[:, :, -1:], axis=1))
        sigma_nugget = ops.std(h_residual, axis=1)
        tau_spatial = ops.std(h_rough, axis=1)
        
        # Log-ratio
        spatial_logratio = ops.log(tau_spatial + 1e-8) - \
            ops.log(sigma_nugget + 1e-8)
        
        # Step 5: Additional Spatial Statistics

        # 5a. The Rayleigh Quotient (Graph Frequency)
        # Formula: (h^T L h) / (h^T h)
        # numerator: h * (L @ h) -> then sum over nodes
        # denominator: h * h -> then sum over nodes
        # Shape: (batch, 64)
        rayleigh_stat = ops.sum(h * h_rough, axis=1) \
            / (ops.sum(ops.square(h), axis=1) + 1e-8)
        
        # Step 5: Concatenate and Project
        summary = ops.concatenate([
            f, f_smooth, #f_resid, f_rough, 
            rayleigh_stat, sigma_nugget, tau_spatial, spatial_logratio,
            #y_scale_log, geary_stat, moran_stat
        ], axis=-1)
        
        summary_hidden = keras.activations.swish(self.fc_hidden(summary))
        
        return self.fc_out(summary_hidden)
    
    def get_config(self):
        config = super().get_config()
        config.update({
            "adjacency_matrix": self.adjacency_matrix.tolist(),
            "compress_dim": self.compress_dim,
            "gnn_dim": self.gnn_dim,
            "hidden_dim": self.hidden_dim,
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
    
