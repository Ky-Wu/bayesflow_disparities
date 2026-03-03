#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  2 20:21:03 2026

@author: kylel
"""

import keras
import bayesflow as bf
from keras import layers
  
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
    
