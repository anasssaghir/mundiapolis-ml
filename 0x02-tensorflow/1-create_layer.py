#!/usr/bin/env python3
"""
Module to create a layer
"""
import tensorflow as tf

def create_layer(prev, n, activation):
    """
    a function that create layers
    :parameter prev: the tensor output of the previous layer
    :parameter n: the number of nodes in the layer to create
    :parameter activation: is the activation function that the layer should use
    :return: the tensor output of the layer
    """
    init = tf.contrib.layers.variance_scaling_initializer(mode="FAN_AVG")
    layer = tf.layers.Dense(n, activation=activation, kernel_initializer=init,name="layer")
    return layer(prev)
