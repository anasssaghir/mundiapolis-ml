#!/usr/bin/env python3
import tensorflow.keras as K
"""
Saves and load Model for Keras
"""
def save_model(network, filename):
    K.models.save_model(model=network,filepath=filename)
    return None


def load_model(filename):
    return K.models.load_model(filepath=filename)
