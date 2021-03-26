#!/usr/bin/env python3
"""
Module contenant la classe neurone
"""
import numpy as np


class NeuralNetwork:
    """
    Une classe qui définit un réseau de neurones avec une couche cachée effectuant une classification binaire
    """

    def __init__(self, nx, nodes):
        """
        constructeur de la classe
        la variable nx: est le nombre d'entités d'entrée du neurone
        """
        if type(nx) is not int:
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")
        self.nx = nx
        if type(nodes) is not int:
            raise TypeError("nodes must be an integer")
        if nodes < 1:
            raise ValueError("nodes must be a positive integer")
        # hidden layer
        self.W1 = np.random.randn(nodes, nx)
        self.b1 = np.zeros((nodes, 1))
        self.A1 = 0
        # output neuron
        self.W2 = np.random.randn(1, nodes)
        self.b2 = 0
        self.A2 = 0
