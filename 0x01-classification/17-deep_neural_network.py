#!/usr/bin/env python3
"""
Module contenant la classe neurone
"""
import numpy as np


class DeepNeuralNetwork:
    """
    Une classe qui définit un réseau neuronal profond avec une couche cachée effectuant une classification binaire
    """

    def __init__(self, nx, layers):
        """
        constructeur de classe
         paramètre nx: est le nombre d'entités d'entrée du neurone
         paramètre layers: une liste représentant le nombre de nœuds dans chaque couche
        """
        if type(nx) is not int:
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")
        self.nx = nx
        if type(layers) is not list or len(layers) == 0:
            raise TypeError("layers must be a list of positive integers")
        # L : est le nombre de couches dans le réseau neuronal
        self.__L = len(layers)
        # cache : est un dictionnaire pour contenir toutes les valeurs intermédiaires du réseau
        self.__cache = {}
        # weights : est un dictionnaire pour tenir tous les poids et biais du réseau
        weights = {}
        for i in range(len(layers)):
            if layers[i] < 1:
                raise TypeError("layers must be a list of positive integers")
            key_w = 'W' + str(i + 1)
            key_b = 'b' + str(i + 1)
            if i == 0:
                weights[key_w] = np.random.randn(layers[i], nx)*np.sqrt(2 / nx)
            else:
                weights[key_w] = np.random.randn(layers[i], layers[
                    i-1]) * np.sqrt(2 / layers[i-1])
            weights[key_b] = np.zeros((layers[i], 1))
        self.__weights = weights

    @property
    def cache(self):
        """
        fonction getter pour le cache
        elle retourne : cache
        """
        return self.__cache

    @property
    def L(self):
        """
        fonction getter pour L (nombre de couches)
         elle retourne : L
        """
        return self.__L

    @property
    def weights(self):
        """
        fonction getter pour weights
        elle retourne : weights
        """
        return self.__weights