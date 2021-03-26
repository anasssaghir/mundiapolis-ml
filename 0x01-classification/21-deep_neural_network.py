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

    def forward_prop(self, X):
        """
        calcule la propagation directe du réseau neuronal profond
         Le paramètre X: tableau np avec les données d'entrée de forme (nx, m)
         elle retourne : la sortie du réseau neuronal profond et du cache,
         où le cache est la sortie activée de chaque couche
        """
        # Input layer
        self.__cache['A0'] = X
        # Hidden and output layer
        for i in range(self.__L):
            # create keys to access weights(W), biases(b) and store in cache
            key_w = 'W' + str(i + 1)
            key_b = 'b' + str(i + 1)
            key_cache = 'A' + str(i + 1)
            key_cache_last = 'A' + str(i)
            # store activation in cache
            output_Z = np.matmul(self.__weights[key_w], self.__cache[
                key_cache_last]) + self.__weights[key_b]
            output_A = 1 / (1 + np.exp(-output_Z))
            self.__cache[key_cache] = output_A
        return output_A, self.__cache

    def cost(self, Y, A):
        """
        calcule le coût du modèle à l'aide de la régression logistique
         paramètre Y: un tableau np avec des étiquettes de forme correctes (1, m)
         paramètre A: un tableau np avec la sortie activée de shape (1, m)
         elle retourne : le coût
        """
        cost = Y * np.log(A) + (1 - Y) * np.log(1.0000001 - A)
        cost = np.sum(cost)
        cost = - cost / A.shape[1]
        return cost

    def evaluate(self, X, Y):
        """
        Fonction évalue la prédiction du réseau neuronal profond
         paramètre X: tableau np avec des données d'entrée de forme (nx, m)
         paramètre Y: tableau np avec étiquette de forme correcte (1, m)
         elle retourne : prédiction des neurones et coût du réseau
        """
        A, _ = self.forward_prop(X)
        prediction = np.where(A >= 0.5, 1, 0)
        cost = self.cost(Y, A)
        return prediction, cost

    def gradient_descent(self, Y, cache, alpha=0.05):
        """
        calcule une passe de descente de gradient sur le neurone
         Le paramètre Y: tableau np avec des étiquettes correctes de forme (1, m)
         Le paramètre cache: dictionnaire contenant toutes les valeurs intermédiaires du
         réseau
         Le paramètre alpha: le taux d'apprentissage
         elle ne retourne rien 
        """
        m = Y.shape[1]
        for i in reversed(range(self.__L)):
            # create keys to access weights(W), biases(b) and store in cache
            key_w = 'W' + str(i + 1)
            key_b = 'b' + str(i + 1)
            key_cache = 'A' + str(i + 1)
            key_cache_dw = 'A' + str(i)
            # Activation
            A = cache[key_cache]
            A_dw = cache[key_cache_dw]
            if i == self.__L - 1:
                dz = A - Y
                W = self.__weights[key_w]
            else:
                da = A * (1 - A)
                dz = np.matmul(W.T, dz)
                dz = dz * da
                W = self.__weights[key_w]
            dw = np.matmul(A_dw, dz.T) / m
            db = np.sum(dz, axis=1, keepdims=True) / m
            self.__weights[key_w] = self.__weights[key_w] - alpha * dw.T
            self.__weights[key_b] = self.__weights[key_b] - alpha * db

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
