#!/usr/bin/env python3
"""
Module contenant la classe neurone
"""
import numpy as np


class Neuron:
    """
    la classe qui définit un neurone
    """

    def __init__(self, nx):
        """
        constructeur de la classe
        la variable nx: est le nombre d'entités d'entrée du neurone
        """
        if type(nx) is not int:
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")
        self.nx = nx
        self.__W = np.random.randn(1, nx)
        self.__b = 0
        self.__A = 0

    def forward_prop(self, X):
        """
        calcule de la propagation avant du neurone
        elle retourne: attribut privé __A
        """
        preactivation = np.matmul(self.__W, X) + self.__b
        self.__A = 1 / (1 + np.exp(-preactivation))
        return self.__A

    def cost(self, Y, A):
        """
        le calcule du coût du modèle à l'aide de la régression logistique
         elle retourne: le coût
        """
        cost = Y * np.log(A) + (1 - Y) * np.log(1.0000001 - A)
        cost = np.sum(cost)
        cost = - cost / A.shape[1]
        return cost

    def evaluate(self, X, Y):
        """
        elle évalue la prédiction des neurones
         paramètre X: un tableau np avec données d'entrée et forme (nx, m)
         paramètre Y: un tableau np avec étiquette et forme correctes (1, m)
         elle retourne: prédiction des neurones et coût du réseau
        """
        self.forward_prop(X)
        prediction = np.where(self.__A >= 0.5, 1, 0)
        cost = self.cost(Y, self.__A)
        return prediction, cost

    @property
    def W(self):
        """
        fonction getter pour W
        elle retourne: vecteur de poids du neurone
        """
        return self.__W

    @property
    def b(self):
        """
        fonction getter pour b
        elle retourne: biais pour le neurone
        """
        return self.__b

    @property
    def A(self):
        """
        fonction getter pour A
        elle retourne: sortie activée du neurone
        """
        return self.__A
