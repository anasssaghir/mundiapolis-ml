#!/usr/bin/env python3
"""
Module pour créer un neurone
"""
import numpy as np


class Neuron:
    """
    Une classe qui définit un seul neurone
    """

    def __init__(self, nx):
        """
        constructeur de classe
            Le paramètre nx: est le nombre d'entités d'entrée du neurone
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
        calcule la propagation avant du neurone
            Le paramètre X: un tableau np avec les données d'entrée de forme (nx, m)
        elle retourne: attribut privé __A
        """
        preactivation = np.matmul(self.__W, X) + self.__b
        self.__A = 1 / (1 + np.exp(-preactivation))
        return self.__A

    def cost(self, Y, A):
        """
        le calcule du coût du modèle à l'aide de la régression logistique
         Le paramètre Y: un tableau np avec des étiquettes de forme correctes (1, m)
         Le paramètre A: un tableau np avec la sortie activée de shape (1, m)
         elle retourne : le coût
        """
        cost = Y * np.log(A) + (1 - Y) * np.log(1.0000001 - A)
        cost = np.sum(cost)
        cost = - cost / A.shape[1]
        return cost

    def evaluate(self, X, Y):
        """
        elle évalue la prédiction des neurones
         Le paramètre X: tableau np avec des données d'entrée de forme (nx, m)
         Le paramètre Y: tableau np avec étiquette de forme correcte (1, m)
         elle retourne : prédiction des neurones et coût du réseau
        """
        self.forward_prop(X)
        prediction = np.where(self.__A >= 0.5, 1, 0)
        cost = self.cost(Y, self.__A)
        return prediction, cost

    def gradient_descent(self, X, Y, A, alpha=0.05):
        """
        calcule une passe de descente de gradient sur le neurone
         Le paramètre X: tableau np avec des données d'entrée de forme (nx, m)
         Le paramètre Y: tableau np avec des étiquettes correctes de forme (1, m)
         Le paramètre A: tableau np avec sortie activée de la forme (1, m)
         Le paramètre alpha: le taux d'apprentissage
         elle ne retourne rien
        """
        dz = A - Y
        dw = np.matmul(X, dz.T) / A.shape[1]
        db = np.sum(dz) / A.shape[1]
        self.__W = self.__W - alpha * dw.T
        self.__b = self.__b - alpha * db

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
