#!/usr/bin/env python3
'''Module contenant la classe neurone
'''

import numpy as np


class Neuron:
    '''Classe qui définit le neurone
    '''

    def __init__(self, nx):
        '''constructeur de la classe
        la variable nx: est le nombre d'entités d'entrée du neurone
        '''

        if not isinstance(nx, int):
            raise TypeError('nx must be an integer')
        if nx < 1:
            raise ValueError('nx must be a positive integer')
        self.__W = np.random.normal(0, 1, (1, nx))
        self.__b = 0
        self.__A = 0

    @property
    def W(self):
        '''Returns the value of __W
        '''

        return self.__W

    @property
    def b(self):
        '''calcule de la propagation avant du neurone
            var X: un tableau np avec la forme (nx, m) qui contient les données d'entrées
            retourne: attribut privé __A
        '''

        return self.__b

    @property
    def A(self):
        '''fonction getter pour W
            retourne: vecteur de poids du neurone
        '''

        return self.__A

    def forward_prop(self, X):
        '''fonction getter pour b
            retourne: biais pour le neurone
        '''

        x = np.matmul(self.__W, X) + self.__b
        self.__A = 1 / (1 + np.exp(-x))
        return self.__A
