#!/usr/bin/env python3
'''Module contenant la classe neurone
'''

import numpy as np


class Neuron:
    '''Classe qui définit le neurone
    '''

    def __init__(self, nx):
        '''Initialization function for the Neuron class

        Arguments.
            nx: Le nombre d'entrée dans le neurone.
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
        '''Renvoie la valeur de __W
        '''

        return self.__W

    @property
    def b(self):
        '''Renvoie la valeur de __b
        '''

        return self.__b

    @property
    def A(self):
        '''Renvoie la valeur de __A
        '''

        return self.__A