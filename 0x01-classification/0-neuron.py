#!/usr/bin/env python3
'''la classe des neurones'''

import numpy as np


class Neuron:
    '''Classe qui a initialisé le neurone'''

    def __init__(self, nx):
        '''la fonction de la classe Neuron'''

        if not isinstance(nx, int):
            raise TypeError('nx must be an integer')
        if nx < 1:
            raise ValueError('nx must be a positive integer')
        self.W = np.random.normal(0, 1, (1, nx))
        self.b = 0
        self.A = 0
