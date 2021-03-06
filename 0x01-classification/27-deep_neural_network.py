#!/usr/bin/env python3
'''Deep Neural Network'''
import numpy as np
import matplotlib.pyplot as plt
import pickle


class DeepNeuralNetwork:
    '''Deep Neural Network définit un réseau de neurones profonds effectuant une classification binaire'''

    def __init__(self, nx, layers):
        '''Constructeur de la classe'''
        if type(nx) != int:
            raise TypeError('nx must be an integer')
        elif nx < 1:
            raise ValueError('nx must be a positive integer')
        elif (
            type(layers) != list
        ) or (
            len(layers) < 1
        ) or (
            min(layers) < 1
        ):
            raise TypeError("layers must be a list of positive integers")
        else:
            self.__L = len(layers)
            self.__cache = {}
            self.__weights = {
                "W1": np.random.randn(
                    layers[0], nx
                ) * np.sqrt(2 / nx),
                "b1": np.zeros((layers[0], 1))
            }
            for i in range(1, self.L):
                self.weights[
                    'W{}'.format(i + 1)
                ] = np.random.randn(
                        layers[i],
                        layers[i - 1]
                    ) * np.sqrt(2 / layers[i - 1])
                self.weights[
                    'b{}'.format(i + 1)
                ] = np.zeros((layers[i], 1))

    @property
    def L(self):
        '''L'''
        return self.__L

    @property
    def cache(self):
        '''cache'''
        return self.__cache

    @property
    def weights(self):
        '''weights'''
        return self.__weights

    def forward_prop(self, X):
        '''Calculates the forward'''
        self.__cache['A0'] = X
        for i in range(1, self.L + 1):
            self.__cache[
                'A' + str(i)
            ] = self.sigmoid(
                    self.__cache['A'+str(i - 1)],
                    self.weights['W'+str(i)],
                    self.weights['b'+str(i)]
                )
        self.__cache[
            'A' + str(self.L)
        ] = self.softmax(
                self.__cache['A'+str(self.L - 1)],
                self.weights['W'+str(self.L)],
                self.weights['b'+str(self.L)]
            )
        return self.cache['A' + str(self.L)], self.cache

    def sigmoid(self, X=None, w=None, b=None, x=None):
        '''Sigmoid function'''
        if x:
            return 1 / (1 + np.exp(-x))
        else:
            return 1 / (1 + np.exp(
                -np.add(
                    np.matmul(w, X), b
                )
            ))

    def softmax(self, X=None, w=None, b=None, x=None):
        ''' Softmax function '''
        if x:
            return np.exp(x) / (np.sum(np.exp(x), axis=0))
        else:
            return np.exp(
                np.add(
                    np.matmul(w, X), b)
            ) / (
                np.sum(
                    np.exp(np.add(np.matmul(w, X), b)),
                    axis=0
                )
            )

    def cost(self, Y, A):
        '''Calcule le coût (cost) du modèle à l'aide de la régression logistique'''
        m = A.shape[1]
        return np.sum(
            -Y * np.log(A)
        ) / m

    def one_hot_encode(Y, classes):
        '''converts a numeric label
        vector into a one-hot matrix'''
        if (
            Y is not None
        ) and (
            type(Y) is np.ndarray
        ) and (
            type(classes) is int
        ):
            try:
                oneMatrix = np.zeros((classes, Y.shape[0]))
                oneMatrix[Y, np.arange(Y.shape[0])] = 1
                return oneMatrix
            except Exception:
                return None
        else:
            return None

    def one_hot_decode(one_hot):
        '''converts a one-hot matrix
        into a vector of labels'''
        if (
            type(one_hot) is np.ndarray
        ) and (
            len(one_hot.shape) is 2
        ):
            try:
                return np.argmax(one_hot, axis=0)
            except Exception:
                return None
        else:
            return None

    def evaluate(self, X, Y):
        '''Évalue les prédictions du neurone'''
        A = self.forward_prop(X)[0]
        cost = self.cost(Y, A)
        oneDecode = self.one_hot_decode(A)
        A = self.one_hot_encode(oneDecode, Y.shape[0])
        return (A.astype(int), cost)

    def gradient_descent(self, Y, cache, alpha=0.05):
        '''Calcule une passe de descente de gradient sur le réseau neuronal profond'''

        weight_copy = self.weights.copy()
        dzi = np.subtract(
            self.cache['A' + str(self.L)], Y
        )
        for i in reversed(range(1, self.L + 1)):
            b = weight_copy['b' + str(i)]
            if i == self.L:
                np.subtract(
                    self.cache['A' + str(self.L)], Y
                )
            else:
                w = weight_copy['W' + str(i + 1)]
                dzi = np.multiply(
                    (
                        self.cache[
                            'A' + str(i)
                        ] * (
                            1 - self.cache['A' + str(i)]
                        )
                    ), np.matmul(w.T, dzi)
                )
            self.__weights['b' + str(i)] = weight_copy[
                'b' + str(i)
            ] - alpha * np.mean(
                    dzi, axis=1, keepdims=True
            )
            self.__weights['W' + str(i)] = weight_copy[
                'W' + str(i)
            ] - alpha * np.matmul(
                    dzi, self.cache['A' + str(i - 1)].T
            ) / Y.shape[1]

    def train(
        self, X, Y,
        iterations=5000,
        alpha=0.05,
        verbose=True,
        graph=True,
        step=100
    ):
        '''Entraîne le neurone'''
        if type(iterations) != int:
            raise TypeError('iterations must be an integer')
        if iterations < 0:
            raise ValueError('iterations must be a positive integer')
        if type(alpha) != float:
            raise TypeError('alpha must be a float')
        if alpha < 0:
            raise ValueError('alpha must be positive')
        if verbose or graph:
            if type(step) != int:
                raise TypeError('step must be an integer')
            if (step < 0) or (step > iterations):
                raise ValueError('step must be positive and <= iterations')
        allCost, stepper = [], 0
        for i in range(iterations):
            self.gradient_descent(Y, self.forward_prop(X)[1], alpha)
            allCost.append(self.cost(Y, self.cache['A' + str(self.L)]))
            if verbose and (i - 1 == stepper - 1):
                print(
                    'Cost after {} iterations: {}'.format(
                        i, allCost[i]
                    )
                )
                stepper += step
        evaluation, cost = self.evaluate(X, Y)
        i += 1
        print("Cost after {} iterations: {}".format(i, cost))
        if graph:
            plt.plot(allCost)
            plt.xlabel('iteration')
            plt.ylabel('cost')
            plt.title('Training Cost')
            plt.show()
        return (evaluation, cost)

    def save(self, filename):
        '''Enregistre l'objet d'instance dans un fichier au format pickle'''
        if type(filename) is str:
            if filename[-4:] != '.pkl':
                filename += '.pkl'
            with open(filename, 'wb') as file:
                pickle.dump(self, file)
        else:
            return None

    @staticmethod
    def load(filename):
        '''Loads a pickled
        DeepNeuralNetwork object'''
        try:
            with open(filename, 'rb') as file:
                return pickle.load(file)
        except Exception:
            return None