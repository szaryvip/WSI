import numpy as np
from math import exp
from neuron import Neuron

class NeuralNetwork:
    _neurons = []
    
    def __init__(self):
        # TODO SZYMON
        # tu musisz jakoś stworzyc neurony z losowymi wagami,
        # calego inita zostawiam tobie
        self._neurons.append(Neuron())

    def transfer_derivate(output): 
        # for sigmoid function --szary
        return output * (1.0 - output)

    def backward_propagate_error(self, expected):
        # szary
        pass
    
    def update_weights(self, row, l_rate):
        # szary
        pass
