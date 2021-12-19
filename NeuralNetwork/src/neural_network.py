import numpy as np
from math import exp
from neuron import Neuron
from layer import Layer
import random


class NeuralNetwork:
    _hidden_layers = []
    _output_layer = None
    _input_layer = None
    _layers = []
    _activation_type = 'sigmoida'

    def __init__(
        self,
        hidden_layers_number: int,
        neurons_in_layer_number: int
    ):
        # TODO SZYMON
        # tu musisz jakoś stworzyc neurony z losowymi wagami,
        # i dać je do layerów chyba
        # calego inita zostawiam tobie

        for _ in range(hidden_layers_number):
            layer_neurons = []

            # create layer neurons
            for _ in range(neurons_in_layer_number):
                weights = [
                    random.uniform(-1, 1)
                    for _ in range(neurons_in_layer_number)
                ]

                layer_neurons.append(Neuron(weights))

            # add layers to the network
            self._hidden_layers.append(
                Layer(layer_neurons)
            )

            # set new layer as "next_layer" of previous layer
            self._layers[-2].set_next_layer(self._layers[-1])

        return self

    def transfer_derivate(self, output):
        # for sigmoid function --szary
        if self._activation_type == "sigmoida":
            return output * (1.0 - output)

    def backward_propagate_error(self, expected):
        # szary expected to wartosc ktora chcemy przewidziec
        for layer in reversed(self._layers):
            errors = []
            if layer != self._output_layer:
                for index in range(len(layer)):
                    error = 0.0
                    for neuron in layer.get_next_layer():
                        error += neuron.get_weight()[index] * neuron.get_delta()
                    errors.append(error)
            else:
                for neuron in layer:
                    errors.append(neuron.get_output() - expected)
            for index, neuron in enumerate(layer):
                delta = errors[index] * self.transfer_derivate(neuron.get_output())
                neuron.set_delta(delta)

    def update_weights(self, row, learning_rate):
        # szary row to matrix znormalizowany z danych trenujacych
        for index, layer in enumerate(self._layers):
            inputs = row
            if layer != self._input_layer:
                inputs = [neuron.get_output() for neuron in self._layers[index-1]]
            for neuron in layer:
                for index, input in enumerate(inputs.flatten()):
                    new_weight = neuron.get_weight()[index] - learning_rate * neuron.get_delta() * input
                    neuron.set_weight(new_weight, index)
                neuron.get_weight()[-1] = neuron.get_weight()[-1] - learning_rate * neuron.get_delta()
