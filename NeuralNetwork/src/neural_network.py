import numpy as np
from math import exp
from neuron import Neuron
from layer import Layer
import random


class NeuralNetwork:
    _layers = []
    _hidden_layers = []
    inputs_number = None
    outputs_number = None
    _activation_type = 'sigmoid'

    def __init__(
        self,
        hidden_layers_number: int,
        neurons_in_layer_number: int,
        inputs_number: int = 28 * 28,
        outputs_number: int = 10,
        activation_type: str = 'sigmoid'
    ):
        self._inputs_number = inputs_number
        self._outputs_number = outputs_number
        self._activation_type = activation_type

        # add hidden layers
        for layer_number in range(hidden_layers_number+1):
            layer_neurons = []

            # create layer neurons
            if layer_number == 0:
                # first hidden layer
                weights_number = inputs_number
            elif layer_number < hidden_layers_number:
                # hidden layers (except the first one)
                weights_number = neurons_in_layer_number
            else:
                # output layer
                weights_number = neurons_in_layer_number
                neurons_in_layer_number = outputs_number

            for neuron in range(neurons_in_layer_number):
                weights = [
                    random.uniform(-1, 1)
                    for weight in range(weights_number)
                ]

                layer_neurons.append(Neuron(weights))

            layer = Layer(layer_neurons)

            # add layer to the network
            self._layers.append(layer)

            # # set new layer as "next_layer" of previous layer
            # if len(self._layers) > 1:
            #     self._hidden_layers[-2].set_next_layer(self._layers[-1])

        self.hidden_layers = self._layers[:-1]
        self._output_layer = self._layers[-1]

    def forward_propagate(self, inputs):
        if len(inputs) != self._inputs_number:
            raise ValueError('Invalid number of inputs')

        for layer in self._layers:
            outputs = []
            for neuron in layer.get_neurons():
                neuron.update_output(inputs)
                outputs.append(neuron.get_output())
            inputs = outputs
        return outputs

    def predict(self, inputs):
        if len(inputs) != self._inputs_number:
            raise ValueError('Invalid number of inputs')

        outputs = self.forward_propagate(inputs)
        return max(outputs)

    def transfer_derivate(self, output):
        # for sigmoid function --szary
        if self._activation_type == "sigmoid":
            return output * (1.0 - output)
        else:
            pass

    def backward_propagate_error(self, expected):
        # szary expected to wartosc ktora chcemy przewidziec
        for layer in reversed(self._layers):
            errors = []
            if layer != self._output_layer:
                for index in range(len(layer)):
                    error = 0.0
                    for neuron in layer.get_next_layer():
                        error += neuron.get_weight()[index] *\
                            neuron.get_delta()
                    errors.append(error)
            else:
                for neuron in layer:
                    errors.append(neuron.get_output() - expected)
            for index, neuron in enumerate(layer):
                delta = errors[index] * self.transfer_derivate(
                    neuron.get_output()
                )
                neuron.set_delta(delta)

    def update_weights(self, row, learning_rate):
        # szary row to matrix znormalizowany z danych trenujacych
        for index, layer in enumerate(self._layers):
            inputs = row
            if layer != self._input_layer:
                inputs = [
                    neuron.get_output() for neuron in self._layers[index-1]
                ]
            for neuron in layer:
                for index, input in enumerate(inputs.flatten()):
                    new_weight = neuron.get_weight()[index] - learning_rate *\
                        neuron.get_delta() * input
                    neuron.set_weight(new_weight, index)
                neuron.get_weight()[-1] = neuron.get_weight()[-1] -\
                    learning_rate * neuron.get_delta()
