import numpy as np
from neuron import Neuron
from layer import Layer
from math import exp
import random
from typing import List
import time
import os


class NeuralNetwork:
    _layers = []
    _hidden_layers = []
    _output_layer = []
    _epochs_number = None
    inputs_number = None
    outputs_number = None
    _activation_type = 'sigmoid'

    def __init__(
        self,
        hidden_layers_number: int,
        neurons_in_layer_number: int,
        epochs_number: int = 1,
        inputs_number: int = 28 * 28,
        outputs_number: int = 10,
        activation_type: str = 'sigmoid'
    ):
        """Initializes NeuralNetwork object

        Args:
            hidden_layers_number (int): number of hidden layers
            neurons_in_layer_number (int): number of neuron in each hidden
            layer epochs_number (int, optional):
            How many iteractionwill be done. Defaults to 1.
            inputs_number (int, optional): Size of input. Defaults to 28*28.
            outputs_number (int, optional): Size of output. Defaults to 10.
            activation_type (str, optional): What function is
                        used for activation (sigmoid or relu).
                        Defaults to 'sigmoid'.
        """
        self._inputs_number = inputs_number
        self._outputs_number = outputs_number
        self._activation_type = activation_type
        self._epochs_number = epochs_number
        self._start_time = time.time()

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

            # set new layer as "next_layer" of previous layer
            if len(self._layers) > 1:
                self._layers[-2].set_next_layer(self._layers[-1])

        self._hidden_layers = self._layers[:-1]
        self._output_layer = self._layers[-1]

    def estimate_time(self, epochs_done: int, clear: bool = False):
        if epochs_done == 0:
            time_left = '?'
        else:
            epochs_left = self._epochs_number - epochs_done
            elapsed_time = time.time() - self._start_time
            duration_of_epoch = elapsed_time / (epochs_done)
            time_left = str(round(epochs_left * duration_of_epoch))
            time_left += 's'

        if clear:
            os.system('clear')

        print(
            f'Epoch: {epochs_done + 1}/{self._epochs_number}'
            f' | Time left: {time_left}'
        )

    def forward_propagate(self, inputs: np.ndarray):
        """Propagating input signal and generate outputs
        through each layer

        Args:
            inputs (List[float]): data

        Raises:
            ValueError: raise when number of inputs is
                        not correct

        Returns:
            List[float]: output of output layer
        """
        if len(inputs) != self._inputs_number:
            raise ValueError('Invalid number of inputs')

        for layer in self._layers:
            outputs = []
            for neuron in layer.get_neurons():
                neuron.update_output(inputs)
                outputs.append(neuron.get_output())
            inputs = outputs
        return outputs

    def softmax(self, outputs: np.ndarray):
        """Choices output by softmax algorithm

        Args:
            outputs (List[float]): outputs list to choose from

        Returns:
            float: choosen value from outputs
        """
        sum_of_exp = sum([exp(output) for output in outputs])
        probability = [exp(output)/sum_of_exp for output in outputs]
        return np.random.choice(outputs, p=probability)

    def predict(self, inputs: np.ndarray):
        """Generates prediction from network

        Args:
            inputs (List[float]): data to predict

        Raises:
            ValueError: raise when number of inputs
                        is not correct

        Returns:
            int: predicted value
        """
        if len(inputs) != self._inputs_number:
            raise ValueError('Invalid number of inputs')

        outputs = self.forward_propagate(inputs)
        
        # return outputs.index(max(outputs))
        
        choice = self.softmax(outputs)
        return outputs.index(choice)

    def backward_propagate_error(self, expected: int):
        """Calculates error for each output neuron
        and propogate error signal backwards through network

        Args:
            expected (int): expected output value
        """
        for layer in reversed(self._layers):
            errors = []
            if layer != self._output_layer:
                for index in range(len(layer)):
                    error = 0.0
                    for neuron in layer.get_next_layer().get_neurons():
                        error += neuron.get_weight()[index] *\
                            neuron.get_delta()
                    errors.append(error)
            else:
                for index, neuron in enumerate(layer.get_neurons()):
                    errors.append(neuron.get_output() - expected[index])
            for index, neuron in enumerate(layer.get_neurons()):
                delta = errors[index] * neuron.transfer_derivate(
                                        self._activation_type)
                neuron.set_delta(delta)

    def update_weights(self, row: np.ndarray, learning_rate: float):
        """Updates weights in network

        Args:
            row (List[list]): input data
            learning_rate (float): hyper parameter of neural network
        """
        for index, layer in enumerate(self._layers):
            inputs = row
            if layer != self._hidden_layers[0]:
                inputs = [
                    neuron.get_output() for neuron in
                    self._layers[index-1].get_neurons()
                ]
            for neuron in layer.get_neurons():
                for index, input in enumerate(inputs):
                    new_weight = neuron.get_weight()[index] - learning_rate *\
                        neuron.get_delta() * input
                    neuron.set_weight(new_weight, index)
                neuron.get_weight()[-1] = neuron.get_weight()[-1] -\
                    learning_rate * neuron.get_delta()

    def train(self, data: np.ndarray, learning_rate: int):
        """Train network epochs-time for each row in data

        Args:
            data (np.ndarray): input data to train
            learning_rate (int): hyper parameter
        """

        for epoch in range(self._epochs_number):
            self.estimate_time(epoch)

            for row in data:
                image = row[0]
                label = row[1]
                self.forward_propagate(image)
                expected = [0 for _ in range(self._outputs_number)]
                expected[label] = 1
                self.backward_propagate_error(expected)
                self.update_weights(image, learning_rate)

    def back_propagation(
        self,
        training_data: np.ndarray,
        test_data: np.ndarray,
        learning_rate: int
    ):
        """Backpropagation using gradient descent.
        Predicts output for test_data.

        Args:
            training_data (np.ndarray): data to train
            test_data (np.ndarray): data to predict output
            learning_rate (int): hyper parameter

        Returns:
            List[int]: predictions for each test data
        """
        self.train(training_data, learning_rate)
        predictions = []
        for row in test_data:
            image = row[0]
            prediction = self.predict(image)
            predictions.append(prediction)
        return predictions
