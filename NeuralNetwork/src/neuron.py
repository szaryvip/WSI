from typing import List
from math import exp


class Neuron:
    def __init__(self, weights: List[float], bias=0):
        """Initialize Neuron class that
        contains weights and bias

        Args:
            weights (List[float]): weights if inputs
            bias (int, optional): Defaults to 0.
        """
        self._weights = weights
        self._output = 0
        self._delta = 0
        self._bias = bias

    def get_weight(self):
        """Returns weights of inputs

        Returns:
            List[float]: weights
        """
        return self._weights

    def set_weight(self, weight: float, index: int):
        """Sets weight in neuron on index

        Args:
            weight (float): weight to set
            index (int): which weight it is
        """
        self._weights[index] = weight

    def get_output(self):
        """Returns outputs of neuron

        Returns:
            list[float]: outputs
        """
        return self._output

    def set_output(self, output):
        """Sets outputs in neuron

        Args:
            output (list[float]): outputs to set
        """
        self._output = output

    def get_delta(self):
        """Returns delta of neuron

        Returns:
            float: delta
        """
        return self._delta

    def set_delta(self, delta):
        """Sets delta in neuron

        Args:
            delta (float): delta to set
        """
        self._delta = delta

    def activate(self, inputs: list):
        """Calculates activation of neuron
        given input

        Args:
            inputs (list): data

        Returns:
            float: calculated activation
        """
        weights = self._weights

        return sum(
            [
                input * weight
                for input, weight in zip(inputs, weights)
            ]
        ) + self._bias

    def transfer(
        self,
        activation: float,
        activation_function: str = 'sigmoid'
    ):
        """Calculates real output of neuron

        Args:
            activation (float): calculated activation
            activation_function (str, optional): Function type.
            Defaults to 'sigmoid'.

        Returns:
            float: output of neuron
        """
        if activation_function == 'sigmoid':
            return 1 / (1 + exp(-activation))
        if activation_function == 'relu':
            return max(0, activation)

    def transfer_derivate(self, activation_function: str):
        """Calculates derivative of output for activate function

        Args:
            activation_function (str): type of function to use

        Returns:
            float: calculated derivative
        """
        if activation_function == "sigmoid":
            return self._output * (1.0 - self._output)
        if activation_function == 'relu':
            if self._output >= 0:
                return 1
            else:
                return 0

    def update_output(self, inputs: list):
        """Updates output of neuron

        Args:
            inputs (list): input to neuron

        Returns:
            list: output of neuron
        """
        output = self.transfer(
            self.activate(inputs)
        )
        self.set_output(output)

        return output
