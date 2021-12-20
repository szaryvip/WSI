from typing import List
from math import exp


class Neuron:
    def __init__(self, weights: List[float], bias=0):
        self._weights = weights
        self._output = 0
        self._delta = 0
        self._bias = bias

    def get_weight(self):
        return self._weights

    def set_weight(self, weight: float, index: int):
        self._weights[index] = weight

    def get_output(self):
        return self._output

    def set_output(self, output):
        self._output = output

    def get_delta(self):
        return self._delta

    def set_delta(self, delta):
        self._delta = delta

    def activate(self, inputs: list):
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
        if activation_function == 'sigmoid':
            return 1 / (1 + exp(-activation))
        else:
            return max(0, activation)

    def update_output(self, inputs: list):
        output = self.transfer(
            self.activate(inputs)
        )
        self.set_output(output)

        return output
