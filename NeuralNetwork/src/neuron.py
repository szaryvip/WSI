from typing import List


class Neuron:
    def __init__(self, weights: List[float]):
        self._weights = weights
        self._output = 0
        self._delta = 0

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
