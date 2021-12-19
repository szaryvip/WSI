class Layer:
    def __init__(self, neurons):
        self._neurons = neurons
        self._next_layer = None

    def get_next_layer(self):
        return self._next_layer

    def set_next_layer(self, next_layer):
        self._next_layer = next_layer
