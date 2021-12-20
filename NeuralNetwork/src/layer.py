class Layer:
    def __init__(self, neurons: list):
        self._neurons = neurons
        self._next_layer = None

    def get_neurons(self):
        return self._neurons

    def get_next_layer(self):
        return self._next_layer

    def set_next_layer(self, next_layer):
        self._next_layer = next_layer

    def __len__(self):
        return len(self._neurons)
