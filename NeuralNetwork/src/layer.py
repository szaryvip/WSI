class Layer:
    def __init__(self, neurons: list):
        """Initialize Layer object that contains
        list of neurons in this layer and pointer
        to next layer

        Args:
            neurons (list): neurons in layer
        """
        self._neurons = neurons
        self._next_layer = None

    def get_neurons(self):
        """Returns neurons from this layer

        Returns:
            List[Neuron]: list of neurons
        """
        return self._neurons

    def get_next_layer(self):
        """Returns next layer

        Returns:
            Layer: next layer
        """
        return self._next_layer

    def set_next_layer(self, next_layer):
        """Sets next layer pointer

        Args:
            next_layer (Layer): pointer to next layer
        """
        self._next_layer = next_layer

    def __len__(self):
        """Returns size of layer 
        which is equal to number of neurons in 
        this layer

        Returns:
            int: size of layer
        """
        return len(self._neurons)
