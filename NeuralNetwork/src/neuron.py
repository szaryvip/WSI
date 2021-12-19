class Neuron:
    _weights = None
    _delta = None
    _output = None
    
    def __init__(self):
        self._weights
        self._output
        self._delta
        
    def get_weight(self):
        return self._weights
        
    def set_weight(self, weight, index):
        self._weights[index] = weight
    
    def get_output(self):
        return self._output
        
    def set_output(self, output):
        self._output = output
        
    def get_delta(self):
        return self._delta
        
    def set_delta(self, delta):
        self._delta = delta
    