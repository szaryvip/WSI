class Neuron:
    _weight = None
    _delta = None
    _output = None
    _expected = None
    
    def __init__(self):
        self._weight
        self._output
        self._delta
        self._expected
        
    def get_weight(self):
        return self._weight
        
    def set_weight(self, weight, index):
        self._weight[index] = weight
    
    def get_output(self):
        return self._output
        
    def set_output(self, output):
        self._output = output
        
    def get_delta(self):
        return self._delta
        
    def set_delta(self, delta):
        self._delta = delta
    
    def get_expected(self):
        return self._expected
    
    def set_expected(self, expected):
        self._expected = expected
    