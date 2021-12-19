class Layer:
    _next_layer = None
    _neurons = []
    
    def __init__(self):
        pass
    
    def get_next_layer(self):
        return self._next_layer
    