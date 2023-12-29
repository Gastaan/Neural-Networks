from rsdl.layers import Linear


class Optimizer:
    def __init__(self, layers: [Linear]):
        self.layers = layers
    
    def zero_grad(self):
        for l in self.layers:
            l.zero_grad()