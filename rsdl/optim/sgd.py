from rsdl.optim import Optimizer
from rsdl.layers import Linear


class SGD(Optimizer):
    def __init__(self, layers: [Linear], learning_rate=0.1):
        super().__init__(layers)
        self.learning_rate = learning_rate

    def step(self):
        for layer in self.layers:
            layer.weight = layer.weight - self.learning_rate * layer.weight.grad
            if layer.need_bias:
                layer.bias = layer.bias - self.learning_rate * layer.bias.grad

