from rsdl.optim import Optimizer
from rsdl.layers import Linear
import numpy as np

class Momentum(Optimizer):
    def __init__(self, layers: [Linear], learning_rate=0.1, momentum=0.3):
        super().__init__(layers)
        self.learning_rate = learning_rate
        self.momentum = momentum

        for layer in self.layers:
            layer.weight_momentum = np.zeros_like(layer.weight)
            if layer.need_bias:
                layer.bias_momentum = np.zeros_like(layer.bias)

    def step(self):
        for layer in self.layers:
            momentum = self.momentum
            learning_rate = self.learning_rate

            layer.weight_momentum = momentum * layer.weight_momentum - (1.0 - momentum) * layer.weight.grad
            layer.weight = layer.weight + learning_rate * layer.weight_momentum

            if layer.need_bias:
                layer.bias_momentum = momentum * layer.bias_momentum - (1.0 - momentum) * layer.bias.grad
                layer.bias = layer.bias + learning_rate * layer.bias_momentum
