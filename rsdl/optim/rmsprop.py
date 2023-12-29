from rsdl.optim import Optimizer
from rsdl.layers import Linear
import numpy as np


class RMSprop(Optimizer):
    def __init__(self, layers: [Linear], learning_rate=0.2, decay_factor=0.5):
        super().__init__(layers)
        self.decay_factor = decay_factor
        self.learning_rate = learning_rate

        for layer in self.layers:
            layer.weight_squared_gradient = np.zeros_like(layer.weight)
            if layer.need_bias:
                layer.bias_squared_gradient = np.zeros_like(layer.bias)

    def step(self):
        decay_factor = self.decay_factor
        learning_rate = self.learning_rate
        epsilon = 1e-15

        for layer in self.layers:
            layer.weight_squared_gradient = (decay_factor * layer.weight_squared_gradient +
                                             (1 - decay_factor) * layer.weight.grad ** 2)
            layer.weight = (layer.weight - (learning_rate * (layer.weight_squared_gradient ** 0.5 + epsilon) ** -1)
                            * layer.weight.grad)

            if layer.need_bias:
                layer.bias_squared_gradient = (decay_factor * layer.bias_squared_gradient +
                                               (1 - decay_factor) * layer.bias.grad ** 2)
                layer.bias = (layer.bias - (learning_rate * (layer.bias_squared_gradient ** 0.5 + epsilon) ** -1)
                              * layer.bias.grad)
