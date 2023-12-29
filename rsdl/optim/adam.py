from rsdl.optim import Optimizer
from rsdl.layers import Linear
import numpy as np


class Adam(Optimizer):
    def __init__(self, layers: [Linear], learning_rate=0.2, decay_factor_b0=0.5, decay_factor_b1=0.5):
        super().__init__(layers)
        self.decay_factor_b0 = decay_factor_b0
        self.decay_factor_b1 = decay_factor_b1
        self.learning_rate = learning_rate
        self.time = 0

        for layer in self.layers:
            layer.weight_first_moment = np.zeros_like(layer.weight)
            layer.weight_second_moment = np.zeros_like(layer.weight)
            if layer.need_bias:
                layer.bias_first_moment = np.zeros_like(layer.weight)
                layer.bias_second_moment = np.zeros_like(layer.weight)

    def step(self):
        decay_factor_b0 = self.decay_factor_b0
        decay_factor_b1 = self.decay_factor_b1
        learning_rate = self.learning_rate
        epsilon = 1e-15
        time = self.time

        for layer in self.layers:
            layer.weight_first_moment = (decay_factor_b0 * layer.weight_first_moment +
                                         (1 - decay_factor_b0) * layer.weight.grad)
            layer.weight_second_moment = (decay_factor_b1 * layer.bias_second_moment +
                                          (1 - decay_factor_b1) * (layer.weight.grad ** 2))

            first = layer.weight_first_moment / (1 - decay_factor_b0 ** time)

            second = layer.weight_second_moment / (1 - decay_factor_b1 ** time)

            layer.weight = layer.weight - learning_rate * first * ((second + epsilon) ** -1)

            if layer.need_bias:
                layer.bias_first_moment = (decay_factor_b0 * layer.bias_first_moment +
                                           (1 - decay_factor_b0) * layer.bias.grad)
                layer.bias_second_moment = (decay_factor_b0 * layer.bias_second_moment +
                                            (1 - decay_factor_b0) * layer.bias.grad)

                first = layer.bias_first_moment / (1 - decay_factor_b0 ** time)

                second = layer.bias_second_moment / (1 - decay_factor_b1 ** time)

                layer.bias = layer.bias - learning_rate * first * ((second + epsilon) ** -1)

        self.time += 1
