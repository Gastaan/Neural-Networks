from rsdl import Tensor
from rsdl.layers import initializer, Init


class Linear:

    def __init__(self, in_channels, out_channels, need_bias=True, mode='xavier') -> None:
        # set input and output shape of layer
        self.shape = (in_channels, out_channels)
        self.need_bias = need_bias

        self.weight = Tensor(
            data=initializer(shape=(in_channels, out_channels), mode=mode),
            requires_grad=True
        )

        if self.need_bias:
            self.bias = Tensor(
                data=initializer(shape=(1, out_channels), mode=Init.ZERO),
                requires_grad=True
            )

    def forward(self, inp: 'Tensor') -> 'Tensor':
        # Perform linear transformation
        linear_result = input @ self.weight

        # Add bias if needed
        if self.need_bias:
            linear_result = linear_result + self.bias

        return linear_result
    
    def parameters(self):
        if self.need_bias:
            return [self. weight, self.bias]
        return [self. weight]
    
    def zero_grad(self):
        self.weight.zero_grad()
        if self.need_bias:
            self.bias.zero_grad()
            
    def __call__(self, inp):
        return self.forward(inp)
