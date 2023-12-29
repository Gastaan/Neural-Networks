from rsdl import Tensor, Dependency
import numpy as np


def Sigmoid(t: Tensor) -> Tensor:
    return (-t.exp() + Tensor(data=np.ones_like(1))) ** -1


def Tanh(t: Tensor) -> Tensor:
    return (t.exp() - (-t).exp()) * ((t.exp() + (-t).exp()) ** -1)


def Softmax(t: Tensor) -> Tensor:
    return t.exp() * (Tensor(np.ones_like(t) * t.exp().sum()) ** -1)


def Relu(t: Tensor) -> Tensor:
    data = np.maximum(0, t.data)

    req_grad = t.requires_grad
    if req_grad:
        def grad_fn(grad: np.ndarray):
            # Use np.where to create a mask for positive values
            mask = np.where(t.data > 0, 1, 0)
            return grad * mask

        depends_on = [Dependency(t, grad_fn)]
    else:
        depends_on = []

    return Tensor(data=data, requires_grad=req_grad, depends_on=depends_on)


def LeakyRelu(t: Tensor, leak=0.05) -> Tensor:
    data = np.maximum(leak * t.data, t.data)

    req_grad = t.requires_grad
    if req_grad:
        def grad_fn(grad: np.ndarray):
            mask = np.where(t.data > 0, 1, leak)
            return grad * mask

        depends_on = [Dependency(t, grad_fn)]
    else:
        depends_on = []

    return Tensor(data=data, requires_grad=req_grad, depends_on=depends_on)