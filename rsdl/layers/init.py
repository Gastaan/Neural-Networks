import numpy as np
from enum import Enum


class Init(Enum):
    XAVIER = "xavier"
    HE = "he"
    ZERO = "zero"
    ONE = "one"


def xavier_initializer(shape):
    return np.random.randn(*shape) * np.sqrt(1/shape[0], dtype=np.float64)


def he_initializer(shape):
    return np.random.randn(*shape) * np.sqrt(2/shape[0], dtype=np.float64)


def zero_initializer(shape):
    return np.zeros(shape, dtype=np.float64)


def one_initializer(shape):
    return np.ones(shape, dtype=np.float64)


def initializer(shape, mode="xavier"):
    if mode == Init.XAVIER.value:
        return xavier_initializer(shape)
    if mode == Init.HE.value:
        return he_initializer(shape)
    if mode == Init.ZERO.value:
        return he_initializer(shape)
    if mode == Init.ONE.value:
        return he_initializer(shape)
    else:
        raise NotImplementedError("Not implemented initializer method")
