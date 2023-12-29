from rsdl import Tensor
import numpy as np


def MeanSquaredError(preds: Tensor, actual: Tensor):
    mse = ((preds - actual) ** 2).mean()

    return mse


def CategoricalCrossEntropy(preds: Tensor, actual: Tensor):
    epsilon = 1e-15
    preds = np.clip(preds.data, epsilon, 1 - epsilon)
    cross_entropy = -np.sum(actual.data * np.log(preds))
    return cross_entropy
