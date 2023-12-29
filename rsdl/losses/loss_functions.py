from rsdl import Tensor


def mean_square_errors(predicts: Tensor, actual: Tensor):
    mse = ((predicts - actual) ** 2).sum()
    mse.data /= predicts.shape[0]
    return mse


def cce(predicts: Tensor, actual: Tensor):
    cross_entropy = actual.__mul__(predicts.log()).sum().__neg__()
    return cross_entropy
