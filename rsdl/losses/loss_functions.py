from rsdl import Tensor


def mean_square_errors(predicts: Tensor, actual: Tensor):
    mse = ((predicts - actual) ** 2).sum()
    mse.data = mse.data * (predicts.shape[0] ** -1)
    return mse


def cross_entropy(predicts: Tensor, actual: Tensor):
    entropy = -(actual * predicts.log()).sum()
    return entropy
