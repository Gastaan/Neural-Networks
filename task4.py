# Task 4
import numpy as np

from rsdl import Tensor
from rsdl.losses.loss_functions import mean_square_errors

X = Tensor(np.random.randn(100, 3))
coef = Tensor(np.array([-7, +3, -9]))
y = X @ coef + 6

w = Tensor(np.random.randn(3), requires_grad=True)
b = Tensor(np.random.randn(), requires_grad=True)

print(w)
print(b)

learning_rate = 0.1
batch_size = 5

for epoch in range(100):

    epoch_loss = 0.0

    for start in range(0, 100, batch_size):
        end = start + batch_size

        inputs = X[start:end]

        predicted = inputs @ w + b
        actual = y[start:end]

        loss = mean_square_errors(predicted, actual)

        loss.zero_grad()
        loss.backward()

        epoch_loss += loss.data

        w.data = w.data - learning_rate * w.grad.data
        b.data = b.data - learning_rate * b.grad.data

        w.zero_grad()
        b.zero_grad()

print(w)
print(b)
