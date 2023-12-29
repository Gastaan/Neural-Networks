# Task 5
import numpy as np

from rsdl import Tensor
from rsdl.layers import Linear, Init
from rsdl.optim import SGD
from rsdl.losses.loss_functions import mean_square_errors

X = Tensor(np.random.randn(100, 3))
coef = Tensor(np.array([-7, +3, -9]))
y = X @ coef + 5

fc = Linear(3, 1)

optimizer = SGD(layers=[fc])


batch_size = 5

for epoch in range(100):

    epoch_loss = 0.0

    for start in range(0, 100, batch_size):
        end = start + batch_size

        inputs = X[start:end]

        predicted = fc.forward(inputs)

        actual = y[start:end]
        actual.data = actual.data.reshape(batch_size, 1)
        loss = mean_square_errors(predicted, actual)

        loss.zero_grad()
        loss.backward()
        print(loss.data)

        epoch_loss += loss

        optimizer.step()
        fc.zero_grad()

        if loss.data < 0.000001:
            break

print(fc.weight)
print(fc.bias)