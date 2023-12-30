import sys
import numpy as np
from torch.utils.data import DataLoader

from torchvision import datasets
from torchvision.transforms import ToTensor

from rsdl import Tensor
from rsdl.activations import Softmax, Tanh, Sigmoid, Relu, LeakyRelu
from rsdl.layers import Linear
from rsdl.optim import Adam
from rsdl.losses import mean_square_errors, cross_entropy

sys.setrecursionlimit(10000)

train_set = datasets.MNIST(
    root='./data',
    train=True,
    download=True,
    transform=ToTensor()
)

test_set = datasets.MNIST(
    root='./data',
    train=False,
    download=False,
    transform=ToTensor()
)

batch_size = 32

train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)


class Model:
    def __init__(self, input_size: int, hidden_size: int, output_size: int) -> None:
        super(Model, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        self.layer0 = Linear(input_size, hidden_size, need_bias=True)
        # self.layer1 = Linear(hidden_size, hidden_size, need_bias=True)
        # self.layer2 = Linear(hidden_size, hidden_size, need_bias=True)
        # self.layer3 = Linear(hidden_size, hidden_size, need_bias=True)
        # self.layer4 = Linear(hidden_size, hidden_size, need_bias=True)
        self.layer5 = Linear(hidden_size, output_size, need_bias=True)
        self.optimizer = Adam(layers=[self.layer0, self.layer5])

    def forward(self, x):
        # Assuming x is a tensor with size (batch_size, 1, 28, 28)
        x = Relu(self.layer0.forward(x))
        # x = Relu(self.layer1(x))
        # x = Relu(self.layer2(x))
        # x = Relu(self.layer3(x))
        # x = Relu(self.layer4(x))
        x = Relu(self.layer5.forward(x))
        return x

    def evaluate_accuracy(self, data_loader):
        correct_count = 0
        total_count = 0

        for images, labels in data_loader:
            flattened_images = images.view([images.size(0), -1])
            numpy_images = flattened_images.numpy()
            inputs = Tensor(numpy_images)

            predicted = self.forward(inputs)
            predicted_labels = np.argmax(predicted.data, axis=1)

            correct_count += np.sum(predicted_labels == labels.numpy())
            total_count += len(labels)

        accuracy = correct_count / total_count
        return accuracy


model = Model(input_size=784, hidden_size=100, output_size=10)


for epoch in range(5):

    epoch_loss = 0.0
    batch = 0

    for images, labels in train_loader:
        batch += 1

        flattened_images = images.view([images.size(0), -1])
        numpy_images = flattened_images.numpy()
        inputs = Tensor(numpy_images)

        predicted = model.forward(inputs)

        actual = Tensor(np.zeros((len(labels), 10)))
        actual.data[np.arange(len(labels)), labels] = 1

        loss = cross_entropy(predicted, actual)

        loss.zero_grad()
        loss.backward()

        epoch_loss += loss.data

        model.optimizer.step()
        model.optimizer.zero_grad()

    print(f"Epoch {epoch + 1}, Loss: {epoch_loss}, Training Accuracy: {model.evaluate_accuracy(train_loader)}, Test Accuracy: {model.evaluate_accuracy(test_loader)}")