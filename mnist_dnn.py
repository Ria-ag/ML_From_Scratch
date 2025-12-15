"""
Fully Connected Neural Network on MNIST

Goal:
- Move beyond linear models to learned representations

What this explores:
- Multilayer perceptrons with ReLU activations
- Training with backpropagation using PyTorch
- How depth improves classification performance on MNIST
"""

import numpy as np
import matplotlib.pyplot as plt
import sklearn
import torchvision as tv
import torchvision.transforms as transforms
import torch
import torch.nn as nn

train_data = tv.datasets.MNIST('./data', train=True, transform=transforms.ToTensor(), download=True)
test_data = tv.datasets.MNIST('./data', train=False, transform=transforms.ToTensor(), download=True)
print("Train data shape:", train_data.data.numpy().shape)
print("Train labels shape:", train_data.targets.numpy().shape)
print("Test data shape:", test_data.data.numpy().shape)
print("Test labels shape:", test_data.targets.numpy().shape)

batch_size = 60
train_loader = torch.utils.data.DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_data, batch_size=batch_size, shuffle=False)

num_of_digits_to_viz = 3
for i in range(num_of_digits_to_viz):
    to_reshape = train_data.data.numpy()[i]
    plt.matshow(to_reshape.reshape(28, 28))
    plt.show()
    print(f"Associated Label: {train_data.targets.numpy()[i]}")

class MNIST_DNN(nn.Module):
    def __init__(self, input_size=784, hidden_sizes=[256, 128, 64], output_size=10):
        super(MNIST_DNN, self).__init__()

        layers = []
        in_dim = input_size
        for h in hidden_sizes:
            layers.append(nn.Linear(in_dim, h))
            layers.append(nn.ReLU())
            in_dim = h
        layers.append(nn.Linear(in_dim, output_size))

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)

model = MNIST_DNN()
print(model)

loss_func = nn.CrossEntropyLoss() # Mean Squared Error
optimizer = torch.optim.Adam(model.parameters(), lr=0.001) # Adaptive Optimizer

num_epochs = 3
for epoch in range(num_epochs):
    for images, labels in train_loader:
        # Flatten 28x28 images into vectors of size 784
        images = images.reshape(-1, 28*28)

        # Forward pass
        outputs = model(images)
        loss = loss_func(outputs, labels)

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

with torch.no_grad():
    n_correct = 0
    n_samples = 0
    for images, labels in test_loader:
        images = images.reshape(-1, 28*28)
        outputs = model(images)
        # max returns (value ,index)
        _, predicted = torch.max(outputs.data, 1)
        n_samples += labels.size(0)
        n_correct += (predicted == labels).sum().item()
    acc = 100.0 * n_correct / n_samples
    print(f'Accuracy of the network on the 10000 test images: {acc} %')
