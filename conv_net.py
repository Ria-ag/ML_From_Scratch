"""
Convolutional Neural Networks on CIFAR-10

Goal:
- Compare fully connected networks with convolutional architectures

What this explores:
- Image normalization and visualization
- Dense networks vs CNNs for image classification
- How spatial structure improves learning on visual data
"""

import random
import numpy as np
import matplotlib.pyplot as plt
import torchvision as tv
import torchvision.transforms as transforms
import torch
import torch.nn as nn
from PIL import Image

transform = transforms.Compose(
    [transforms.ToTensor(),
     # normalization - Adjusts features so that they are on a similar scale
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

train_set = tv.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)

test_set = tv.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform)

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
print("Number of training images:", len(train_set))
print("Number of testing images:", len(test_set))
image, label = train_set[0]
print("Shape of each image:", image.numpy().shape)
print("Example label:", label)

batch_size = 20
train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size,
                                          shuffle=True, num_workers=2)
test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size,
                                         shuffle=False, num_workers=2)

num_to_viz = 3
toPilImage = transforms.ToPILImage()

for i in range(num_to_viz):
    j = random.randrange(50000)
    # unnormalize image
    image_to_viz = toPilImage(train_set[j][0] / 2 + 0.5)
    image_to_viz_label = train_set[j][1]
    plt.imshow(image_to_viz)
    plt.title(classes[image_to_viz_label])
    plt.show()

class CIFAR10_DNN(nn.Module):
  def __init__(self, input_size = 3*32*32, hidden_sizes=[256,128,64], output_size=10):
    super(CIFAR10_DNN, self).__init__()

    all_layers = []
    in_dim = input_size
    for h in hidden_sizes:
        all_layers.append(nn.Linear(in_dim, h))
        all_layers.append(nn.ReLU())
        in_dim = h

loss_func = nn.CrossEntropyLoss() # Mean Squared Error
optimizer = torch.optim.Adam(model.parameters(), lr=0.001) # Adaptive Optimizer

num_epochs = 5
for epoch in range(num_epochs):
  for images, labels in train_loader:
    images = images.reshape(-1, 3*32*32)

    outputs = model(images)
    loss = loss_func(outputs, labels)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

with torch.no_grad():
    n_correct = 0
    n_samples = 0
    for images, labels in test_loader:
        images = images.reshape(-1, 3*32*32)
        outputs = model(images)
        # max returns (value ,index)
        _, predicted = torch.max(outputs.data, 1)
        n_samples += labels.size(0)
        n_correct += (predicted == labels).sum().item()
    acc = 100.0 * n_correct / n_samples
    print(f'Accuracy of the network on the 10000 test images: {acc} %')

class CNN_CIFAR10(nn.Module):
    def __init__(self):
        super(CNN_CIFAR10, self).__init__()

        input_channels = 3  # RGB images
        num_classes = 10

        # Convolutional Layers
        self.conv_l1 = nn.Conv2d(in_channels=input_channels, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.conv_l2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.conv_l3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)

        # Pooling layer
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Fully connected layers
        # CIFAR-10 images: 32x32 → after 3 pools: 4x4
        self.fc1 = nn.Linear(128 * 4 * 4, 256)
        self.fc2 = nn.Linear(256, num_classes)

        # Activations
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)

    def forward(self, input):
        x = self.relu(self.conv_l1(input))
        x = self.maxpool(x)

        x = self.relu(self.conv_l2(x))
        x = self.maxpool(x)

        x = self.relu(self.conv_l3(x))
        x = self.maxpool(x)

        x = torch.flatten(x, 1)
        x = self.dropout(self.relu(self.fc1(x)))
        output = self.fc2(x)
        return output

cnn_model = CNN_CIFAR10()

loss_func = nn.CrossEntropyLoss() # Mean Squared Error

num_epochs = 5

for epoch in range(num_epochs):
    for images, labels in train_loader:
        outputs = cnn_model(images)
        loss = loss_func(outputs, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

with torch.no_grad():
    n_correct = 0
    n_samples = 0
    for images, labels in test_loader:
        outputs = cnn_model(images)
        # max returns (value ,index)
        _, predicted = torch.max(outputs.data, 1)
        n_samples += labels.size(0)
        n_correct += (predicted == labels).sum().item()
    acc = 100.0 * n_correct / n_samples
    print(f'Accuracy of the network on the 10000 test images: {acc} %')
