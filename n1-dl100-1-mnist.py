#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
@File    :   n1-dl100-1-mnist.py
@Time    :   2025/03/13 16:07:17
@Author  :   ljc 
@Version :   1.0
@Desc    :   day1 deeplearning 100 samples, mnist
'''

# here put the import lib
import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

if torch.backends.mps.is_available():
    device = torch.device('mps')
else:
    device = torch.device('cpu')

# 定义数据转换
transforms = transforms.Compose([
    transforms.ToTensor(), 
    transforms.Normalize((0.5,), (0.5,))])

# 下载mnist 数据集
trainset = torchvision.datasets.MNIST(root='data', train=True, download=True, transform=transforms)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)

testset = torchvision.datasets.MNIST(root='data', train=False, download=True, transform=transforms)
testloader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=False)

train_images, train_labels = next(iter(trainloader))
test_images, test_labels = next(iter(testloader))
# 输出数据形状
print(f"Train Images Shape: {train_images.shape}, Train Labels Shape: {train_labels.shape}")
print(f"Test Images Shape: {test_images.shape}, Test Labels Shape: {test_labels.shape}")
train_images, train_labels = train_images.to(device), train_labels.to(device)
test_images, test_labels = test_images.to(device), test_labels.to(device)

# 定义神经网络
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64*7*7, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = F.dropout(x,0.5,training=self.training)
        x = F.relu(self.fc2(x))
        return F.log_softmax(x, dim=1)

net = Net().to(device)
print(net)
optimizer = optim.Adam(net.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()
num_epochs = 5
for epoch in range(num_epochs):
    net.train()
    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        if i % 100 == 99:
            print(f"Epoch: {epoch+1}, Batch: {i+1}, Loss: {running_loss/100}")
            running_loss = 0.0
    net.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    print(f"Epoch: {epoch+1}, Accuracy: {100*correct/total}")