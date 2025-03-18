#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
@File    :   n1-dl100-3-clothscls.py
@Time    :   2025/03/18 17:25:24
@Author  :   ljc 
@Version :   1.0
@Desc    :   git/n1-dl100-3-clothscls.py, fashion mnist classification
'''

# here put the import lib
import torch
import torch.nn as nn
import torchvision, torchvision.transforms as transforms
import matplotlib.pyplot as plt
import os

# 1.使用mps作为device
if torch.backends.mps.is_available():
    device = torch.device('mps')
else:    
    device = torch.device('cpu')

# 2.加载数据集，导入数据
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])
train_set = torchvision.datasets.FashionMNIST(root='data', train=True, download=True, transform=transform)
test_set = torchvision.datasets.FashionMNIST(root='data', train=False, download=True, transform=transform)

train_loader = torch.utils.data.DataLoader(train_set, batch_size=64, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_set, batch_size=64, shuffle=False)
# 输出数据形状
train_images, train_labels = next(iter(train_loader))
test_images, test_labels = next(iter(test_loader))
print(f"Train Images Shape: {train_images.shape}, Train Labels Shape: {train_labels.shape}")
print(f"Test Images Shape: {test_images.shape}, Test Labels Shape: {test_labels.shape}")

# 3.数据可视化
# class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
# plt.figure(figsize=(20,10))
# for i in range(20):
#     plt.subplot(5,10,i+1)
#     plt.xticks()
#     plt.yticks()
#     plt.grid(False)
#     plt.imshow(train_images[i].squeeze().numpy(), cmap='gray')
#     plt.xlabel(class_names[train_labels[i].item()])
# plt.show()

# 4.定义神经网络
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(256)

        self.fc1 = nn.Linear(2304, 128)
        self.fc2 = nn.Linear(128, 10)

        self.relu = nn.ReLU(inplace=True) # inplace=True表示原地操作，节省内存
        self.dropout = nn.Dropout(0.2)

    def forward(self,x):
        x = self.pool(self.relu(self.bn1(self.conv1(x))))
        x = self.pool(self.relu(self.bn2(self.conv2(x))))
        x = self.relu(self.bn3(self.conv3(x)))
        x = self.pool(self.relu(self.bn4(self.conv4(x))))
        x = torch.flatten(x, start_dim=1)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x
    
# 5.实例化模型、损失函数、优化器
model = CNN().to(device)
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)
num_epochs = 5

# 6.训练模型
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for i, data in enumerate(train_loader, 0):
        images, labels = data
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = loss_fn(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        if i % 100 == 99:
            print(f"Epoch: {epoch+1}, Batch: {i+1}, Loss: {running_loss/100}")
            running_loss = 0.0
    scheduler.step()

model.eval()
correct = 0
total = 0
with torch.no_grad():
    for data in test_loader:
        images, labels = data
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
print(f"Epoch: {epoch+1}, Accuracy: {100*correct/total :.2f}%")
# 7.保存模型
if not os.path.exists('models'):
    os.makedirs('models')
torch.save(model.state_dict(), 'models/fashion_mnist_cnn.pth')