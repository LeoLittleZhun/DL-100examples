#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
@File    :   n1-dl100-5-weathercls.py
@Time    :   2025/03/26 10:59:23
@Author  :   ljc 
@Version :   1.0
@Desc    :   使用resnet18 识别天气，准确率99.11%;
             测试resnet34，准确率95.11%；多次训练，取最优准确率96.89%
@Conclusion:  resnet34的准确率比resnet18低，可能是因为数据集太小了，resnet34的参数更多，容易过拟合
'''

# here put the import lib
import os
import torch.utils.data.dataloader
import torch,torchvision
import torchvision.transforms as transforms
import torchvision.datasets as dataset
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from PIL import Image

# 1. 使用mps
if torch.backends.mps.is_available():
    device = torch.device('mps')
else:
    device = torch.device('CPU')

# 2.数据处理
# 图像不是很标准的图像，分辨率大小不一，根据图像情况进行统一处理
# data_dir = os.getcwd() + '/weather_photos'
# print(os.path.exists(data_dir)) 
def get_padding(img, target_size):
    width, height = img.size
    scale = target_size / min(width, height)  
    new_width, new_height = int(width * scale), int(height * scale)

    delta_width = max(target_size - new_width, 0)
    delta_height = max(target_size - new_height, 0)
    left = delta_width // 2
    top = delta_height // 2
    right = delta_width - left
    bottom = delta_height - top

    return (new_width, new_height), (left, top, right, bottom)
target_size = 512
transform = transforms.Compose([
    transforms.Lambda(lambda img: img.resize(get_padding(img, target_size)[0], Image.BILINEAR)),
    transforms.Lambda(lambda img: transforms.functional.pad(img, get_padding(img, target_size)[1], padding_mode='reflect')),
    transforms.Resize((target_size,target_size)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])
data_set = dataset.ImageFolder('data/weather_photos', transform=transform)
train_size = int(len(data_set) * 0.8)
test_size = len(data_set) - train_size
train_set, test_set = torch.utils.data.random_split(data_set, [train_size, test_size])
train_loader = torch.utils.data.DataLoader(train_set, batch_size = 32, shuffle = True)
test_loader = torch.utils.data.DataLoader(test_set, batch_size = 32, shuffle=False)

# 3.实例化模型
# 考虑到自己做CNN的局限性，还是直接用Res_net18
# model = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.DEFAULT)
model = torchvision.models.resnet34(weights=torchvision.models.ResNet34_Weights.DEFAULT)
model.fc = nn.Linear(model.fc.in_features, 4)
model = model.to(device)
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr= 0.001)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)
num_epochs = 10

# 4.训练模型
for epoch in range(num_epochs):
    model.train()
    running_loss =0.0
    progress_bar = tqdm(enumerate(train_loader), total= len(train_loader), desc=f"Epoch {epoch+1}/{num_epochs}")

    for i,data in progress_bar:
        images, labels = data
        images = images.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = loss_fn(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        if i % 10 == 9 :
            progress_bar.set_postfix(loss=running_loss / 10)
            running_loss = 0.0
    scheduler.step()

# 5.保存模型
if not os.path.exists('models'):
    os.makedirs('models')
torch.save(model.state_dict(), 'models/5_weather_cls_resnet34.pth')

# 6.评估模型
model.eval()
correct = 0 
total = 0 
with torch.no_grad():
    for data in test_loader:
        images,labels = data
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f"Epoch: {epoch+1}, Accuracy: {100 * correct/total :.2f}%")