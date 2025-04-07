#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
@File    :   n1-dl100-7-catanddogcls.py
@Time    :   2025/04/07 09:31:44
@Author  :   ljc 
@Version :   1.0
@Desc    :   None
'''

# here put the import lib
import os
import torch,torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F
import torchvision.datasets as dataset
from tqdm import tqdm
from PIL import Image

# 1.使用mps作为device
if torch.backends.mps.is_available():
    device = torch.device('mps')
else:
    device = torch.device('cpu')

# 2.数据处理
# 还是同样的标准，先处理一下所有的图片，统一大小
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
target_size = 448
transform = transforms.Compose([
    transforms.Lambda(lambda img: img.resize(get_padding(img, target_size)[0], Image.BILINEAR)),
    transforms.Lambda(lambda img: transforms.functional.pad(img, get_padding(img, target_size)[1], padding_mode='reflect')),
    transforms.Resize((target_size,target_size)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])
train_data_set = dataset.ImageFolder('data/1-cat-dog/train', transform=transform)
test_data_set = dataset.ImageFolder('data/1-cat-dog/val', transform=transform)
train_loader = torch.utils.data.DataLoader(train_data_set, batch_size = 32, shuffle = True)
test_loader = torch.utils.data.DataLoader(test_data_set, batch_size = 32, shuffle=False)

# 3.实例化模型，这次先试试VGG16
# model = models.vgg16(weights=torchvision.models.VGG16_Weights.DEFAULT)
# model.classifier[6] = nn.Linear(4096, 2) # 修改最后一层的输出为2
## vgg16的参数量太大,图片数量少，所以可能效果不好，改用resnet18
model = models.resnet18(weights=torchvision.models.ResNet18_Weights.DEFAULT)
model.fc = nn.Linear(model.fc.in_features, 2)
model = model.to(device)
# 损失函数、优化器
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr= 0.001)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)
num_epochs = 15

# 4.训练模型
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
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
torch.save(model.state_dict(), 'models/7_CatAndDog_cls_resnet18.pth')

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
