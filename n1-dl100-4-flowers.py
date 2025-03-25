#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
@File    :   n1-dl100-4-flowers.py
@Time    :   2025/03/19 8:19:13
@Author  :   ljc 
@Version :   1.0
@Desc    :   花朵识别
conclu   :   自己手搓CNN还是差别比较大，很难上升准确率，只有56%左右，调整到resnet18，准确率立马上升到95.1%
'''

# here put the import lib
import torch
import torch.nn as nn
import torchvision, torchvision.transforms as transforms
import torchvision.datasets as dataset
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
from PIL import Image

# 1.使用mps作为device
if torch.backends.mps.is_available():
    device = torch.device('mps')
else:
    device = torch.device('cpu')

# 2.下载数据并处理
dataset_url = "https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz"
data_dir = 'data/flower_photos'
if not os.path.exists(data_dir):
    torchvision.datasets.utils.download_and_extract_archive(dataset_url, 'data', filename='flower_photos.tgz')
    print('Downloaded and extracted the flower dataset')
else:
    print('Dataset already exists')
# 加载数据集
# 原始数据是 320 * 240 大小的数据，不是所有的height都是固定的，先把图片统一处理一下
# def get_padding(img, target_size):
#     # 计算填充边界的大小，是图片保持原始比例的情况下变为正方形，尽可能不丢失信息
#     width,height = img.size
#     scale = target_size / min(width,height) #短边对齐
#     new_width, new_height = int(width * scale), int(height * scale)

#     delta_width = target_size - new_width
#     delta_height = target_size - new_height
#     padding = (delta_width // 2,delta_height // 2, delta_width - (delta_width//2), delta_height - (delta_height//2))
#     return new_width, new_height, padding
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
target_size = 320
transform = transforms.Compose([
    # transforms.Lambda(lambda img: img.resize(get_padding(img,target_size)[0:2], Image.BILINEAR)),
    # transforms.Lambda(lambda img:transforms.functional.pad(img, get_padding(img,target_size)[2], padding_mode='reflect')),
    transforms.Lambda(lambda img: img.resize(get_padding(img, target_size)[0], Image.BILINEAR)),
    transforms.Lambda(lambda img: transforms.functional.pad(img, get_padding(img, target_size)[1], padding_mode='reflect')),
    # 因为原始图片height差别有点大，导致最后数据并不一致，报错：
    # RuntimeError: stack expects each tensor to be equal size, but got [3, 320, 320] at entry 0 and [3, 320, 319] at entry 49
    # 增加强制resize
    transforms.Resize((320,320)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])
data_set = dataset.ImageFolder('data/flower_photos',transform=transform)
train_size = int (len(data_set)*0.8)
test_size = len(data_set) - train_size
train_set, test_set = torch.utils.data.random_split(data_set, [train_size, test_size])
train_loader = torch.utils.data.DataLoader(train_set, batch_size= 128, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_set, batch_size=128, shuffle=False)
train_images, train_labels = next(iter(train_loader))
print(f"Train Images Shape:{train_images.shape}, Labels Shape:{train_labels.shape}")
# 计算参数特征量
def get_fc_input_dim(conv_block, input_size=(3, target_size, target_size)):
    dummy_input = torch.randn(1, *input_size)
    dummy_output = conv_block(dummy_input)
    return torch.numel(dummy_output)

# 3.定义神经网络
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv_block = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2)
        )

        # self.fc1 = nn.Linear(262144,128)
        # 算出来都是错的，看到有自动计算的方法，使用这个方法
        fc_input_dim = get_fc_input_dim(self.conv_block)
        self.fc1 = nn.Linear(fc_input_dim, 128)
        self.fc2 = nn.Linear(128,5)
        self.dropout = nn.Dropout(0.2)
    
    def forward(self,x):
        x = self.conv_block(x)
        x = torch.flatten(x, start_dim=1)
        x = self.fc1(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x
    
# 4. 实例化模型
# model = torch.compile(CNN().to(device))   # compile 好像和mps有兼容性问题，加速不能用。查了一下主要是inductor不兼容
# 可能解决方案：
# torch._inductor.config.triton = False  # 关闭 inductor 依赖 Triton
# torch._dynamo.config.suppress_errors = True  # 遇到错误回退到 eager mode
# model = torch.compile(model, backend="nvfuser")  # 用 nvfuser 代替 inductor
# model = CNN().to(device) #自己做CNN网络局限性比较大，实施resnet18
# model = torchvision.models.resnet18(pretrained=True) # pretrained 参数已经被weights 替代
model = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.DEFAULT)
model.fc = nn.Linear(model.fc.in_features,5)
model = model.to(device)
loss_fn = nn.CrossEntropyLoss()  #@label_smoothing避免模型过拟合，提高泛化能力。
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
# scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)
num_epochs = 10

# 5. 训练模型
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    progress_bar = tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Epoch {epoch+1}/{num_epochs}")

    for i,data in progress_bar:
        images, labels = data
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = loss_fn(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        if i % 10 == 9:
            progress_bar.set_postfix(loss=running_loss / 10)
            running_loss = 0.0
    scheduler.step()

# 6. 保存模型,因为训练太慢了，先保存模型，避免后面重复训练
if not os.path.exists('models'):
    os.makedirs('models')
torch.save(model.state_dict(), 'models/4_flowers_cls_cnn.pth')

# 6. 评估模型
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

print(f"Epoch: {epoch+1}, Accuracy: {100 * correct/total :.2f}%")
