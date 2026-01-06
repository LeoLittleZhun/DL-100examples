#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
@File    :   n1-dl100-8-WeatherClsYolo.py
@Time    :   2025/04/15 18:23:11
@Author  :   ljc 
@Version :   1.0
@Desc    :   使用yolov12x进行目标检测，并且标记出目标
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
from ultralytics import YOLO
import cv2
# 然后尝试加载模型

# 1. 使用mps
if torch.backends.mps.is_available():
    device = torch.device('mps')
else:
    device = torch.device('CPU')

# 2.加载yolo模型
model = YOLO('models/yolo11x.pt')
model = model.to(device)

# 3.随机取一张猫或者狗的图片
pic_dir = 'data/cat-test.jpg'
# /Users/liujiancheng/Documents/develop/vscode/0.test/data
#读取整个照片并更换为RGB格式
img_bgr = cv2.imread(pic_dir)
img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

# 4.使用yolo12进行目标检测
results = model(img_rgb)[0]
#获取目标坐标
boxes = results.boxes.data.cpu().numpy()
#获取目标置信度
confidences = results.boxes.conf.cpu().numpy()
#获取目标类别
classes = results.boxes.cls.cpu().numpy()

class_names = [
    'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
    'train', 'truck', 'boat', 'traffic light', 'fire hydrant',
    'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog',
    'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra',
    'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase',
    'frisbee', 'skis', 'snowboard', 'sports ball', 'kite',
    'baseball bat', 'baseball glove', 'skateboard', 'surfboard',
    'tennis racket', 'bottle', 'wine glass', 'cup', 'fork',
    'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich',
    'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut',
    'cake', 'chair', 'couch', 'potted plant', 'bed', 'dining table',
    'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard',
    'cell phone', 'microwave', 'oven', 'toaster', 'sink',
    'refrigerator', 'book', 'clock', 'vase', 'scissors',
    'teddy bear', 'hair drier', 'toothbrush'
]
#把框都读出来并标记上
for box in boxes:
    x1, y1, x2, y2, conf, cls = box
    cls = int(cls)
    # 画框
    cv2.rectangle(img_rgb, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)
    # 画标签
    label = f'Class: {int(cls)}, Conf: {conf:.2f}'
    cv2.putText(img_rgb, label, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

# 5.显示图片
cv2.imshow('Image', img_rgb)
cv2.waitKey(0)
cv2.destroyAllWindows()

