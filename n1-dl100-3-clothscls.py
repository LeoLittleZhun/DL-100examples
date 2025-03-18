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

