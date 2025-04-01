#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
@File    :   n1-dl100-6-weatherclsAPP.py
@Time    :   2025/04/01 09:29:32
@Author  :   ljc 
@Version :   1.0
@Desc    :   使用训练好的pth文件，推理应用，实现上传一张天气图片，识别出结果
上一文件训练好的模型是：models/5_weather_cls_resnet18.pth；使用flask实现web应用
@Conclusion:  训练好的模型，推理应用，训练准确率99.11%

'''

# here put the import lib
from flask import Flask, request, jsonify, render_template
import torch
import torchvision.transforms as transforms
import torchvision.models as models
import torch.nn.functional as F
import os
from PIL import Image

app = Flask(__name__)

# 1.使用mps作为device
if torch.backends.mps.is_available():
    device = torch.device('mps')
else:
    device = torch.device('cpu')

# 2.数据处理
# 图像不是很标准的图像，分辨率大小不一，根据图像情况进行统一处理
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
# 3.加载模型
model = models.resnet18(weights=None)
num_classes = 4
model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
model.load_state_dict(torch.load('models/5_weather_cls_resnet18.pth', map_location=device))
model.to(device)
model.eval()  # 设置模型为评估模式
# 4.推理应用
# 类别标签确保顺序与训练时 weather_photos 读取的顺序一致
class_names = ["cloudy", "rain", "shine", "sunrise"]
# 预测函数
def predict_image(image_path, class_names):
    image = Image.open(image_path).convert("RGB")  # 打开并转换为 RGB
    image = transform(image).unsqueeze(0)  # 添加 batch 维度
    image = image.to(device)

    with torch.no_grad():
        output = model(image)
        probabilities = F.softmax(output, dim=1)[0]  # 计算 softmax 得到概率
        predicted_class = torch.argmax(probabilities).item() 
    
    predicted_label = class_names[predicted_class]
    predicted_prob = probabilities[predicted_class].item()
    all_probs = {class_names[i]: probabilities[i].item() for i in range(len(class_names))}
    return predicted_label, predicted_prob, all_probs

# 5. Flask 路由
@app.route('/')
def index():
    return render_template('n1-dl100-6-weatherclsAPP.html')

@app.route('/predict', methods=['POST'])
def predict_image_from_request():
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded.'}), 400
    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'No selected file.'}), 400
    if file:
        image_path = os.path.join('upload', file.filename)
        file.save(image_path)
        predicted_label, predicted_prob, all_probs = predict_image(image_path, class_names)
        os.remove(image_path)
        return jsonify({
            'predicted_label': predicted_label,
            'predicted_prob': predicted_prob,
            'all_probs': all_probs
        })
    else:
        return jsonify({'error': 'File not found.'}), 400
# 6.运行 Flask 应用
if __name__ == '__main__':
    if not os.path.exists('upload'):
        os.makedirs('upload')
    app.run(debug=True,port=8080)