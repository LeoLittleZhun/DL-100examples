#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
@File    :   0.py
@Time    :   2024/02/23 10:52:38
@Author  :   ljc 
@Version :   1.0
@Desc    :   torch bert mps device test, for getting two text's similarity
'''

import torch
from transformers import BertTokenizer, BertModel
import torch.nn as nn
import torch.nn.functional as F #导入激活函数库，activate function
import time

# 使用mps作为device
device = torch.device('mps')

# 加载预训练的 BERT 模型和分词器
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
bert_model = BertModel.from_pretrained('bert-base-uncased')

# 定义一个简单的文本相似性模型
class TextSimilarityModel(nn.Module):
    def __init__(self, bert_model):
        super(TextSimilarityModel, self).__init__()
        self.bert = bert_model
        self.fc = nn.Linear(768, 1)

    def forward(self, input_ids_A, input_ids_B):
        outputs_A = self.bert(input_ids_A)[1].to(device) # 取 BERT 的 CLS token 输出
        outputs_B = self.bert(input_ids_B)[1].to(device)
        # similarity_score = F.relu(torch.cosine_similarity(outputs_A, outputs_B, dim=1).unsqueeze(1))
        similarity_score = torch.cosine_similarity(outputs_A, outputs_B, dim=1).unsqueeze(1) # f(x)=max(0,x) relu函数，小于0时输出0，大于0时输出原值

        #.unsqueeze(1): 余弦相似度计算的结果是一个标量值，但是我们想要保持结果的维度与输入相同，所以我们使用 .unsqueeze(1) 将结果的维度扩展一维，使得结果变成一个列向量。
        return similarity_score

# a = "i've eat the breakfast"
# b = "i've eat dinner"
a = "我们是好朋友"
b = "camel is an animal"

model = TextSimilarityModel(bert_model)

model.to(device)

input_ids_A = tokenizer.encode(a, return_tensors='pt').to(device) # 使用 BERT 分词器（tokenizer）将文本 a 编码为模型所需的输入张量，并将结果以 PyTorch 张量的形式返回
input_ids_B = tokenizer.encode(b, return_tensors='pt').to(device) # to将张量从CPU移动到GPU中
time1 = time.time()
score = model(input_ids_A, input_ids_B).item()
time2 = time.time()
process_time = time2 - time1
print(f"a:{a}, b:{b}, similarity score:{score}, time:{process_time}")