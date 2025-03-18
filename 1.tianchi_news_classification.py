#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
@File    :   1.tianchi_news_classification.py
@Time    :   2024/02/25 21:46:01
@Author  :   ljc 
@Version :   1.0
@Desc    :   data from tianchi, news data classification
Validation Accuracy:  0.9227
'''
import torch
import pandas as pd
from transformers import AutoTokenizer, BertForSequenceClassification
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset
import os
from tqdm import tqdm
from sklearn.metrics import accuracy_score
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

# 1.使用mps作为device
if torch.backends.mps.is_available():
    device = torch.device('mps')
else:
    device = torch.device('cpu')

### 2.读入数据
# 如果使用相对路径不可行，需要在vscode - 设置中 - python › Terminal: Execute In File Dir 打开
# 引入数据集，从CSV文件中读入数据
df = pd.read_csv('data/news_train_set.csv', encoding='utf-8',delimiter='\t')
# print(df.head(10))
# 解析 text 字段，将字符串转换为整数列表
df["text"] = df["text"].apply(lambda x: [int(i) for i in x.split()])
# 设定最大长度(比如取95%句子的最大长度)
MAX_LEN = min(int(df["text"].apply(len).quantile(0.95)), 512)
print(f"MAX_LEN: {MAX_LEN}")
# 进行padding，使所有的句子长度一致
sqs_i = 0
def pad_sequence(seq, max_len):
    global sqs_i
    sqs_i += 1
    if sqs_i % 10 == 0:
        print(f"处理第{sqs_i}个句子")
    # print(f"max_len: {max_len}")
    if len(seq) < max_len:
        seq += [0] * (max_len - len(seq)) # 用0填充
        return seq
    else:
        return seq[:max_len] # 超过最大长度则截断
df["text"] = df["text"].apply(lambda x: pad_sequence(x, MAX_LEN)) # 使用apply函数对每一行进行操作

# 划分训练集和验证集
train_texts, val_texts, train_labels, val_labels = train_test_split(
    df["text"].tolist(), df["label"].tolist(), test_size=0.2, random_state=42)
print(f"train_texts: {len(train_texts)}, val_texts: {len(val_texts)}")
# 3.处理tokenizer
# 使用fast tokenizer加速
tokenizer = AutoTokenizer.from_pretrained('bert-base-chinese', use_fast=True)
print(f"tokenizer: {tokenizer}")
# 分批进行tokenizer
def batch_tokenize(texts, batch_size=1000):
    encodings = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        # 🚀 关键修改：把 List[int] 转成字符串
        batch = [" ".join(map(str, seq)) for seq in batch]
        encoding = tokenizer(batch, padding="max_length", truncation=True, max_length=MAX_LEN, return_tensors="pt")
        encodings.append(encoding)
    return {
        "input_ids": torch.cat([e["input_ids"] for e in encodings]),
        "attention_mask": torch.cat([e["attention_mask"] for e in encodings])
    }
train_encodings = batch_tokenize(train_texts,batch_size=500)
val_encodings = batch_tokenize(val_texts,batch_size=500)
print(f"train_encodings, val_encodings 完成")
# 转换为 Pytroch Tensor
train_labels = torch.tensor(train_labels, dtype=torch.long)
val_labels = torch.tensor(val_labels, dtype=torch.long)
print(f"pytorch tensor 完成")
# 4.定义Dataset和DataLoader
class TextDataset(Dataset):
    def __init__(self, encodings, labels):
        self.input_ids = encodings['input_ids']
        self.attention_mask = encodings['attention_mask']
        self.labels = labels
    
    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            "input_ids": self.input_ids[idx],
            "attention_mask": self.attention_mask[idx],
            "labels": self.labels[idx]
        }

# # 加载预训练的 BERT 模型和分词器
# tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
# print(f"tokenizer: {tokenizer}")
# Tokenize 处理
# def tokenize_function(text):
#     return tokenizer(
#         text,
#         padding='max_length',
#         truncation=True,
#         max_length=MAX_LEN,
#         # return_tensors='pt'
#     )
# 对所有数据进行tokenization
# 先看一下train_texts的数据类型
# print(f"train_texts: {type(train_texts)},list_data:{type(train_texts[0])}, val_texts: {type(val_texts)}")
# train_texts = sum(train_texts, [])  # 将二维列表转换为一维列表,上一步print的是list类型，需要展开
# val_texts = sum(val_texts, [])
# print(f"train_texts: {type(train_texts)}, val_texts: {type(val_texts)}")
# # train_encodings = tokenize_function(train_texts)
# train_encodings = tokenizer(train_texts, padding=True, truncation=True, max_length=MAX_LEN, return_tensors='pt')
# print(f"train_encodings")
# # val_encodings = tokenize_function(val_texts)
# val_encodings = tokenizer(val_texts, padding=True, truncation=True, max_length=MAX_LEN, return_tensors='pt')
# print(f"val_encodings")

BATCH_SIZE = 32
train_dataset = TextDataset(train_encodings, train_labels)
val_dataset = TextDataset(val_encodings, val_labels)
print('train_loader start')
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

print(f"train_dataset: {len(train_dataset)}, val_dataset: {len(val_dataset)}, 数据转换已经完成")

# 5.定义模型
# 计算分类数量
num_labels = len(df["label"].unique())
model = BertForSequenceClassification.from_pretrained('bert-base-chinese', num_labels=num_labels)
model.to(device, dtype=torch.float32)
print(f"模型定义完成")
optimizer = optim.AdamW(model.parameters(), lr=5e-5)
# 定义损失函数
loss_fn = nn.CrossEntropyLoss()
loss_fn.to(device, dtype=torch.float32)

num_epochs = 5 # 训练5轮
for epoch in range(num_epochs):
    model.train() # 设定为训练模式
    total_loss = 0

    loop = tqdm(train_loader, leave=True)
    for batch in loop:
        optimizer.zero_grad()

        # 数据移动到GPU
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        # 前向传播
        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss

        # 反向传播
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        # 更新进度条
        loop.set_description(f"Epoch {epoch+1}/{num_epochs}")
        loop.set_postfix(loss=total_loss / len(train_loader))
    print(f"Epoch {epoch+1}, Loss: {total_loss / len(train_loader)}")

# 7.验证评估模型
model.eval() # 评估模式
all_preds = []
all_labels = []

with torch.no_grad():
    for batch in val_loader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        outputs = model(input_ids, attention_mask=attention_mask)
        preds = outputs.logits.argmax(dim=1)

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

accuracy = accuracy_score(all_labels, all_preds)
print(f"Validation Accuracy: {accuracy: .4f}")

# 保存模型
if not os.path.exists('models'):
    os.makedirs('models')
model.save_pretrained('models/news_classification_model')
tokenizer.save_pretrained('models/news_classification_model')
print("模型已经保存")
