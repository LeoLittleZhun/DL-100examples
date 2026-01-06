#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
@File    :   test.py
@Time    :   2025/08/11 10:36:15
@Author  :   ljc 
@Version :   1.0
@Desc    :   None
'''

# here put the import lib

# 获取输入数据（假设 input 是一个字典）
input_data = {
    "left_DS": input.get("left_DS"),
    "right_DS": input.get("right_DS"),
    "left_DC": input.get("left_DC"),
    "right_DC": input.get("right_DC")
}

# 转换为浮点数，处理空值
try:
    left_DS = float(input_data["left_DS"]) if input_data["left_DS"] not in [None, ''] else 0
    right_DS = float(input_data["right_DS"]) if input_data["right_DS"] not in [None, ''] else 0
    left_DC = float(input_data["left_DC"]) if input_data["left_DC"] not in [None, ''] else 0
    right_DC = float(input_data["right_DC"]) if input_data["right_DC"] not in [None, ''] else 0
except (ValueError, KeyError):
    output = {'output': "输入数据异常"}
else:
    # 计算球镜等效值（SE）
    if left_DC == 0 or right_DC == 0:
        output = {'output': ""}
    else:
        left_SE = left_DS + (left_DC / 2)
        right_SE = right_DS + (right_DC / 2)

        # 判断近视情况（修正了原逻辑错误：右眼应该是 >= -0.50）
        if left_SE >= -0.50 and right_SE >= -0.50:
            result = "不近视"
        else:
            result = "近视"

    output = {'output': result}