#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
@File    :   n1-dl100-9-img2base64.py
@Time    :   2026/01/06 14:12:15
@Author  :   ljc
@Version :   1.0
@Desc    :   图片转base64编码
'''

# here put the import lib
import base64
def image_to_base64(image_path:str) -> str:
    """
    将图片转换为base64编码字符串
    :param image_path string 图片文件路径
    :return string base64编码字符串
    """
    with open(image_path, "rb") as image_file:
        image_bytes = image_file.read() # file -> bytes
        base64_bytes = base64.b64encode(image_bytes) # bytes encode base64_bytes
        base64_string = base64_bytes.decode('utf-8')
        return base64_string
    
if __name__ == "__main__":
    image_path = "data/cat-test.jpg"
    base64_str = image_to_base64(image_path)
    with open("data/base64_output.txt", "w", encoding='utf-8') as text_file:
        text_file.write(base64_str)
        print("Base64编码已保存到 data/base64_output.txt")
