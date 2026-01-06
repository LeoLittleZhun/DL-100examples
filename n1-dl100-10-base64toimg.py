#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
@File    :   n1-dl100-10-base64toimg.py
@Time    :   2026/01/06 14:59:18
@Author  :   ljc 
@Version :   1.0
@Desc    :   base64编码字符串转图片
'''

# here put the import lib
import base64
import PIL.Image as img

def base64_to_image(base64_string: str, output_image_path:str) -> None:
    """
    base64_to_image 的 将base64编码字符串转换为图片并保存到指定路径
    
    :param base64_string: base64编码字符串
    :type base64_string: str
    :param output_image_path: 图片保存路径
    :type output_image_path: str
    """
    base64_bytes = base64_string.encode('utf-8')
    image_bytes = base64.b64decode(base64_bytes)
    with open(output_image_path, mode="wb") as img_file:
        img_file.write(image_bytes)

if __name__ == "__main__":
    # 读取base64编码字符串
    with open("data/base64_output.txt", "r", encoding='utf-8') as text_file:
        base64_str = text_file.read()
    # 转换为图片并保存
    output_image_path = "data/decoded_image.jpg"
    base64_to_image(base64_str, output_image_path)
    print(f"图片已保存到 {output_image_path}")
    img.open(output_image_path).show()
