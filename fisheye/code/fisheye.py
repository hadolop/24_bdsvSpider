# !/usr/bin/env python
# -*- coding:utf-8 -*-
# @Author  : hadolop
# @QQ      ：664110559
# @Github  : 
# @Time    : 2022/4/22 17:00
# @Function: 将360°全景图转换为鱼眼图

import cv2
import numpy as np
import math
from PIL import Image
from transformers import MaskFormerImageProcessor, MaskFormerForInstanceSegmentation
from torch import nn
import matplotlib.pyplot as plt


def ade_palette():
    """ADE20K palette that maps each class to RGB values."""
    return [[120, 120, 120], [180, 120, 120], [6, 230, 230], [80, 50, 50],
            [4, 200, 3], [120, 120, 80], [140, 140, 140], [204, 5, 255],
            [230, 230, 230], [4, 250, 7], [224, 5, 255], [235, 255, 7],
            [150, 5, 61], [120, 120, 70], [8, 255, 51], [255, 6, 82],
            [143, 255, 140], [204, 255, 4], [255, 51, 7], [204, 70, 3],
            [0, 102, 200], [61, 230, 250], [255, 6, 51], [11, 102, 255],
            [255, 7, 71], [255, 9, 224], [9, 7, 230], [220, 220, 220],
            [255, 9, 92], [112, 9, 255], [8, 255, 214], [7, 255, 224],
            [255, 184, 6], [10, 255, 71], [255, 41, 10], [7, 255, 255],
            [224, 255, 8], [102, 8, 255], [255, 61, 6], [255, 194, 7],
            [255, 122, 8], [0, 255, 20], [255, 8, 41], [255, 5, 153],
            [6, 51, 255], [235, 12, 255], [160, 150, 20], [0, 163, 255],
            [140, 140, 140], [250, 10, 15], [20, 255, 0], [31, 255, 0],
            [255, 31, 0], [255, 224, 0], [153, 255, 0], [0, 0, 255],
            [255, 71, 0], [0, 235, 255], [0, 173, 255], [31, 0, 255],
            [11, 200, 200], [255, 82, 0], [0, 255, 245], [0, 61, 255],
            [0, 255, 112], [0, 255, 133], [255, 0, 0], [255, 163, 0],
            [255, 102, 0], [194, 255, 0], [0, 143, 255], [51, 255, 0],
            [0, 82, 255], [0, 255, 41], [0, 255, 173], [10, 0, 255],
            [173, 255, 0], [0, 255, 153], [255, 92, 0], [255, 0, 255],
            [255, 0, 245], [255, 0, 102], [255, 173, 0], [255, 0, 20],
            [255, 184, 184], [0, 31, 255], [0, 255, 61], [0, 71, 255],
            [255, 0, 204], [0, 255, 194], [0, 255, 82], [0, 10, 255],
            [0, 112, 255], [51, 0, 255], [0, 194, 255], [0, 122, 255],
            [0, 255, 163], [255, 153, 0], [0, 255, 10], [255, 112, 0],
            [143, 255, 0], [82, 0, 255], [163, 255, 0], [255, 235, 0],
            [8, 184, 170], [133, 0, 255], [0, 255, 92], [184, 0, 255],
            [255, 0, 31], [0, 184, 255], [0, 214, 255], [255, 0, 112],
            [92, 255, 0], [0, 224, 255], [112, 224, 255], [70, 184, 160],
            [163, 0, 255], [153, 0, 255], [71, 255, 0], [255, 0, 163],
            [255, 204, 0], [255, 0, 143], [0, 255, 235], [133, 255, 0],
            [255, 0, 235], [245, 0, 255], [255, 0, 122], [255, 245, 0],
            [10, 190, 212], [214, 255, 0], [0, 204, 255], [20, 0, 255],
            [255, 255, 0], [0, 153, 255], [0, 41, 255], [0, 255, 204],
            [41, 0, 255], [41, 255, 0], [173, 0, 255], [0, 245, 255],
            [71, 0, 255], [122, 0, 255], [0, 255, 184], [0, 92, 255],
            [184, 255, 0], [0, 133, 255], [255, 214, 0], [25, 194, 194],
            [102, 255, 0], [92, 0, 255]]


## step1 语义分割
def pic_segment(image):
    '''
    :param image: model file and segmented image save file
    :return:  segmented image np.array()
    '''
    image = Image.open(image) # PIL.Image.open() is RGB,    cv2.imread() is BGR
    processor = MaskFormerImageProcessor.from_pretrained("../facebook/") # pretrained model save file address
    inputs = processor(images=image, return_tensors="pt")

    model = MaskFormerForInstanceSegmentation.from_pretrained("../facebook/")
    outputs = model(**inputs)

    predicted_semantic_map = processor.post_process_semantic_segmentation(outputs, target_sizes=[image.size[::-1]])[0]

    seg = predicted_semantic_map

    color_seg = np.zeros((seg.shape[0], seg.shape[1], 3), dtype=np.uint8)  # height, width, 3

    palette = np.array(ade_palette())
    for label, color in enumerate(palette):
        color_seg[seg == label, :] = color

    img = color_seg.astype(np.uint8)

    ## image + mask ，save
    img_seg = np.array(image) * 0.6 + color_seg * 0.4
    img_seg = Image.fromarray(img_seg.astype('uint8'), 'RGB')
    img_seg.save('segment.png')

    return img

def calc_feature_percent(seg_img_array):
    '''
    计算各分割要素占比
    :param seg_img_array:
    :return:
    '''
    pass



## step2  将360°全景图转换为鱼眼图
def transform(img):
    '''
    :param img: 是一张分割完的全景图
    :return: 分割后全景图转成鱼眼图
    '''
    rows,cols,c = img.shape
    R = np.int64(cols/2/math.pi)
    D = R*2
    cx = R
    cy = R
    new_img = np.zeros((D,D,c),dtype = np.uint8)
    for i in range(D):
        for j in range(D):
            r = math.sqrt((i-cx)**2+(j-cy)**2)
            if r > R:
                continue
            tan_inv = np.arctan((j-cy)/(i-cx+1e-10))
            if(i<cx):
                theta = math.pi/2+tan_inv
            else:
                theta = math.pi*3/2+tan_inv
            xp = np.int64(np.floor(theta/2/math.pi*cols))
            yp = np.int64(np.floor(r/R*rows)-1)
            new_img[j,i] = img[yp,xp]
    return new_img


## step3 计算鱼眼要素占比
def sky_percent_calc(new_img,file_n):

    # 定义需要保留的颜色
    # 这里根据语义分割label颜色确定天空的RGB值
    black = np.array([0, 0, 0])
    # blue = np.array([70,130,180]) # RGB
    sky_feature = np.array([255, 0, 163])  #  sky RGB
    build_feature = np.array([214, 255, 0]) # build
    green_feature = np.array([163, 0, 255]) # green
    # blue = np.array([180, 130, 70]) # BGR
    sky_ = np.array([255,255,255])

    # 创建掩码
    black_mask = (new_img == black).all(axis=2)
    sky_mask = (new_img == sky_feature).all(axis=2)
    build_mask = (new_img == build_feature).all(axis=2)
    green_mask = (new_img == green_feature).all(axis=2)
    # 计算天空占比
    other_mask = ~(black_mask | sky_mask)

    # 对不同掩码区域赋予不同值
    new_img[sky_mask] = sky_
    new_img[black_mask] = black
    new_img[other_mask] = [128, 128, 128]  # 灰色

    # 计算占比：sky、 build、 green
    sky_percent = round((new_img[sky_mask].shape[0] / (new_img[sky_mask].shape[0] + new_img[other_mask].shape[0]))*100,2)
    build_percent = round((new_img[build_mask].shape[0] / (new_img[sky_mask].shape[0] + new_img[other_mask].shape[0])) * 100, 2)
    green_percent = round((new_img[green_mask].shape[0] / (new_img[sky_mask].shape[0] + new_img[other_mask].shape[0])) * 100, 2)
    print(f"天空占比为：{sky_percent}%\n建筑占比为：{build_percent}%\n绿植占比为：{green_percent}%")

    # # 将处理后的数组保存为新图像
    processed_img = Image.fromarray(new_img.astype('uint8'), 'RGB')
    output_path = r"/fisheye/sky_11.png"
    processed_img.save(output_path)
    return sky_percent

# 建筑色彩
def calc_build_color(new_img,image):
    '''
    提取建筑要素，计算色彩
    :return:
    '''
    build_feature = np.array([214, 255, 0])  # build

    build_mask = (new_img == build_feature).all(axis=2)
    # masked_img = cv2.bitwise_and(img, mask)
    # 对不同掩码区域赋予不同值
    build_img = image[build_mask]
    print(type(build_img), build_img.shape)









def show_eye_percent():
    '''
    用来可视化结果
    :return:
    '''
    pass


if __name__=='__main__':

    file_n = r'/fisheye/test_0.jpg'
    image = Image.open(file_n)
    seg_img = pic_segment(file_n)
    # calc_build_color(seg_img, image)

    #
    img_ = transform(seg_img)
    img_a = Image.fromarray(img_.astype('uint8'), 'RGB')
    sky_percent = sky_percent_calc(img_,file_n)

    img_a.save('{}_fishsky_3.png'.format(file_n.split('.')[0]))



