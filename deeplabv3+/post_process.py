# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   Project Name :  deeplabv3+
   File Name    :  post_process
   Author       :  sjw
   Time         :  19-8-1 19:23
   Description  :  后处理label
-------------------------------------------------
   Change Activity: 
                   19-8-1 19:23
-------------------------------------------------
"""
from skimage import io
import numpy as np
import os
from tqdm import tqdm
import cv2
from skimage import morphology
from keras.utils import to_categorical
import argparse
import cv2
import tensorflow as tf
from PIL import Image

def get_arguments():
    parser = argparse.ArgumentParser(description="process")
    parser.add_argument("--model_name", type=str, required=True, help="the root dir of the data")
    parser.add_argument("--step", type=str, required=True, help="the root dir of the output data")
    parser.add_argument("--area_threshold", type=int,  default=150000, help="area_threshold")
    return parser.parse_args()


args = get_arguments()
MODEL_NAME = args.model_name
STEP = args.step
AREA_THRESHOLD = args.area_threshold
size = [[], [], [], [20767, 42614], [29003, 35055]]

def vis(label, img=None, alpha=0.5):
    '''
    :param label:原始标签
    :param img: 原始图像
    :param alpha: 透明度
    :return: 可视化标签
    '''
    r = np.where(label == 1, 255, 0)
    g = np.where(label == 2, 255, 0)
    b = np.where(label == 3, 255, 0)
    white = np.where(label == 4, 255, 0)
    anno_vis = np.dstack((r, g, b)).astype(np.uint8)
    # 白色分量(红255, 绿255, 蓝255)
    anno_vis = anno_vis + np.expand_dims(white, axis=2)
    if img is None:
        return anno_vis
    else:
        overlapping = cv2.addWeighted(img.astype(np.uint8), alpha, anno_vis.astype(np.uint8), 1-alpha, 0)
        return overlapping


def open_big_pic(path):
    '''
    :param path: 图像路径
    :return: 图像numpy数组
    '''
    Image.MAX_IMAGE_PIXELS = 100000000000
    print('open{}'.format(path))
    img = Image.open(path)   # 注意修改img路径
    img = np.asarray(img)
    print('img_shape:{}'.format(img.shape))
    if len(img.shape) == 3:
        if img.shape == 4:
            return img[:, :, :-1]
        else:
            return img
    else:
        return img


def post_process(src_label,image_num, output_dir):
    src_label = src_label.astype(np.uint8)
    src_label = to_categorical(src_label, num_classes=5, dtype='uint8')
    # 原先是背景，烤烟，玉米，薏仁米，建筑， 转换优先级 烤烟, 薏仁米，玉米, 建筑, 背景
    # src_label = src_label[..., [1, 3, 2, 4, 0]].astype(np.uint8)
    if AREA_THRESHOLD != 0:
        for i in [0, 1, 2, 3, 4]:
            print(i)
            src_label[:, :, i] = morphology.remove_small_objects(src_label[:, :, i] == 1, min_size=AREA_THRESHOLD,
                                                                 connectivity=1,
                                                                 in_place=True) + 0
            src_label[:, :, i] = morphology.remove_small_holes(src_label[:, :, i] == 1, area_threshold=AREA_THRESHOLD,
                                                               connectivity=1, in_place=True) + 0
    else:
        print('do not postprocess')

    src_label = np.argmax(src_label, axis=2).astype(np.uint8)
    # src_label = np.piecewise(src_label, [src_label == 0, src_label == 1, src_label == 2, src_label == 3, src_label == 4], [1, 3, 2, 4, 0])
    mask = open_big_pic('../data/image_{}_mask.png'.format(image_num))
    src_label = src_label * mask
    src_label = src_label.astype(np.uint8)
    assert src_label.shape[0] == size[image_num][0]
    assert src_label.shape[1] == size[image_num][1]
    print(src_label.shape)
    cv2.imwrite(os.path.join(output_dir, 'image_{}_predict.png'.format(image_num)), src_label)
    img = io.imread('../data/image_{}_vis.png'.format(image_num))
    if img.shape[2] != 3:
        img = img[:, :, 0:3]
    io.imsave(os.path.join(output_dir, 'image_{}_predict_vis.png'.format(image_num)), vis(label=cv2.resize(src_label, (
        size[image_num][1] // 10, size[image_num][0] // 10), interpolation=cv2.INTER_NEAREST), img=img))


if __name__ == '__main__':
    output_dir = os.path.join('.', 'result', MODEL_NAME, 'step_{}'.format(STEP), 'post_{}'.format(AREA_THRESHOLD))
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    for image_num in [3,4]:
        src_label_path = os.path.join('.', 'result', MODEL_NAME, 'step_{}'.format(STEP), 'image_{}_predict.png'.format(image_num))
        src_label = open_big_pic(src_label_path)
        post_process(src_label=src_label, image_num=image_num, output_dir=output_dir)
