# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   Project Name :  tools
   File Name    :  rawdata_crop
   Author       :  sjw
   Time         :  19-7-29 15:04
   Description  :  剪切原始图片
-------------------------------------------------
   Change Activity: 
                   19-7-29 15:04
-------------------------------------------------
"""
import numpy as np
from skimage import io
import cv2
import os
from PIL import Image
import argparse

def get_arguments():
    parser = argparse.ArgumentParser(description="crop")
    parser.add_argument("--unit", type=int, default=1024)
    parser.add_argument("--image-num", type=int, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--image_path", type=str, required=True)
    parser.add_argument("--label_path", type=str, required=True)
    return parser.parse_args()

args = get_arguments()

UNIT = args.unit
IMAGE = args.image_num
OUTPUT_DIR = args.output_dir
IMAGE_PATH = args.image_path
LABEL_PATH = args.label_path

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
        return img[:, :, :-1]
    else:
        return img


def crop_img_label(img, label, output_dir):
    if not os.path.exists(os.path.join(output_dir, 'JPEGImages')):
        print('output dir not exists make {} dir'.format(output_dir))
        os.makedirs(os.path.join(output_dir, 'JPEGImages'))
    if not os.path.exists(os.path.join(output_dir, 'SegmentationClass')):
        print('output dir not exists make {} dir'.format(output_dir))
        os.makedirs(os.path.join(output_dir, 'SegmentationClass'))
    # if not os.path.exists(os.path.join(output_dir, 'vis')):
    #     print('output dir not exists make {} dir'.format(output_dir))
    #     os.makedirs(os.path.join(output_dir, 'vis'))

    label_h, label_w = label.shape
    img_h, img_w, _ = img.shape
    assert label_h == img_h
    assert label_w == img_w
    h_index = 0
    k = 0
    while h_index < label_h - UNIT:
        w_index = 0
        while w_index < label_w - UNIT:
            img_unit = img[h_index:h_index + UNIT, w_index:w_index + UNIT, :]
            # 删除黑色大于7/8的图片
            if np.sum(np.where(np.sum(img_unit, axis=2) != 0, 1, 0)) > UNIT*UNIT//8:
                k = k + 1
                print('\rcrop {} unit image'.format(k), end='', flush=True)
                label_unit = label[h_index:h_index + UNIT, w_index:w_index + UNIT]
                path_unit_img = os.path.join(output_dir, 'JPEGImages', '{}_{}.png'.format(IMAGE, k))
                path_unit_label = os.path.join(output_dir, 'SegmentationClass', '{}_{}.png'.format(IMAGE, k))
                # path_unit_vis = os.path.join(output_dir, 'vis', '{}_{}.png'.format(IMAGE, k))
                io.imsave(path_unit_img, img_unit)
                io.imsave(path_unit_label, label_unit)
                # io.imsave(path_unit_vis, vis(label=label_unit, img=img_unit))
                # 如果0类个数小于1/3，则减小步长
                if np.sum(np.where(label_unit == 0, 1, 0)) < UNIT*UNIT//3:
                    w_index = w_index + UNIT//2
                else:
                    w_index = w_index + UNIT // 10 * 9
            else:
                w_index = w_index + UNIT // 10 * 9
        h_index = h_index + UNIT // 10 * 9


if __name__ == '__main__':
    img_path = IMAGE_PATH
    label_path = LABEL_PATH
    img = open_big_pic(img_path)
    label = open_big_pic(label_path)
    crop_img_label(img, label, OUTPUT_DIR)

