# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   Project Name :  code
   File Name    :  creat_test_unit
   Author       :  sjw
   Time         :  19-6-23 16:30
   Description  :  将大图分割为小图，并且进行分割后组合回大图，
                   小图的文件名按照当前图片在大图中的位置进行
                   命名，格式如：IMAGE_h_w.png(h,w)为当前小
                   图在大图中的左上点的坐标)

-------------------------------------------------
   Change Activity: 
                   19-6-23 16:30
-------------------------------------------------
"""
import os
import numpy as np
import cv2
from skimage import io
from PIL import Image
import argparse
from multiprocessing import Pool

def get_arguments():
    parser = argparse.ArgumentParser(description="process")
    parser.add_argument("--image_num", type=int, required=True, help="image_{}.png")
    parser.add_argument("--unit", type=int, default=2048, help="the size of the unit img")
    parser.add_argument("--step", type=int, default=1024, help="the size of step")
    parser.add_argument("--pad", type=int, default=100, help="the size of pad")
    parser.add_argument("--data_path", type=str, required=True, help="the root dir of the raw data")
    parser.add_argument("--output_path", type=str, required=True, help="the root dir of the output data")
    return parser.parse_args()

args = get_arguments()

IMAGE = args.image_num
UNIT = args.unit
STEP = args.step
PAD = args.pad
img_path = os.path.join(args.data_path, 'image_{}.png'.format(IMAGE))
output_dir = os.path.join(args.output_path, 'image_{}_unit{}_step{}_pad{}'.format(IMAGE, UNIT, STEP, PAD))


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

if __name__ == '__main__':
    # npy_path = '/home/sjw/Desktop/县域农业大脑AI挑战赛/train/image_{}.npy'.format(IMAGE)
    img = open_big_pic(img_path)
    img = img.astype(np.uint8)
    h, w, _ = img.shape
    k = 0
    img_resize = np.pad(img, ((PAD, PAD), (PAD, PAD), (0, 0)), mode='constant', constant_values=0)
    h, w, _ = img_resize.shape
    assert img.shape[2] == 3

    h_index = np.arange(0, h - UNIT, STEP)
    h_index = np.append(h_index, h - UNIT)
    w_index = np.arange(0, w - UNIT, STEP)
    w_index = np.append(w_index, w - UNIT)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    def crop(i,j):
        img_unit = img_resize[i:i + UNIT, j:j + UNIT, :]
        if np.sum(img_unit) != 0:
            path = os.path.join(output_dir, '{}_{}_{}.png'.format(IMAGE, i, j))
            io.imsave(path, img_unit)
    # for i in h_index:
    #     for j in w_index:
    #         img_unit = img_resize[i:i + UNIT, j:j + UNIT, :]
    #         if np.sum(img_unit) != 0:
    #             path = os.path.join(output_dir, '{}_{}_{}.png'.format(IMAGE, i, j))
    #             io.imsave(path, img_unit)
    #             k = k + 1
    #             print(k, i, j)
    P = Pool(8)
    for i in h_index:
        for j in w_index:
            P.apply_async(crop, (i, j))
    P.close()
    P.join()




