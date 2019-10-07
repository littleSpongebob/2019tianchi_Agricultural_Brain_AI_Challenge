# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   Project Name :  code
   File Name    :  creat_vis_and_mask
   Author       :  sjw
   Time         :  19-6-23 16:30
   Description  :  生成一系列后续处理需要的图片

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
    parser.add_argument("--data_path", type=str, required=True, help="the root dir of the raw data")
    parser.add_argument("--output_path", type=str, required=True, help="the root dir of the output data")
    return parser.parse_args()

args = get_arguments()

IMAGE = args.image_num
img_path = os.path.join(args.data_path, 'image_{}.png'.format(IMAGE))
output_dir = args.output_path
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
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
        return img
    else:
        return img

if __name__ == '__main__':
    img = open_big_pic(img_path)
    
    # img_mask = np.where(np.sum(img, -1) != 0, 1, 0)
    img_mask = img[:,:,-1]//255
    img = img[:,:,:-1]
    
    img_mask_path = os.path.join(output_dir, 'image_{}_mask.png'.format(IMAGE))
    print('store mask into {}'.format(img_mask_path))
    io.imsave(img_mask_path, img_mask)
    
    mask_vis = cv2.resize(img_mask, (img_mask.shape[1] // 10, img_mask.shape[0] // 10), interpolation=cv2.INTER_NEAREST)
    mask_vis = (mask_vis * 255).astype(np.uint8)
    mask_vis_path = os.path.join(output_dir, 'image_{}_mask_vis.png'.format(IMAGE))
    print('store mask_vis into {}'.format(mask_vis_path))
    io.imsave(mask_vis_path, mask_vis)
    
    image_vis = cv2.resize(img, (img.shape[1] // 10, img.shape[0] // 10), interpolation=cv2.INTER_AREA)
    image_vis_path = os.path.join(output_dir, 'image_{}_vis.png'.format(IMAGE))
    io.imsave(mask_vis_path, mask_vis)
    print('store vis into {}'.format(image_vis_path))
    io.imsave(image_vis_path, image_vis)


