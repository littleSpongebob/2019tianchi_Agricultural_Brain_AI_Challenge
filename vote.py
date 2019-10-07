# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   Project Name :  result_vote
   File Name    :  vote
   Author       :  sjw
   Time         :  19-7-16 13:46
   Description  :  融合结果
-------------------------------------------------
   Change Activity: 
                   19-7-16 13:46
-------------------------------------------------
"""



from keras.utils import to_categorical
from cv2 import cv2
import os
import numpy as np
from skimage import morphology
from skimage import io
import argparse
from tqdm import tqdm
from PIL import Image
import gc

def get_arguments():
    parser = argparse.ArgumentParser(description="fusion")
    parser.add_argument("--one_dir", type=str, required=True, help="shi_result.")
    parser.add_argument("--two_dir", type=str, required=True, help="lin_result.")
    parser.add_argument("--three_dir", type=str, required=True, help="huang_result.")
    parser.add_argument("--vis_dir", type=str, default='./data/vis_and_mask', help="vis_dir.")
    parser.add_argument("--output_dir", type=str, required=True, help="output_dir.")
    parser.add_argument("--area_threshold", type=int, default=50000,
                        help="threshold of remove small area")

    return parser.parse_args()

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

if __name__ == '__main__':
    args = get_arguments()
    area_threshold = args.area_threshold
    huang_dir = args.one_dir
    lin_dir = args.two_dir
    shi_dir = args.three_dir
    vis_dir = args.vis_dir
    output_root_dir = args.output_dir

    png_name = ['image_5_predict.png', 'image_6_predict.png']
    data_dir = [huang_dir, lin_dir, shi_dir]

    for png in png_name:
        for i, who in enumerate(data_dir):
            temp = open_big_pic(os.path.join(who, png))
            print('{}:{}'.format(who, temp.shape))
            temp = to_categorical(temp, num_classes=5, dtype='uint8')
            if i == 0:
                result = temp
            else:
                result = result + temp
        del temp
        gc.collect()

        # 原先是背景，烤烟，玉米，薏仁米，建筑， 转换优先级 烤烟, 薏仁米，玉米, 建筑, 背景
        result = result[..., [1, 3, 2, 4, 0]].astype(np.uint8)
        # 先做一步argmax再转化为onehot是为了确保每一个像素点只有一个类别，后处理的时候可以正确的进行
        result = np.argmax(result, axis=2).astype(np.uint8)
        # 由于顺序变化转变代号
        result = np.piecewise(result, [result == 0, result == 1, result == 2, result == 3, result == 4], [1, 3, 2, 4, 0])
        if area_threshold !=0:
            result = to_categorical(result, num_classes=5, dtype='uint8')
            for i in tqdm(range(5)):
                result[:, :, i] = morphology.remove_small_objects(result[:, :, i] == 1, min_size=area_threshold, connectivity=1, in_place=True) + 0
                result[:, :, i] = morphology.remove_small_holes(result[:, :, i] == 1, area_threshold=area_threshold, connectivity=1, in_place=True) + 0
                # result[:, :, i] = cv2.morphologyEx(result[:, :, i], cv2.MORPH_OPEN, morphology.square(200))
                # result[:, :, i] = cv2.morphologyEx(result[:, :, i], cv2.MORPH_CLOSE, morphology.square(200))
            result = np.argmax(result, axis=2).astype(np.uint8)
        output_dir = os.path.join(output_root_dir)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        io.imsave(os.path.join(output_dir, '{}'.format(png)), result)
        # io.imsave(os.path.join(output_dir, '{}_vis.png'.format(png.split('.')[0])), vis(label=cv2.resize(result, (
        #     result.shape[1] // 10, result.shape[0] // 10), interpolation=cv2.INTER_NEAREST), img=io.imread(
        #     os.path.join(vis_dir, 'image_{}_vis.png'.format(png.split('_')[1])))))
