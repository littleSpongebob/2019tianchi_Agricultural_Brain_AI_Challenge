# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   Project Name :  code
   File Name    :  unit_to_overall
   Author       :  sjw
   Time         :  19-6-24 14:37
   Description  :  组合分割结果,拼接，加去除小区域，加乘以mask
-------------------------------------------------
   Change Activity: 
                   19-6-24 14:37
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
    parser.add_argument("--data_dir", type=str, required=True, help="the root dir of the data")
    parser.add_argument("--output_dir", type=str, required=True, help="the root dir of the output data")
    parser.add_argument("--unit_size", type=int,  default=2048, help="the size of the img after crop")
    parser.add_argument("--padding_size", type=int, default=50, help="the size of pad the original img")
    parser.add_argument("--model_path", type=str, required=True, help="the model data path")
    parser.add_argument("--area_threshold", type=int, default=150000, help="area_threshold of post process to remove")
    parser.add_argument("--image_num", type=int, required=True, help="image_num")
    return parser.parse_args()

args = get_arguments()
IMAGE = args.image_num
UNIT_SIZE = args.unit_size
PAD_SIZE = args.padding_size
AREA_THRESHOLD = args.area_threshold
MODEL_PATH = args.model_path
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

def predict(data_dir, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    fileslist = os.listdir(data_dir)
    with open(MODEL_PATH, "rb") as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        tf.import_graph_def(graph_def, name='')
    sess = tf.Session()
    input_x = sess.graph.get_tensor_by_name('ImageTensor:0')
    output = sess.graph.get_tensor_by_name('SemanticPredictions:0')
    output = tf.one_hot(output, depth=5, axis=-1)
    print('start predict {}'.format(data_dir))
    label = np.zeros((size[IMAGE][0] + 2 * PAD_SIZE, size[IMAGE][1] + 2 * PAD_SIZE, 5), dtype='uint8')
    for file in tqdm(fileslist):
        h = int(file.split('_')[1])
        w = int(file.split('_')[2].split('.')[0])
        read_path = os.path.join(data_dir, file)
        img = cv2.imread(read_path)
        b, g, r = cv2.split(img)
        img_rgb = cv2.merge([r, g, b])
        img = np.expand_dims(img_rgb, axis=0).astype(np.uint8)
        result = sess.run(output, {input_x: img})
        result = result[0].astype(np.uint8)
        # 原先是背景，烤烟，玉米，薏仁米，建筑， 转换优先级 烤烟, 薏仁米，玉米, 建筑, 背景
        result = result[..., [1, 3, 2, 4, 0]].astype(np.uint8)
        result = result[PAD_SIZE:-PAD_SIZE, PAD_SIZE:-PAD_SIZE, :]
        label[h + PAD_SIZE:h - PAD_SIZE + UNIT_SIZE, w + PAD_SIZE:w - PAD_SIZE + UNIT_SIZE, :] = label[h + PAD_SIZE:h - PAD_SIZE + UNIT_SIZE,
                                                                                 w + PAD_SIZE:w - PAD_SIZE + UNIT_SIZE,
                                                                                 :] + result
    label = label.astype(np.uint8)
    label = np.argmax(label, axis=2).astype(np.uint8)
    # 由于顺序变化转变代号
    label = np.piecewise(label, [label == 0, label == 1, label == 2, label == 3, label == 4], [1, 3, 2, 4, 0])

    if AREA_THRESHOLD != 0:
        label = to_categorical(label, num_classes=5, dtype='uint8')
        for i in range(5):
            print(i)
            label[:, :, i] = morphology.remove_small_objects(label[:, :, i] == 1, min_size=AREA_THRESHOLD, connectivity=1,
                                                             in_place=True) + 0
            label[:, :, i] = morphology.remove_small_holes(label[:, :, i] == 1, area_threshold=AREA_THRESHOLD,
                                                           connectivity=1, in_place=True) + 0
        label = np.argmax(label, axis=2).astype(np.uint8)
    else:
        print('do not postprocess')
    mask = open_big_pic('../data/image_{}_mask.png'.format(IMAGE))
    label = label[PAD_SIZE:-PAD_SIZE, PAD_SIZE:-PAD_SIZE] * mask
    label = label.astype(np.uint8)
    assert label.shape[0] == size[IMAGE][0]
    assert label.shape[1] == size[IMAGE][1]
    print(label.shape)
    cv2.imwrite(os.path.join(output_dir, 'image_{}_predict.png'.format(IMAGE)), label)
    img = io.imread('../data/image_{}_vis.png'.format(IMAGE))
    if img.shape[2] != 3:
        img = img[:, :, 0:3]
    io.imsave(os.path.join(output_dir, 'image_{}_predict_vis.png'.format(IMAGE)), vis(label=cv2.resize(label, (
    size[IMAGE][1] // 10, size[IMAGE][0] // 10), interpolation=cv2.INTER_NEAREST), img=img))


if __name__ == '__main__':
    predict(data_dir=args.data_dir, output_dir=args.output_dir)