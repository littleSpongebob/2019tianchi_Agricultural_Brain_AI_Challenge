# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   Project Name :  code
   File Name    :  count_weight
   Author       :  sjw
   Time         :  19-6-25 12:55
   Description  :  检测每个类别的实例所占的比例
-------------------------------------------------
   Change Activity:
                   19-6-25 12:55
-------------------------------------------------
"""
import os
import numpy as np
from tqdm import tqdm
import cv2


data_dir = '/home/sjw/Desktop/县域农业大脑AI挑战赛/git/models-master/research/deeplab/datasets/dataset/SegmentationClass'
file_list = os.listdir(data_dir)

total_zero = 0
total_one = 0
total_two = 0
total_three = 0
for label_path in tqdm(file_list):
    label = cv2.imread(os.path.join(data_dir, label_path), 0)
    zero = np.sum(np.where(label == 0, 1, 0))
    one = np.sum(np.where(label == 1, 1, 0))
    two = np.sum(np.where(label == 2, 1, 0))
    three = np.sum(np.where(label == 3, 1, 0))
    if zero + one + two + three == 1048576:
        total_zero = total_zero + zero
        total_one = total_one + one
        total_two = total_two + two
        total_three = total_three + three
    else:
        print('error in {}'.format(label_path.split('/')[-1]))
total = total_zero + total_one + total_two + total_three
print('zero {}'.format(total_zero/total))
print('one {}'.format(total_one/total))
print('two {}'.format(total_two/total))
print('three {}'.format(total_three/total))




