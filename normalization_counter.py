# -*- coding: utf-8 -*-
# @Time    : 2020/8/14 18:40
# @Author  : runze.wang

import os
import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

def normalize(image):

    img = Image.open(image).convert('L').resize((256,256))
    arr = np.array(img) / 255
    arr_norm = (arr - arr.mean()) / arr.std()
    # arr_norm = arr - arr.mean()
    return arr_norm

if __name__ == '__main__':
    # train_flist = '/home/gdp/codes/GatedConvolution_pytorch/data/spine_simulation_2D/image_flist_for_train'
    # with open(train_flist,'r') as f:
    #     images_list = f.read().splitlines()
    #
    # img_0 = Image.open(images_list[0]).convert('L').resize((256,256))
    # result = np.array(img_0)
    # for i in range(1,int(0.1*len(images_list))):
    #     img = Image.open(images_list[i]).convert('L').resize((256,256))
    #     img_arr = np.array(img)
    #     result = np.dstack((result, img_arr))
    #     print('loop:{},name:{},result_shape:{}'.format(i,images_list[i],result.shape))
    #
    # print('Mean:{}, Std:{}'.
    #       format(result.mean()/255,result.std()/255))

#######single test
    img_path = '/home/gdp/codes/GatedConvolution_pytorch/result_logs/202008152223_spine_norm/val_0_whole/real/0.png'
    img0 = Image.open(img_path).convert('L')
    arr0 = np.array(img0)
    print(arr0.mean())
    # cv2.imwrite('./img_0.png',np.array(img0))
    img1 = normalize(img_path)
    # cv2.imwrite('./img_1.png',img1)
    img2 = Image.open(img_path).convert('L')
    img2 = np.array(img2) / 255
    # img2 = img2 - 0.265
    img2 = (img2 - 0.265) / 0.196
    # cv2.imwrite('./img_2.png',img2)

