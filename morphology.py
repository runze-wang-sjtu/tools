#!/usr/bin/env python
# encoding: utf-8
'''
@author: runze.wang
@time: 2020-09-08 18:18
'''

import os
import cv2
import argparse
import numpy as np
import SimpleITK as sitk
from resample import *

def close(img_sitk, kernel_size):

    img_arr = sitk.GetArrayFromImage(img_sitk)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, kernel_size)
    closing = cv2.morphologyEx(img_arr, cv2.MORPH_CLOSE, kernel)
    out_sitk = sitk.GetImageFromArray(closing)
    out = reference(out_sitk, img_sitk)

    return out

def dilated(img_sitk, kernel_size=[3,3], iterations=1):

    img_arr = sitk.GetArrayFromImage(img_sitk)
    kernel = np.ones(kernel_size, np.uint8)
    erosion = cv2.dilate(img_arr, kernel, iterations=iterations)
    out_sitk = sitk.GetImageFromArray(erosion)
    out = reference(out_sitk, img_sitk)

    return out

Threshold = 1500
Filename = '4634640'

parser = argparse.ArgumentParser()
parser.add_argument('--image_in', default='/Users/runze.wang/PycharmProjects/generative_inpainting/test/reality/image/{}.nii.gz'.format(Filename),
                    type=str,help='The filename of image to be loaded.')
parser.add_argument('--image_out', default='/Users/runze.wang/PycharmProjects/generative_inpainting/test/reality/mask/{}.nii.gz'.format(Filename),
                    type=str,help='The filename of mask to be writed.')

if __name__ == '__main__':
    args, unknown = parser.parse_known_args()
    img_sitk = sitk.ReadImage(args.image_in)
    img_arr = sitk.GetArrayFromImage(img_sitk)
    img_arr[img_arr < Threshold] = 0
    img_arr[img_arr >= Threshold] = 1
    # mask_arr = dilated(img_arr, iterations=1)
    mask_sitk = sitk.GetImageFromArray(img_arr)
    mask_sitk = reference(mask_sitk, reference_sitk=img_sitk)
    sitk.WriteImage(mask_sitk, args.image_out)

    print('Done')