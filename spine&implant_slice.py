# -*- coding: utf-8 -*-
# @Author  : runze.wang
# @Time    : 2020/8/24 12:49 PM

import os
import cv2
import numpy as np
import SimpleITK as sitk
from implant_segmentation import *

def dilated(x, iterations=None):

    'x: np.array'
    kernel = np.ones((5,5), np.uint8)
    erosion = cv2.dilate(x, kernel, iterations=iterations)

    return erosion

if __name__ == '__main__':

    path = '/Users/runze.wang/Desktop'
    img_sitk = sitk.ReadImage(os.path.join(path, '2805012.nii.gz'))
    img_array = sitk.GetArrayFromImage(img_sitk)
    img_array = img_array - np.min(img_array)
    img = img_array / np.max(img_array) * 255
    implant = implant_segmentation(img_sitk, threshold=2000)
    implant[implant==1] = 255

    # ### check in 2D
    # plt.imshow(implant[57, :, :], cmap='gray')
    # plt.show()

    implant_zrange = np.unique(np.where(implant)[0])
    for i in implant_zrange:
        
        cv2.imwrite(os.path.join(path,'mask','implant_{}.png').format(i), dilated(implant[i,:,:],iterations=3))
        cv2.imwrite(os.path.join(path,'image','image_{}.png').format(i), img[i,:,:])
        print('{} has done'.format(os.path.join(path,'image','image_{}.png').format(i)))


    print('Done')


