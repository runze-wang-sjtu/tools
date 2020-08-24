# -*- coding: utf-8 -*-
# @Author  : runze.wang
# @Time    : 2020/8/23 6:36 PM
import os
import SimpleITK as sitk
import matplotlib.pyplot as plt

def seg_3D_16bit(array, threshold):

    array_copy = array.copy()
    array_copy[array_copy<threshold] = 0
    array_copy[array_copy>=threshold] = 1

    return array_copy

def implant_segmentation(img_sitk, threshold):

    img_array = sitk.GetArrayFromImage(img_sitk)
    implant_16 = seg_3D_16bit(img_array,threshold=threshold)

    return implant_16

if __name__ == '__main__':
    path = '/Users/runze.wang/Desktop'

    img_sitk = sitk.ReadImage(os.path.join(path, '2805012.nii.gz'))
    img_array = sitk.GetArrayFromImage(img_sitk)
    implant_16 = seg_3D_16bit(img_array,threshold=2500)
    implant_16_sitk = sitk.GetImageFromArray(implant_16)

    #### check in 2D
    # implant_arr = sitk.GetArrayFromImage(implant_16_sitk)
    # implant_arr[implant_arr==1] = 255
    # plt.imshow(implant_arr[57, :, :], cmap='gray')
    # plt.show()

    sitk.WriteImage(implant_16_sitk, os.path.join(path, 'implant.nii.gz'))


    print('Done')

