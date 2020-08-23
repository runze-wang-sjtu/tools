# -*- coding: utf-8 -*-
# @Time    : 2020/8/13 9:18
# @Author  : runze.wang
import os
import time
import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
from reorient_reference_to_rai_general import *


def load_nifti(nifti_path):
 img1 = nib.load(nifti_path)
 data = img1.get_data()
 affine = img1.affine
 data = np.asarray(data)
 return data, img1

def num_one(source_array):
    count = 0
    for x in source_array:
        if x == 1:
            count += 1
    return count

def plot_without_margin(arr, nii_name, i, implant_name=None,is_label=True):
    fig, ax = plt.subplots()
    ax.imshow(arr, cmap='gray',aspect='equal')
    plt.axis('off')
    # 去除图像周围的白边
    height, width = arr.shape
    # 如果dpi=300，那么图像大小=height*width
    fig.set_size_inches(width / 100.0 / 3.0, height / 100.0 / 3.0)
    plt.gca().xaxis.set_major_locator(plt.NullLocator())
    plt.gca().yaxis.set_major_locator(plt.NullLocator())
    plt.subplots_adjust(top=1, bottom=0, left=0, right=1, hspace=0, wspace=0)
    plt.margins(0, 0)

    # dpi是设置清晰度的，大于300就很清晰了，但是保存下来的图片很大
    if is_label:
        plt.savefig(os.path.join(target_path,'labels','{}_level_{}.png'.format(
            nii_name.split('.nii')[0],i)), dpi=300)
    else:
        plt.savefig(os.path.join(target_path,'images','{}_{}_level_{}.png'.format(
            nii_name.split('.nii')[0],implant_name.split('implant_')[-1].split('.nii')[0],i)), dpi=300)

if __name__ == '__main__':

    source_path = '/home/gdp/data/spine_large/VerSeg_complete/VerSeg'
    target_path = '/home/gdp/data/spine_simulation_2D'

    image_names = os.listdir(os.path.join(source_path, 'image'))
    label_names = os.listdir(os.path.join(source_path, 'label'))

    num = 0
    for image_name in image_names:
        implant_names = []
        for label_name in label_names:
            if 'implant' in label_name and label_name.split('_')[0] == image_name.split('.')[0]:
                implant_names.append(label_name)
        for implant_name in implant_names:
            image = itk.imread((os.path.join(source_path, 'image', image_name)))
            reoriented = reorient_to_rai(image)
            m = itk.GetMatrixFromArray(np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]], np.float64))
            reoriented.SetDirection(m)
            reoriented.Update()
            image_data_array = itk.GetArrayFromImage(reoriented)
            image_data_image = sitk.GetImageFromArray(image_data_array)
            image_data_resample = resample(image_data_image,out_spacing=(1,1,1))
            image_data = sitk.GetArrayFromImage(image_data_resample)


            implant = itk.imread(os.path.join(source_path, 'label', implant_name))
            implant.SetDirection(image.GetDirection())  ####!!!!!!!!!!
            implant.SetSpacing(image.GetSpacing())  ####!!!!!!!!!!!!
            reoriented_implant = reorient_to_rai(implant)
            n = itk.GetMatrixFromArray(np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]], np.float64))
            reoriented_implant.SetDirection(n)
            reoriented_implant.Update()
            implant_data_array = itk.GetArrayFromImage(reoriented_implant)
            implant_data_image = sitk.GetImageFromArray(implant_data_array)
            implant_data_resample = resample(implant_data_image,out_spacing=(1,1,1),is_label=True)
            implant_data = sitk.GetArrayFromImage(implant_data_resample)


            implant_voxel = np.where(implant_data==1.)
            range_z = np.max(implant_voxel[0]) - np.min(implant_voxel[0])
            for z in range(int(0.2*range_z), int(0.8*range_z)):
                num += 1
                plot_without_margin(implant_data[(np.min(implant_voxel[0])+z), :, :], implant_name, z)
                plot_without_margin(image_data[(np.min(implant_voxel[0])+z), :, :], image_name, z, implant_name,is_label=False)
                print('Time:{}==>'.format(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())),
                      'Number:{}==>'.format(num),
                      os.path.join(target_path, 'images', '{}_{}_level_{}.png'.format(
                          image_name.split('.nii')[0], implant_name.split('implant_')[-1].split('.nii')[0], z)))
