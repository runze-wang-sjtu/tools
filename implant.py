# -*- coding: utf-8 -*-
# @Author  : runze.wang
# @Time    : 2020/8/29 10:58 AM
import os
import random
import numpy as np
import SimpleITK as sitk
# from data_augmentation import *

def implant_random_trasform(implant, trans_value):

    implant_exdim = np.expand_dims(np.expand_dims(implant, axis=0), axis=0)
    trans_implant = affine_transformation(vol=implant_exdim,
                                          radius=(random.uniform(0.1,0.9)*np.pi,
                                                  random.uniform(0.1,0.9)*np.pi,
                                                  random.uniform(0.1,0.9)*np.pi),
                                          translate=trans_value,
                                          scale=(1, 1, 1),
                                          bspline_order=0,
                                          border_mode="nearest",
                                          constant_val=0,
                                          is_reverse='False')
    implant = np.squeeze(trans_implant)
    return implant

def resample(input_image, reference_image, is_label=False):
    #   reference_image, target_image ---> image_sitk

    resample = sitk.ResampleImageFilter()
    resample.SetOutputDirection(reference_image.GetDirection())
    resample.SetOutputOrigin(reference_image.GetOrigin())
    resample.SetOutputSpacing(reference_image.GetSpacing())
    resample.SetSize(reference_image.GetSize())

    if is_label:
        resample.SetInterpolator(sitk.sitkNearestNeighbor)
    else:
        resample.SetInterpolator(sitk.sitkLinear)

    out = resample.Execute(input_image)

    return out


def main():
    pass

if __name__ == '__main__':

    implant_path = '/Users/runze.wang/PycharmProjects/generative_inpainting/test/spine_nii/output/2805012_implant.nii.gz'
    mask_path = '/Users/runze.wang/Desktop/verse537_seg.nii.gz'

    mask = sitk.ReadImage(mask_path)
    mask_data = sitk.GetArrayFromImage(mask)
    implant = sitk.ReadImage(implant_path)
    implant_data = sitk.GetArrayFromImage(implant)

    implant = resample(implant, mask, is_label=True)
    
    vertebra_mask_value = [20., 21., 22., 23., 24., 25.]
    for index in np.unique(mask_data):
        if index in vertebra_mask_value:
            _, x_list, y_list, z_list = np.where([mask_data == index])  ###vertebra mask location
            for num in range(1,1):
                location_index = random.randint(0, len(x_list))
                target_location = np.array([int(x_list[location_index]),
                                            int(y_list[location_index]),
                                            int(z_list[location_index])])
                source_location = np.array([int(0.5 * seg_data.shape[0]),
                                            int(0.5 * seg_data.shape[1]),
                                            int(0.5 * seg_data.shape[2])])
                translated_distance = target_location - source_location

                implant_trans = implant_random_trasform(implant_data,
                                                        trans_value=(translated_distance[0],
                                                                     translated_distance[1],
                                                                     translated_distance[2]))
                save(implant_trans, os.path.join(path, '{}_implant_vertebra_{}_num_{}.nii.gz'.
                                                 format(file.split('.')[0], int(index), num)))
                print('Time:{}====>'.format(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())),
                      os.path.join(path, '{}_implant_vertebra_{}_num_{}.nii.gz'.
                                   format(file.split('.')[0], int(index), num)))

    main()