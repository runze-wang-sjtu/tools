# -*- coding: utf-8 -*-
# @Author  : runze.wang
# @Time    : 2020/8/29 10:58 AM
import os
import random
import numpy as np
import nibabel as nib
import nibabel.processing
import SimpleITK as sitk
from data_augmentation import *
from reorient_reference_to_rai_general import resample

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

def padding_num(a, b):

    assert a >= b
    if (a - b) % 2 == 0:
        pad_before = (a-b)/2
        pad_after = (a-b)/2
    else:
        pad_before = (a-b+1)/2
        pad_after = (a-b-1)/2

    return int(pad_before), int(pad_after)

def padding(image_data, implant_data):

    if image_data.shape[0] >= implant_data.shape[0]:
        pad_before, pad_after = padding_num(image_data.shape[0], implant_data.shape[0])
        implant_data = np.pad(implant_data, ((pad_before, pad_after),(0,0),(0,0)), 'minimum')
    else:
        pad_before, pad_after = padding_num(implant_data.shape[0], image_data.shape[0])
        image_data = np.pad(image_data, ((pad_before, pad_after),(0,0),(0,0)), 'minimum')
    if image_data.shape[1] >= implant_data.shape[1]:
        pad_before, pad_after = padding_num(image_data.shape[1], implant_data.shape[1])
        implant_data = np.pad(implant_data, ((0,0),(pad_before, pad_after),(0,0)), 'minimum')
    else:
        pad_before, pad_after = padding_num(implant_data.shape[1], image_data.shape[1])
        image_data = np.pad(image_data, ((0,0),(pad_before, pad_after),(0,0)), 'minimum')
    if image_data.shape[2] >= implant_data.shape[2]:
        pad_before, pad_after = padding_num(image_data.shape[2], implant_data.shape[2])
        implant_data = np.pad(implant_data, ((0,0),(0,0),(pad_before, pad_after)), 'minimum')
    else:
        pad_before, pad_after = padding_num(implant_data.shape[2], image_data.shape[2])
        image_data = np.pad(image_data, ((0,0),(0,0),(pad_before, pad_after)), 'minimum')

    return image_data, implant_data

def main():
    pass

if __name__ == '__main__':

    path = '/data/rz/codes/tools/data'

    image = sitk.ReadImage(os.path.join(path, 'verse537.nii.gz'))
    image_data = sitk.GetArrayFromImage(image)

    mask = sitk.ReadImage(os.path.join(path, 'verse537_seg.nii.gz'))
    mask_data = sitk.GetArrayFromImage(mask)

    implant = sitk.ReadImage(os.path.join(path, 'implant.nii.gz'))
    implant_resample = resample(implant, out_spacing=image.GetSpacing(), is_label=True)
    implant_resample_data = sitk.GetArrayFromImage(implant_resample)


    image_padding, implant_padding = padding(image_data, implant_resample_data)
    image_padding = sitk.GetImageFromArray(image_padding)
    image_padding.SetSpacing(image.GetSpacing())
    implant_exdim = np.expand_dims(np.expand_dims(implant_padding, axis=0), axis=0)
    implant_trans = affine_transformation(vol=implant_exdim,
                                          radius=(0, 0, 0),
                                          translate=(50, 0, -20),
                                          scale=(1, 1, 1),
                                          bspline_order=0,
                                          border_mode="nearest",
                                          constant_val=0,
                                          is_reverse='False')
    implant_trans = np.squeeze(implant_trans)
    implant_trans = sitk.GetImageFromArray(implant_trans)
    implant_trans.SetSpacing(image.GetSpacing())

    # sitk.WriteImage(image_padding, os.path.join(path, 'image_padding.nii.gz'))
    sitk.WriteImage(implant_trans, os.path.join(path, 'implant_trans.nii.gz'))

    print('Done')
    
    main()