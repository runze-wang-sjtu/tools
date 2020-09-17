#!/usr/bin/env python
# encoding: utf-8
'''
@author: runze.wang
@time: 2020-09-08 14:29
'''

import os
import argparse
import numpy as np
import SimpleITK as sitk

def size(img_sitk, out_size, is_label=False):

    origin_spacing = img_sitk.GetSpacing()
    origin_size = img_sitk.GetSize()
    out_spacing = [
        origin_spacing[0] * (origin_size[0] / out_size[0]),
        origin_spacing[1] * (origin_size[1] / out_size[1]),
        origin_spacing[2] * (origin_size[2] / out_size[2]),
    ]
    resize_worker = sitk.ResampleImageFilter()
    resize_worker.SetDefaultPixelValue(-1024)
    resize_worker.SetOutputOrigin(img_sitk.GetOrigin())
    resize_worker.SetOutputDirection(img_sitk.GetDirection())
    resize_worker.SetSize(out_size)
    resize_worker.SetOutputSpacing(out_spacing)

    if is_label:
        resize_worker.SetInterpolator(sitk.sitkNearestNeighbor)
    else:
        resize_worker.SetInterpolator(sitk.sitkBSpline)
    image_out = resize_worker.Execute(img_sitk)

    return image_out

def spacing(sitk_image, out_spacing=(2, 2, 2), is_label=False):
    # Resample images to 2mm spacing with SimpleITK
    original_spacing = sitk_image.GetSpacing()
    original_size = sitk_image.GetSize()
    out_size = [
        int(np.round(original_size[0] * (original_spacing[0] / out_spacing[0]))),
        int(np.round(original_size[1] * (original_spacing[1] / out_spacing[1]))),
        int(np.round(original_size[2] * (original_spacing[2] / out_spacing[2])))]
    resample_worker = sitk.ResampleImageFilter()
    # resample_worker.SetDefaultPixelValue(-1024)
    resample_worker.SetOutputOrigin(sitk_image.GetOrigin())
    resample_worker.SetOutputDirection(sitk_image.GetDirection())
    resample_worker.SetSize(out_size)
    resample_worker.SetOutputSpacing(out_spacing)
    if is_label:
        resample_worker.SetInterpolator(sitk.sitkNearestNeighbor)
    else:
        resample_worker.SetInterpolator(sitk.sitkBSpline)
    resampled = resample_worker.Execute(sitk_image)

    return resampled

def reference(img_sitk, reference_sitk):

    img_sitk.SetOrigin(reference_sitk.GetOrigin())
    img_sitk.SetSpacing(reference_sitk.GetSpacing())
    img_sitk.SetDirection(reference_sitk.GetDirection())

    return img_sitk

def bit_16to8(img_sitk, array=True):

    img_array = sitk.GetArrayFromImage(img_sitk)
    img_array = img_array - np.min(img_array)
    img = img_array / np.max(img_array) * 255
    if array:
        out = img.astype('uint8')
        return out
    else:
        out = sitk.GetImageFromArray(img)
        out = reference(out, img_sitk)
        return  out
    ###shape: z*x*y  range:[0,255]

# def slice(img_sitk, index=[0,0,0]):
#
#     size = list(img_sitk.GetSize())
#     Extractor = sitk.ExtractImageFilter()
#     Extractor.SetSize(size)
#     Extractor.SetIndex(index)
#     out_sitk = Extractor.Execute(img_sitk)
#
#     return out_sitk

parser = argparse.ArgumentParser()
parser.add_argument('--image_in', default='/data/rz/codes/tools/data/reality/image/4608357.nii.gz',type=str)
parser.add_argument('--image_out', default='/data/rz/codes/tools/data/reality/image_8/4608357.nii.gz', type=str)

if __name__ == "__main__":
    args, unknown = parser.parse_known_args()

    # image_16 = sitk.ReadImage(args.image_in)
    # image_8 = bit_16to8(image_16, array=False)
    # sitk.WriteImage(image_8, args.image_out)

    image_list = os.listdir(args.image_in)
    for i in range(len(image_list)):
        img_sitk = sitk.ReadImage(os.path.join(args.image_in, image_list[i]))
        if img_sitk.GetSpacing()[2] > 0.8:
            z = int(img_sitk.GetSize()[2] * (img_sitk.GetSpacing()[2] / 0.8))
            image_out = size(img_sitk, out_size=[512, 512, z])
        else:
            image_out = size(img_sitk, out_size=[512, 512, img_sitk.GetSize()[2]])
        sitk.WriteImage(image_out, os.path.join(args.image_out, image_list[i].split('.nii.gz')[0]+'_512.nii.gz'))
        print('The {} has finished'.format(image_list[i].split('.nii.gz')[0]+'_512.nii.gz'))
    print('Done')