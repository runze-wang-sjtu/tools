# -*- coding: utf-8 -*-
# @Author  : runze.wang
# @Time    : 2020/9/16 9:52 AM

import os
import numpy as np
import SimpleITK as sitk
from utils import *
from resample import *
from reorient_reference_to_rai_general import *

def segmetation_reference_to_mask(img_sitk, mask_sitk, out_array=False):

    mask_arr = sitk.GetArrayFromImage(mask_sitk)
    img_arr = sitk.GetArrayFromImage(img_sitk)
    mask_arr[mask_arr<20] = 0
    mask_arr[mask_arr>=20] = 1
    img_mul_mask = img_arr * mask_arr
    if out_array == False:
        img_mul_mask = sitk.GetImageFromArray(img_mul_mask)
        mask_sitk = sitk.GetImageFromArray(mask_arr)
        img_mul_mask = reference(img_mul_mask, img_sitk)
        mask_sitk = reference(mask_sitk, img_sitk)
        return img_mul_mask, mask_sitk
    else:
        return img_mul_mask, mask_arr

def crop_reference_to_mask(img_arr, mask_arr):

    mask_range = np.where(mask_arr == 1.)
    z_min, z_max = mask_range[0].min(), mask_range[0].max()
    arr_out = img_arr[z_min:z_max, :, :]

    return arr_out

parser = argparse.ArgumentParser()
parser.add_argument('--input', default='/data/rz/codes/tools/data/seg', type=str,
                    help='The path of image to be read.')
parser.add_argument('--output', default='/data/rz/codes/tools/data/vertebra', type=str,
                    help='The path of image to be read.')

if __name__ == '__main__':

    args, unknown = parser.parse_known_args()

    for root, _, files in os.walk(args.input):
        if len(files) < 6:
            for file in files:
                if 'nii' in file and 'seg' not in file:

                    image = itk.imread(os.path.join(args.input, root, file))
                    reoriented = reorient_to_rai(image)
                    m = itk.GetMatrixFromArray(np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]], np.float64))
                    reoriented.SetDirection(m)
                    reoriented.Update()
                    reoriented_image = itk_sitk(reoriented)
                    reoriented_image_resample = resample(reoriented_image, out_spacing=(1, 1, 1), is_label=False)

                    mask = itk.imread(os.path.join(args.input, root, file.split('.nii')[0]+'_seg.nii.gz'))
                    reoriented = reorient_to_rai(mask)
                    m = itk.GetMatrixFromArray(np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]], np.float64))
                    reoriented.SetDirection(m)
                    reoriented.Update()
                    reoriented_mask = itk_sitk(reoriented)
                    reoriented_mask_resample = resample(reoriented_mask, out_spacing=(1, 1, 1), is_label=True)

                    print('The following file {} is being processed'.format(file))
                    img, mask = segmetation_reference_to_mask(reoriented_image_resample, reoriented_mask_resample, out_array=True)
                    out_arr = crop_reference_to_mask(img, mask)
                    out_sitk = reference(sitk.GetImageFromArray(out_arr), reoriented_image_resample)
                    sitk.WriteImage(out_sitk, os.path.join(args.output, file))
                    print('The following file {} has finished'.format(file))
        else:
            print('There are more than five items in this folder{}'.format(root))
    print('Done')