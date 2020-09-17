# -*- coding: utf-8 -*-
# @Author  : runze.wang
# @Time    : 2020/9/16 12:13 PM
import itk
import SimpleITK as sitk
from resample import *

def itk_sitk(img_itk):

    arr = itk.GetArrayFromImage(img_itk)
    img_sitk = sitk.GetImageFromArray(arr)

    return img_sitk

def sitk_itk(img_sitk):

    arr = sitk.GetArrayFromImage(img_sitk)
    img_itk = itk.GetImageFromArray(arr)

    return img_itk