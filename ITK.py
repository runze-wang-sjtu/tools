# -*- coding: utf-8 -*-
# @Time    : 2020/8/10 19:42
# @Author  : runze.wang
import pickle
import nibabel as nib
import numpy as np
import SimpleITK as sitk

def pickle_dump(item, out_file):
    with open(out_file, "wb") as opened_file:
        pickle.dump(item, opened_file)


def pickle_load(in_file):
    with open(in_file, "rb") as opened_file:
        return pickle.load(opened_file)



def load_nifti(nifti_path):
 img1 = nib.load(nifti_path)
 data = img1.get_data()
 affine = img1.affine
 data = np.asarray(data)
 return data, img1


def save_nifti(img_data, affine, save_path):
 new_image = nib.Nifti1Image(img_data, affine)
 nib.save(new_image, save_path)

def load_nifti_sitk(nifti_path):
 img = sitk.ReadImage(nifti_path)
 img_data = sitk.GetArrayFromImage(img)
 # _original = img.GetOrigin()
 # print(img.GetSize())
 # _spaces = img.GetSpacing()
 return img_data, img

def save_nifti_stik(img_data, img_attr, save_path, origin_zero=False):
 sitk_img = sitk.GetImageFromArray(img_data)
 _origin = sitk_img.GetOrigin()
 _spaces = sitk_img.GetSpacing()
 if origin_zero:
  sitk_img.SetOrigin((0,0,0))
 else:
  sitk_img.SetOrigin(img_attr.GetOrigin())
 sitk_img.SetSpacing(img_attr.GetSpacing())
 sitk_img.SetDirection(img_attr.GetDirection())

 sitk.WriteImage(sitk_img, save_path)