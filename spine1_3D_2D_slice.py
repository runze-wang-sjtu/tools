# -*- coding: utf-8 -*-
# @Time    : 2020/8/13 9:18
# @Author  : runze.wang
import cv2
from spine2_3D_2D_slice import *

if __name__ == '__main__':

    source_path = '/home/gdp/data/spine_large/training_data'
    target_path = '/home/gdp/data/spine_simulation_2D'

    num = 0
    for root, _, files in os.walk(source_path):
        implnat_names = []
        for file in files:
            if 'implant' in file:
                implnat_names.append(file)
            elif 'nii' in file and 'seg' not in file:
                image = itk.imread(os.path.join(root, file))
                reoriented = reorient_to_rai(image)
                m = itk.GetMatrixFromArray(np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]], np.float64))
                reoriented.SetDirection(m)
                reoriented.Update()
                image_data_array = itk.GetArrayFromImage(reoriented)
                image_data_image = sitk.GetImageFromArray(image_data_array)
                image_data_resample = resample(image_data_image, out_spacing=(1, 1, 1))
                image_data = sitk.GetArrayFromImage(image_data_resample)

        for implnat_name in implnat_names:
            implant = itk.imread(os.path.join(root, implnat_name))
            implant.SetDirection(image.GetDirection())  ####!!!!!!!!!!
            implant.SetSpacing(image.GetSpacing())  ####!!!!!!!!!!!!
            reoriented = reorient_to_rai(implant)
            m = itk.GetMatrixFromArray(np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]], np.float64))
            reoriented.SetDirection(m)
            reoriented.Update()
            implant_data_array = itk.GetArrayFromImage(reoriented)
            implant_data_image = sitk.GetImageFromArray(implant_data_array)
            implant_data_resample = resample(implant_data_image, out_spacing=(1, 1, 1), is_label=True)
            implant_data = sitk.GetArrayFromImage(implant_data_resample)

            implant_voxel = np.where(implant_data == 1.)
            try:
                range_z = np.max(implant_voxel[0]) - np.min(implant_voxel[0])
                for z in range(int(0.2 * range_z), int(0.8 * range_z)):
                    num += 1
                    slice_image = image_data[(np.min(implant_voxel[0])+z), :, :]
                    slice_image = slice_image - np.min(slice_image)
                    slice_image = slice_image / np.max(slice_image) * 255
                    cv2.imwrite(os.path.join(target_path, 'images_2','{}_{}_level_{}.png'.format(
                root.split('/')[-1], implnat_name.split('implant_')[-1].split('.nii')[0],z)), slice_image)

                    slice_implant = implant_data[(np.min(implant_voxel[0])+z), :, :]
                    slice_implant = slice_implant - np.min(slice_implant)
                    slice_implant = slice_implant / np.max(slice_implant) * 255
                    cv2.imwrite(os.path.join(target_path, 'labels_2', '{}_level_{}.png'.format(
                        implnat_name.split('.nii')[0], z)), slice_implant)
                    print('Time:{}==>'.format(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())),
                          'Number:{}==>'.format(num),
                          os.path.join(target_path, 'images_2','{}_{}_level_{}.png'.
                          format(root.split('/')[-1], implnat_name.split('implant_')[-1].split('.nii')[0],z)))
            except:
                print('The following file ==>{}<== is not completed!!!'.format(implnat_name))