# -*- coding: utf-8 -*-
# @Time    : 2020/8/10 19:25
# @Author  : runze.wang

import os
import random
import time
from ITK import *
from data_augmentation import *

path = '/home/gdp/data/spine_large/training_data_1'

def generate_cylinder(size, R, H):
    x0, y0, z0 = int(0.5*size[0]), int(0.5*size[1]), int(0.5*size[2])
    theta = np.arange(0, 2*np.pi, 0.01)
    painting = np.zeros(size)
    x, y, z = [], [], []
    if z0 + H < size[2] and y0 + R < size[1] and x0 + R < size[0]:
        for r in range(R):
            x.append(x0 + r * np.cos(theta))
            y.append(y0 + r * np.sin(theta))
        for z in range(z0-H, z0+H):
            for i in range(len(x)):
                for j in range(len(x[i])):
                    painting[int(x[i][j]), int(y[i][j]),z] = 1
    else:
        print('Waining: The following implant image is not completed!!!')

    return painting

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

def generate_implant(seg_data,path):

    vertebra_mask_value = [20., 21., 22., 23., 24., 25.]
    for index in np.unique(seg_data):
        if index in vertebra_mask_value:
            _, x_list, y_list, z_list = np.where([seg_data == index])  ###vertebra mask location
            for num in range(1,11):
                location_index = random.randint(0, len(x_list))
                target_location = np.array([int(x_list[location_index]),
                                            int(y_list[location_index]),
                                            int(z_list[location_index])])
                source_location = np.array([int(0.5 * seg_data.shape[0]),
                                            int(0.5 * seg_data.shape[1]),
                                            int(0.5 * seg_data.shape[2])])
                translated_distance = target_location - source_location
                implant = generate_cylinder(size=seg_data.shape,
                                            R=random.randint(5, 10),
                                            H=random.randint(10, 30))
                implant_trans = implant_random_trasform(implant, trans_value=(translated_distance[0],
                                                                              translated_distance[1],
                                                                              translated_distance[2]))
                save(implant_trans, os.path.join(path, '{}_implant_vertebra_{}_num_{}.nii.gz'.
                                   format(path.split('/')[-1], int(index), num)))
                print('Time:{}====>'.format(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())),
                      os.path.join(path, '{}_implant_vertebra_{}_num_{}.nii.gz'.
                                   format(path.split('/')[-1], int(index), num)))


for root, _, files in os.walk(path):
    for file in files:
        if 'seg.nii' in file:
            seg_data, seg_img = load_nifti(os.path.join(root,file))
            generate_implant(seg_data,root)

#####   debug
# seg_data, seg_img = load_nifti(os.path.join('/home/gdp/data/spine_large/training_data/verse015', 'verse015_seg.nii.gz'))
# generate_implant(seg_data, os.path.join('/home/gdp/data/spine_large/training_data/verse015', 'verse015_implant.nii.gz'))

# print('finished')