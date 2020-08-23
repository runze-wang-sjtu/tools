# -*- coding: utf-8 -*-
# @Time    : 2020/8/14 9:15
# @Author  : runze.wang

import os
import shutil

source_path = '/home/gdp/data/spine_simulation_2D/masks'
target_path = '/home/gdp/data/spine_simulation_2D/masks_2'

img_names = os.listdir(os.path.join(source_path))


for img_name in img_names:
    if 'seg' in img_name:
        shutil.copy(os.path.join(source_path, img_name),
                  os.path.join(target_path,
                  img_name.split('_seg_implant')[0]+
                  img_name.split('_seg_implant')[-1].split('.png')[0]+'_implant.png'))
    else:
        shutil.copy(os.path.join(source_path, img_name),
                  os.path.join(target_path,
                  img_name.split('_implant')[0]+
                  img_name.split('_implant')[-1].split('.png')[0]+'_implant.png'))
    # try:
    #     shutil.copy(os.path.join(source_path, img_name),
    #               os.path.join(target_path,img_name.split('.png')[0]+'_2.png'))
    # except:
    #     print('The following file ==>{}<== rename file fail\r\n'.
    #           format(os.path.join(source_path, img_name)))


