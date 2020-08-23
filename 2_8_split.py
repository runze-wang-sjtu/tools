# -*- coding: utf-8 -*-
# @Time    : 2020/8/14 10:31
# @Author  : runze.wang

import os
import shutil

source_path = '/home/gdp/data/spine_simulation_2D'
train_path = os.path.join(source_path,'train')
val_path = os.path.join(source_path,'val')

if not os.path.exists(os.path.join(train_path,'images')):
    os.makedirs(os.path.join(train_path,'images'))
if not os.path.exists(os.path.join(train_path,'masks')):
    os.makedirs(os.path.join(train_path,'masks'))

if not os.path.exists(os.path.join(val_path,'images')):
    os.makedirs(os.path.join(val_path,'images'))
if not os.path.exists(os.path.join(val_path,'masks')):
    os.makedirs(os.path.join(val_path,'masks'))

img_list = os.listdir(os.path.join(source_path,'images'))

for i in range(len(img_list)):
    if i < int(0.8*len(img_list)):
        shutil.copy(os.path.join(source_path,'images',img_list[i]),
                    os.path.join(train_path,'images'))
        shutil.copy(os.path.join(source_path,'masks',img_list[i].split('.png')[0]+'_implant.png'),
                    os.path.join(train_path,'masks'))
    else:
        shutil.copy(os.path.join(source_path,'images',img_list[i]),
                    os.path.join(val_path,'images'))
        shutil.copy(os.path.join(source_path,'masks',img_list[i].split('.png')[0]+'_implant.png'),
                    os.path.join(val_path,'masks'))