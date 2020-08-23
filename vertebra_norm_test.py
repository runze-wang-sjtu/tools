# -*- coding: utf-8 -*-
# @Time    : 2020/8/19 9:54
# @Author  : runze.wang

import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

path = '/home/gdp/data/spine_simulation_2D/images/verse004_vertebra_20_num_1_level_3.png'
image = Image.open(path).convert('L')
array = np.array(image)
for i in range(array.shape[0]):
    for j in range(array.shape[1]):
        if array[i][j] < 30:
            array[i][j] = 255
plt.imshow(array,cmap='gray')
plt.show()

