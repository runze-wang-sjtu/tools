# -*- coding: utf-8 -*-
# @Time    : 2020/8/12 20:08
# @Author  : runze.wang
# -*- coding: utf-8 -*-
# @Time    : 2020/8/8 15:08
# @Author  : runze.wang
import os
import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt

data_pth = '/home/gdp/data/spine'

nii_names = os.listdir(os.path.join(data_pth,'raw'))

def load_nifti(nifti_path):
 img1 = nib.load(nifti_path)
 data = img1.get_data()
 affine = img1.affine
 data = np.asarray(data)
 return data, img1

def plot_without_margin(arr,nii_name,i):
    fig, ax = plt.subplots()
    ax.imshow(arr, cmap='gray',aspect='equal')
    plt.axis('off')
    # 去除图像周围的白边
    height, width = arr.shape
    # 如果dpi=300，那么图像大小=height*width
    fig.set_size_inches(width / 100.0 / 3.0, height / 100.0 / 3.0)
    plt.gca().xaxis.set_major_locator(plt.NullLocator())
    plt.gca().yaxis.set_major_locator(plt.NullLocator())
    plt.subplots_adjust(top=1, bottom=0, left=0, right=1, hspace=0, wspace=0)
    plt.margins(0, 0)

    # dpi是设置清晰度的，大于300就很清晰了，但是保存下来的图片很大
    plt.savefig('/home/gdp/data/spine/spine_img/{}_{}.png'
                .format(nii_name.split('.')[0],i), dpi=300)
    print('/home/gdp/data/spine/spine_img/{}_{}.png'
                .format(nii_name.split('.')[0],i))

for nii_name in nii_names:
    data = load_nifti(os.path.join(data_pth,'raw',nii_name))
    for i in range(data[0].shape[2]):
        arr = data[0][:,:,i]
        plot_without_margin(arr,nii_name,i)
        # np.save('/home/gdp/data/spine/spine_npy/{}_{}.npy'.format(nii_name.split('.')[0],i),arr)
        # print('./{}_{}.npy'.format(nii_name.split('.')[0],i))


