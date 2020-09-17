# -*- coding: utf-8 -*-
# @Author  : runze.wang
# @Time    : 2020/9/7 12:54 PM

import os
import cv2
import argparse
import numpy as np
import SimpleITK as sitk

# from sklearn.metrics import mean_absolute_error
from skimage.metrics import structural_similarity
from skimage.metrics import peak_signal_noise_ratio
from skimage.metrics import mean_squared_error

class Metrics(object):

    def __init__(self):

        self.update_num = 0
        self.metric = {'mae': 0, 'mse': 0, 'psnr': 0, 'ssim': 0}

    def compute(self, image, image_test):

        # mae = mean_absolute_error(self.image, self.image_test)
        mae = np.average(np.abs(image-image_test))
        mse = mean_squared_error(image, image_test)
        psnr = peak_signal_noise_ratio(image, image_test, data_range=255)
        ssim = structural_similarity(image, image_test, data_range=255)

        return mae, mse, psnr, ssim

    def update(self, mae, mse, psnr, ssim):

        # self.metric['mae'] += self.update()['mae']
        self.metric['mae'] += mae
        self.metric['mse'] += mse
        self.metric['psnr'] += psnr
        self.metric['ssim'] += ssim
        self.update_num += 1

    def mean(self):

        self.metric['mae'] = self.metric['mae'] / self.update_num
        self.metric['mse'] = self.metric['mse'] / self.update_num
        self.metric['psnr'] = self.metric['psnr'] / self.update_num
        self.metric['ssim'] = self.metric['ssim'] / self.update_num

        return self.metric

def crop(mask):

    index = np.where(mask == 1.)
    z_min, z_max = index[0].min(), index[0].max()
    x_min, x_max = index[1].min(), index[1].max()
    y_min, y_max = index[2].min(), index[2].max()

    return z_min, z_max, x_min, x_max, y_min, y_max

parser = argparse.ArgumentParser()
parser.add_argument('--real_path', default='/Users/runze.wang/PycharmProjects/tools/data/evaluation_placetrain/real', type=str,
                    help='The filename of image to be completed.')
parser.add_argument('--fake_path', default='/Users/runze.wang/PycharmProjects/tools/data/evaluation_placetrain/fake', type=str,
                    help='Where to read implant.')
parser.add_argument('--mask_path', default='/Users/runze.wang/PycharmProjects/tools/data/evaluation_placetrain/mask', type=str,
                    help='Where to read implant')

if __name__ == "__main__":

    args, unknown = parser.parse_known_args()

    real_list = os.listdir(args.real_path)
    fake_list = os.listdir(args.fake_path)
    mask_list = os.listdir(args.mask_path)

    metric = Metrics()
    for i in range(len(real_list)):

        assert len(mask_list) == len(real_list) == len(fake_list)
        mask_image = sitk.ReadImage(os.path.join(args.mask_path, mask_list[i]))
        mask_image = sitk.GetArrayFromImage(mask_image)
        real_image = sitk.ReadImage(os.path.join(args.real_path, 'verse{}_real.nii.gz').format(mask_list[i].split('-')[0]))
        real_image = sitk.GetArrayFromImage(real_image)
        fake_image = sitk.ReadImage(os.path.join(args.fake_path, 'verse{}_fake.nii.gz').format(mask_list[i].split('-')[0]))
        fake_image = sitk.GetArrayFromImage(fake_image)

        z_min, z_max, x_min, x_max, y_min, y_max = crop(mask_image)
        real_roi = (real_image * mask_image)[z_min:z_max, x_min:x_max, y_min:y_max]
        fake_roi = (fake_image * mask_image)[z_min:z_max, x_min:x_max, y_min:y_max]

        mae, mse, psnr, ssim = metric.compute(real_roi, fake_roi)
        print('Number: {}, mae:{}, mse:{}, psnr:{}, ssim:{}'.
              format((mask_list[i].split('-')[0]), mae, mse, psnr, ssim))
        metric.update(mae, mse, psnr, ssim)
    result = metric.mean()
    print('MEAN===> mae:{}, mse:{}, psnr:{}, ssim:{}'.
              format(result.get('mae'), result.get('mse'), result.get('psnr'), result.get('ssim')))
    print('Done')
