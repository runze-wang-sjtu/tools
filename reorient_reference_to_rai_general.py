from glob import glob
import os
import numpy as np
import itk
import SimpleITK as sitk

def reorient_to_rai(image):
    # Reorient image to RAI orientation.
    filter = itk.OrientImageFilter.New(image)
    filter.UseImageDirectionOn()
    filter.SetInput(image)
    m = itk.GetMatrixFromArray(np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]], np.float64))
    filter.SetDesiredCoordinateDirection(m)
    filter.Update()
    reoriented = filter.GetOutput()
    return reoriented
    
def resample(sitk_image, out_spacing=(2, 2, 2), is_label=False):
    # Resample images to 2mm spacing with SimpleITK
    original_spacing = sitk_image.GetSpacing()
    original_size = sitk_image.GetSize()
    out_size = [
        int(np.round(original_size[0] * (original_spacing[0] / out_spacing[0]))),
        int(np.round(original_size[1] * (original_spacing[1] / out_spacing[1]))),
        int(np.round(original_size[2] * (original_spacing[2] / out_spacing[2])))]
    resample_worker = sitk.ResampleImageFilter()
    resample_worker.SetDefaultPixelValue(-1024)
    resample_worker.SetOutputOrigin(sitk_image.GetOrigin())
    resample_worker.SetOutputDirection(sitk_image.GetDirection())
    resample_worker.SetSize(out_size)
    resample_worker.SetOutputSpacing(out_spacing)
    if is_label:
        resample_worker.SetInterpolator(sitk.sitkNearestNeighbor)
    else:
        resample_worker.SetInterpolator(sitk.sitkBSpline)
    resampled = resample_worker.Execute(sitk_image)
    return resampled

if __name__ == '__main__':
    image_folder = '/home/gdp/data/spine_large/VerSeg_complete/VerSeg/image'
    output_folder = '/home/gdp/data/spine_large/VerSeg_complete/VerSeg/image_reorient'

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    filenames = glob(os.path.join(image_folder, '*.nii.gz'))
    for filename in sorted(filenames):
        # filename = '/home/D/glo/Data/spine_seg_data/VerSe2020/training_data/segmentation/verse503_seg.nii.gz'
        basename = os.path.basename(filename)
        basename_wo_ext = basename[:basename.find('.nii.gz')]
        out_file_name = os.path.join(output_folder, basename_wo_ext + '.nii.gz')
        print(basename_wo_ext)
        # # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~reorient
        image = itk.imread(os.path.join(image_folder,  basename_wo_ext + '.nii.gz'))
        reoriented = reorient_to_rai(image)
        m = itk.GetMatrixFromArray(np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]], np.float64))
        reoriented.SetDirection(m)
        reoriented.Update()
        itk.imwrite(reoriented, out_file_name)
        print('done')

        # # #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ resample
        # image = sitk.ReadImage(out_file_name)
        # resampled = resample(image, out_spacing=(1, 1, 1), is_label=True)
        # resampled = sitk.GetArrayFromImage(resampled)
        # resampled[resampled<0] = 0
        # resampled = sitk.GetImageFromArray(resampled)
        # resampled.SetOrigin([0, 0, 0])
        # resampled.SetSpacing((1, 1, 1))
        # sitk.WriteImage(resampled, out_file_name)
        print()





