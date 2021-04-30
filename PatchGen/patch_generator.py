import SimpleITK as sitk
import numpy as np
import re
import pandas as pd
import matplotlib.pyplot as plt
import os
import shutil
import pickle
import re
from scipy import ndimage, misc

def generate_patches(data_dir, save_dir):
	# images = dict()
	# masks = dict()


	spacing_parameters = [1, 1, 1]
	size_parameters = [70, 70, 1]
	stride = 70

	for i, folder_name in enumerate(os.listdir(data_dir)):
		print(i)
		if not os.path.isdir(save_dir+'/'+folder_name):
			os.mkdir(save_dir+'/'+folder_name)

		for patient_image_path in os.listdir(data_dir+'/'+folder_name):

			save_path = save_dir+'/'+folder_name+'/'+patient_image_path.split('.')[0]

			if not os.path.isdir(save_path):
				os.mkdir(save_path)

			image_path = data_dir+'/'+folder_name+'/'+ patient_image_path

			images = sitk.ReadImage(image_path)

			factor = np.asarray(images.GetSpacing()) / [spacing_parameters[0], spacing_parameters[1],
													 spacing_parameters[2]]
			factorSize = np.asarray(images.GetSize() * factor, dtype=float)
			newSize = factorSize
			newSize = newSize.astype(dtype=int)
			newSize_tuple = tuple(newSize.tolist())

			resampler = sitk.ResampleImageFilter()
			resampler.SetReferenceImage(images)
			resampler.SetOutputSpacing([spacing_parameters[0], spacing_parameters[1], spacing_parameters[2]])
			resampler.SetSize(newSize_tuple)
			resampler.SetInterpolator(sitk.sitkNearestNeighbor)
			images = resampler.Execute(images)
			images = np.transpose(sitk.GetArrayFromImage(images).astype(dtype=float), [1, 2, 0])

			images = ndimage.zoom(images, (350.0/images.shape[0], 420.0/images.shape[1], 1), order=0)

			total_patches_per_direction = list((np.array(images.shape) - np.array(size_parameters))//stride + 1)

			count = 1
			for x_dir_num in range(total_patches_per_direction[0]):
				for y_dir_num in range(total_patches_per_direction[1]):

					temp_patch = images[stride*x_dir_num : size_parameters[0]+stride*x_dir_num, stride*y_dir_num : size_parameters[1]+stride*y_dir_num].copy()
					temp_patch = temp_patch[:,:,0]
					placeholder_patch = np.zeros((3, size_parameters[0], size_parameters[0]), dtype=type(temp_patch[0,0]))

					for channel in range(3):
						placeholder_patch[channel] = temp_patch

					with open(save_path + '/' + str(count) + '.pickle', 'wb') as f:
						pickle.dump(placeholder_patch, f)
					f.close()
					
					count += 1


generate_patches('/home/Aishik/Data', '/home/Aishik/Patches')