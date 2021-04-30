import SimpleITK as sitk
import numpy as np
import re
import pandas as pd
import scipy.ndimage as ndimage
import matplotlib.pyplot as plt
import os
from scipy.ndimage.interpolation import rotate
from skimage.transform import AffineTransform, warp
import shutil
import pickle
import torch
import datetime
from skimage import data, img_as_float
from skimage import io, exposure


class DataLoader(object):

    def __init__(self, data_path, dataloader_type='train', batchsize=1, device='cpu',image_resolution=[512,512], invert=False, remove_wires=False):
        
        np.random.seed(1)

        self.data_path = data_path
        if dataloader_type == 'test':
            # self.lung_mask_path = '/content/gdrive/My Drive/Project_covid/Covid_xray/lung_segmentation_exp/lung_resunet_multiinput_0vs1_focal_validour_preprocessed/prediction_dir'
            self.lung_mask_path = self.data_path + '/prediction_dir'

        # self.lung_path = data_path + '/CXR_png' 
        # self.lung_mask_path_left = data_path  + '/ManualMask/leftMask' 
        # self.lung_mask_path_right = data_path  + '/ManualMask/rightMask' 
        self.lung_path = data_path + '/Lungs' 
        self.lung_mask_path_left = data_path  + '/Left_Mask' 
        self.lung_mask_path_right = data_path  + '/Right_Mask' 
        # self.lung_mask_path_left = data_path  + '/Lung_Mask' 
        # self.lung_mask_path_right = data_path  + '/Lung_Mask' 
        # self.wires_mask_path_right = data_path  + '/Wires' 
        self.invert=invert
        self.remove_wires=remove_wires
        self.image_resolution = image_resolution
        self.device = device
        self.dataloader_type = dataloader_type
        self.batchsize = batchsize
        self.masking = False
        self.test_pickle = False
        self.generate_data_list()

    def sorted_alphanumeric(self, data):
        convert = lambda text: int(text) if text.isdigit() else text.lower()
        alphanum_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key) ] 
        return sorted(data, key=alphanum_key)

    def merge(self,word):

        #temp_name = ''
        #for j in range(len(word.split('.'))-1):
            #if word.split('.')[j] != 'nii':
                #temp_name = temp_name + word.split('.')[j] + '.'
        #temp_name = temp_name + 'nii.gz'
        return word

        # temp_word_list = word.split('_')
        # try:
        #     a = int(temp_word_list[-1])  # checking if last number
        #     return word
        # except:
        #     if len(temp_word_list)==1:
        #         return word
        #     temp_word = ''
        #     for i in temp_word_list[:-1]:
        #         temp_word += i
        #         temp_word += '_'

        #     return temp_word[:-1]
        

    def generate_data_list(self):

        if self.dataloader_type != 'test':

            if not os.path.isdir(self.data_path + '/pickle_dict'):

                self.lung_pickle_array = dict()
                self.lung_mask_pickle_array = dict()
                self.keys_array = []
                self.original_size_array = dict()
                self.spacing = dict()
                self.size_ = dict()
                self.origin = dict()
                self.wires_array = dict()
                lung_left_mask_pickle_array = dict()
                lung_right_mask_pickle_array = dict()
                        

                for i,files in enumerate(self.sorted_alphanumeric(os.listdir(self.lung_path))[:]):

                    if files.split('.')[-1]=='png':
                        temp_image = plt.imread(self.lung_path+'/'+files)
                        if len(temp_image.shape)==3:
                            temp_image = temp_image[:,:,0]
                        self.spacing[self.merge(files)] = (1.0,1.0,1.0)
                        self.size_[self.merge(files)] = (temp_image.shape[1], temp_image.shape[0])
                        self.origin[self.merge(files)] = (0.0,0.0,0.0)

                    elif files.split('.')[-1]=='dcm' or files.split('.')[-1]=='gz':
                        temp_image = sitk.ReadImage(self.lung_path+'/'+files)

                        self.spacing[self.merge(files)] = temp_image.GetSpacing()
                        self.size_[self.merge(files)] = temp_image.GetSize()
                        self.origin[self.merge(files)] = temp_image.GetOrigin()

                        factor = np.asarray(self.spacing[self.merge(files)]) / [1, 1, 1]
                        factorSize = np.asarray(self.size_[self.merge(files)] * factor, dtype=float)
                        newSize = factorSize
                        newSize = newSize.astype(dtype=int)
                        newSize_tuple = tuple(newSize.tolist())

                        resampler = sitk.ResampleImageFilter()
                        resampler.SetReferenceImage(temp_image)
                        resampler.SetOutputSpacing([1, 1, 1])
                        resampler.SetSize(newSize_tuple)
                        resampler.SetInterpolator(sitk.sitkNearestNeighbor)
                        temp_image = resampler.Execute(temp_image)

                        temp_image = sitk.GetArrayFromImage(temp_image)
                        if len(temp_image.shape) == 3:
                            temp_image = temp_image[0]
                        # print(temp_image.shape)
                    if files.split('.')[-1]=='png' or files.split('.')[-1]=='dcm' or files.split('.')[-1]=='gz':    
                        self.original_size_array[self.merge(files)] = np.asarray(temp_image.shape)
                        temp_image = (temp_image - temp_image.min())/(temp_image.max()-temp_image.min())
                        temp_mask =  np.where(temp_image!=0,1,0)
                        temp_image = exposure.equalize_hist(temp_image, mask = np.where(temp_image!=0,1,0))
                        temp_image = temp_image*temp_mask
                        temp_image = ndimage.zoom(temp_image, np.asarray(self.image_resolution) / np.asarray(temp_image.shape), order=0)
                        temp_image = ndimage.median_filter(temp_image,5)
                        self.lung_pickle_array[self.merge(files)] = temp_image
                        self.keys_array.append(self.merge(files))


                for i,files in enumerate(self.sorted_alphanumeric(os.listdir(self.lung_mask_path_left))):

                    if files.split('.')[-1]=='png':
                        temp_image = plt.imread(self.lung_mask_path_left+'/'+files)
                        if len(temp_image.shape)==3:
                            temp_image = temp_image[:,:,0]
                    elif files.split('.')[-1]=='dcm' or files.split('.')[-1]=='gz':
                        temp_image = sitk.ReadImage(self.lung_mask_path_left+'/'+files)

                        factor = np.asarray(self.spacing[self.merge(files)]) / [1, 1, 1]
                        factorSize = np.asarray(self.size_[self.merge(files)] * factor, dtype=float)
                        newSize = factorSize
                        newSize = newSize.astype(dtype=int)
                        newSize_tuple = tuple(newSize.tolist())

                        resampler = sitk.ResampleImageFilter()
                        resampler.SetReferenceImage(temp_image)
                        resampler.SetOutputSpacing([1, 1, 1])
                        resampler.SetSize(newSize_tuple)
                        resampler.SetInterpolator(sitk.sitkNearestNeighbor)
                        temp_image = resampler.Execute(temp_image)

                        temp_image = sitk.GetArrayFromImage(temp_image)
                        if len(temp_image.shape) == 3:
                            temp_image = temp_image[0]
                        # print(temp_image.shape)

                    if files.split('.')[-1]=='png' or files.split('.')[-1]=='dcm' or files.split('.')[-1]=='gz':
                        temp_image = ndimage.zoom(temp_image, np.asarray(self.image_resolution) / np.asarray(temp_image.shape), order=0)
                        lung_left_mask_pickle_array[self.merge(files)] = temp_image


                for i,files in enumerate(self.sorted_alphanumeric(os.listdir(self.lung_mask_path_right))):

                    if files.split('.')[-1]=='png':
                        temp_image = plt.imread(self.lung_mask_path_right+'/'+files)
                        if len(temp_image.shape)==3:
                            temp_image = temp_image[:,:,0]
                    elif files.split('.')[-1]=='dcm' or files.split('.')[-1]=='gz':
                        temp_image = sitk.ReadImage(self.lung_mask_path_right+'/'+files)

                        factor = np.asarray(self.spacing[self.merge(files)]) / [1, 1, 1]
                        factorSize = np.asarray(self.size_[self.merge(files)] * factor, dtype=float)
                        newSize = factorSize
                        newSize = newSize.astype(dtype=int)
                        newSize_tuple = tuple(newSize.tolist())

                        resampler = sitk.ResampleImageFilter()
                        resampler.SetReferenceImage(temp_image)
                        resampler.SetOutputSpacing([1, 1, 1])
                        resampler.SetSize(newSize_tuple)
                        resampler.SetInterpolator(sitk.sitkNearestNeighbor)
                        temp_image = resampler.Execute(temp_image)

                        temp_image = sitk.GetArrayFromImage(temp_image)
                        if len(temp_image.shape) == 3:
                            temp_image = temp_image[0]

                    if files.split('.')[-1]=='png' or files.split('.')[-1]=='dcm' or files.split('.')[-1]=='gz':
                        temp_image = ndimage.zoom(temp_image, np.asarray(self.image_resolution) / np.asarray(temp_image.shape), order=0)
                        lung_right_mask_pickle_array[self.merge(files)] = temp_image

                if self.remove_wires is True:

                    for i,files in enumerate(self.sorted_alphanumeric(os.listdir(self.wires_mask_path_right))):

                        if files.split('.')[-1]=='png':
                            temp_image = plt.imread(self.wires_mask_path_right+'/'+files)
                            if len(temp_image.shape)==3:
                                temp_image = temp_image[:,:,0]
                        elif files.split('.')[-1]=='dcm' or files.split('.')[-1]=='gz':
                            temp_image = sitk.ReadImage(self.wires_mask_path_right+'/'+files)

                            factor = np.asarray(self.spacing[self.merge(files)]) / [1, 1, 1]
                            factorSize = np.asarray(self.size_[self.merge(files)] * factor, dtype=float)
                            newSize = factorSize
                            newSize = newSize.astype(dtype=int)
                            newSize_tuple = tuple(newSize.tolist())

                            resampler = sitk.ResampleImageFilter()
                            resampler.SetReferenceImage(temp_image)
                            resampler.SetOutputSpacing([1, 1, 1])
                            resampler.SetSize(newSize_tuple)
                            resampler.SetInterpolator(sitk.sitkNearestNeighbor)
                            temp_image = resampler.Execute(temp_image)

                            temp_image = sitk.GetArrayFromImage(temp_image)
                            if len(temp_image.shape) == 3:
                                temp_image = temp_image[0]

                        if files.split('.')[-1]=='png' or files.split('.')[-1]=='dcm' or files.split('.')[-1]=='gz':
                            temp_image = ndimage.zoom(temp_image, np.asarray(self.image_resolution) / np.asarray(temp_image.shape), order=0)
                            self.wires_array[self.merge(files)] = temp_image


                for file_name in self.keys_array:
                    temp_image = lung_left_mask_pickle_array[file_name] + lung_right_mask_pickle_array[file_name]
                    self.lung_mask_pickle_array[file_name] = np.where(temp_image>=1,1,0)

                self.keys_array = np.array(self.keys_array)

                os.mkdir(self.data_path + '/pickle_dict')

                with open(self.data_path + '/pickle_dict/lung_pickle_array.pickle', 'wb') as f:
                    pickle.dump(self.lung_pickle_array, f)
                f.close()   

                with open(self.data_path + '/pickle_dict/lung_mask_pickle_array.pickle', 'wb') as f:
                    pickle.dump(self.lung_mask_pickle_array, f)
                f.close()  

                with open(self.data_path + '/pickle_dict/keys_array.pickle', 'wb') as f:
                    pickle.dump(self.keys_array, f)
                f.close() 

                with open(self.data_path + '/pickle_dict/original_size_array.pickle', 'wb') as f:
                    pickle.dump(self.original_size_array, f)
                f.close()     

                with open(self.data_path + '/pickle_dict/spacing.pickle', 'wb') as f:
                    pickle.dump(self.spacing, f)
                f.close()  

                with open(self.data_path + '/pickle_dict/size_.pickle', 'wb') as f:
                    pickle.dump(self.size_, f)
                f.close()  

                with open(self.data_path + '/pickle_dict/origin.pickle', 'wb') as f:
                    pickle.dump(self.origin, f)
                f.close() 

                if self.remove_wires is True:
                    with open(self.data_path + '/pickle_dict/wires_array.pickle', 'wb') as f:
                        pickle.dump(self.wires_array, f)
                    f.close()   

            else:
                with open(self.data_path + '/pickle_dict/lung_pickle_array.pickle', 'rb') as f:
                    self.lung_pickle_array = pickle.load(f)
                f.close()

                with open(self.data_path + '/pickle_dict/lung_mask_pickle_array.pickle', 'rb') as f:
                    self.lung_mask_pickle_array = pickle.load(f)
                f.close()

                with open(self.data_path + '/pickle_dict/keys_array.pickle', 'rb') as f:
                    self.keys_array = pickle.load(f)
                f.close()

                with open(self.data_path + '/pickle_dict/original_size_array.pickle', 'rb') as f:
                    self.original_size_array = pickle.load(f)
                f.close()

                with open(self.data_path + '/pickle_dict/spacing.pickle', 'rb') as f:
                    self.spacing = pickle.load(f)
                f.close()

                with open(self.data_path + '/pickle_dict/size_.pickle', 'rb') as f:
                    self.size_ = pickle.load(f)
                f.close()

                with open(self.data_path + '/pickle_dict/origin.pickle', 'rb') as f:
                    self.origin = pickle.load(f)
                f.close()

                # print(self.keys_array)
                if self.remove_wires is True:
                    with open(self.data_path + '/pickle_dict/wires_array.pickle', 'rb') as f:
                        self.wires_array = pickle.load(f)
                    f.close() 

            np.random.seed(0)
            np.random.shuffle(self.keys_array)

            # M_list = []
            # A_list = []
            # for name in self.keys_array:
            #     if name[0]=='A':
            #         A_list.append(name)
            #     else:
            #       M_list.append(name)

            # self.keys_array = np.concatenate((M_list,A_list))

            if self.dataloader_type == 'train':
                self.data_list = self.keys_array[:self.keys_array.shape[0]*70//100]

                if self.remove_wires is False:
                    M_list = []
                    for name in self.keys_array[self.keys_array.shape[0]*70//100:]:
                        if name[0]=='M':
                            M_list.append(name)
                    print(self.data_list.shape)
                    self.data_list = np.concatenate((self.data_list,np.array(M_list)))
                    print(self.data_list.shape)

                # for key in self.keys_array[self.keys_array.shape[0]*70//100:]:
                #     del self.lung_pickle_array[key]
                #     del self.lung_mask_pickle_array[key]
                #     if self.remove_wires is True:
                #         del self.wires_array[key]  

            elif self.dataloader_type == 'valid':
                self.data_list = self.keys_array[self.keys_array.shape[0]*70//100:]
                # for key in self.keys_array[:self.keys_array.shape[0]*70//100]:
                #     del self.lung_pickle_array[key]
                #     del self.lung_mask_pickle_array[key]
                #     if self.remove_wires is True:
                #         del self.wires_array[key]

                if self.remove_wires is False:
                    A_list = []
                    for name in self.data_list:
                        if name[0]=='A':
                            A_list.append(name)
                    print(self.data_list.shape)
                    self.data_list = np.array(A_list)
                    print(self.data_list.shape)

            
            elif self.dataloader_type == 'complete':
                self.data_list = self.keys_array

            del self.keys_array

            if self.invert == True:
                for key in self.data_list:
                    self.lung_pickle_array[key] = 1.0 - self.lung_pickle_array[key]

            if self.remove_wires is True:
                for key in self.data_list:
                    if self.masking is True:
                        self.lung_pickle_array[key] = self.lung_pickle_array[key]*self.lung_mask_pickle_array[key]
                    # self.wires_array[key] = np.where(self.wires_array[key]==1,1,0)
                    # self.wires_array[key] = ndimage.binary_dilation(self.wires_array[key],iterations=2)
                    # self.wires_array[key] = ndimage.median_filter(self.wires_array[key],5)
                    # self.wires_array[key] = ndimage.binary_closing(self.wires_array[key])

                    self.lung_mask_pickle_array[key] = self.wires_array[key]*self.lung_mask_pickle_array[key]

                    del self.wires_array[key]
            # else:
            #     for key in self.data_list:
            #         if key[0]=='A':
            #             self.lung_mask_pickle_array[key] = ndimage.median_filter(self.lung_mask_pickle_array[key],5)
            #             self.lung_mask_pickle_array[key] = ndimage.binary_closing(self.lung_mask_pickle_array[key])

        elif self.dataloader_type == 'test':

            if self.test_pickle is False:
                self.lung_pickle_array = dict()
                self.keys_array = []      
                self.original_size_array = dict()
                self.spacing = dict()
                self.size_ = dict()
                self.origin = dict()
                self.lung_mask_pickle_array = dict()

                for i,files in enumerate(self.sorted_alphanumeric(os.listdir(self.data_path))):
                    print(i)
                    if files.split('.')[-1]=='png' or files.split('.')[-1]=='jpg' or files.split('.')[-1]=='jpeg':
                        temp_image = plt.imread(self.data_path+'/'+files)
                        self.spacing[self.merge(files)] = (1.0,1.0,1.0)
                        if len(temp_image.shape)==3:
                                temp_image = temp_image[:,:,0]
                        self.size_[self.merge(files)] = (temp_image.shape[1], temp_image.shape[0])
                        self.origin[self.merge(files)] = (0.0,0.0,0.0)

                    elif files.split('.')[-1]=='dcm' or files.split('.')[-1]=='gz':
                        temp_image = sitk.ReadImage(self.data_path+'/'+files)

                        self.spacing[self.merge(files)] = temp_image.GetSpacing()
                        self.size_[self.merge(files)] = temp_image.GetSize()
                        self.origin[self.merge(files)] = temp_image.GetOrigin()

                        factor = np.asarray(self.spacing[self.merge(files)]) / [1, 1, 1]
                        factorSize = np.asarray(self.size_[self.merge(files)] * factor, dtype=float)
                        newSize = factorSize
                        newSize = newSize.astype(dtype=int)
                        newSize_tuple = tuple(newSize.tolist())

                        resampler = sitk.ResampleImageFilter()
                        resampler.SetReferenceImage(temp_image)
                        resampler.SetOutputSpacing([1, 1, 1])
                        resampler.SetSize(newSize_tuple)
                        resampler.SetInterpolator(sitk.sitkNearestNeighbor)
                        temp_image = resampler.Execute(temp_image)

                        temp_image = sitk.GetArrayFromImage(temp_image)
                        if len(temp_image.shape) == 3:
                            temp_image = temp_image[0]

                    if files.split('.')[-1]=='png' or files.split('.')[-1]=='dcm' or files.split('.')[-1]=='gz' or files.split('.')[-1]=='jpg' or files.split('.')[-1]=='jpeg':
                        self.original_size_array[self.merge(files)] = np.asarray(temp_image.shape)
                        temp_image = (temp_image - temp_image.min())/(temp_image.max()-temp_image.min())
                        temp_mask =  np.where(temp_image!=0,1,0)
                        temp_image = exposure.equalize_hist(temp_image, mask = np.where(temp_image!=0,1,0))
                        temp_image = temp_image*temp_mask
                        temp_image = ndimage.zoom(temp_image, np.asarray(self.image_resolution) / np.asarray(temp_image.shape), order=0)
                        temp_image = ndimage.median_filter(temp_image,5)
                        self.lung_pickle_array[self.merge(files)] = temp_image
                        self.keys_array.append(self.merge(files))

                if self.remove_wires is True:
                    if self.masking is True:
                        for i,files in enumerate(self.sorted_alphanumeric(os.listdir(self.lung_mask_path))):

                            if files.split('.')[-1]=='png':
                                temp_image = plt.imread(self.lung_mask_path+'/'+files)
                                if len(temp_image.shape)==3:
                                    temp_image = temp_image[:,:,0]
                            elif files.split('.')[-1]=='dcm' or files.split('.')[-1]=='gz':
                                temp_image = sitk.ReadImage(self.lung_mask_path+'/'+files)
                                spacing = temp_image.GetSpacing()
                                size_ = temp_image.GetSize()
                                #[10:] to remove Pred_mask_ from saved name, see test.py 
                                factor = np.asarray(spacing) / np.ones(np.asarray(spacing).shape[0])
                                factorSize = np.asarray(size_ * factor, dtype=float)
                                newSize = factorSize
                                newSize = newSize.astype(dtype=int)
                                newSize_tuple = tuple(newSize.tolist())

                                resampler = sitk.ResampleImageFilter()
                                resampler.SetReferenceImage(temp_image)
                                resampler.SetOutputSpacing(np.ones(np.asarray(spacing).shape[0]))
                                resampler.SetSize(newSize_tuple)
                                resampler.SetInterpolator(sitk.sitkNearestNeighbor)
                                temp_image = resampler.Execute(temp_image)

                                temp_image = sitk.GetArrayFromImage(temp_image)
                                if len(temp_image.shape) == 3:
                                    temp_image = temp_image[0]

                            if files.split('.')[-1]=='png' or files.split('.')[-1]=='dcm' or files.split('.')[-1]=='gz':
                                temp_image = ndimage.zoom(temp_image, np.asarray(self.image_resolution) / np.asarray(temp_image.shape), order=0)
                                self.lung_mask_pickle_array[self.merge(files[10:])] = temp_image

            else:
                with open(self.data_path + '/lung_pickle_array.pickle', 'rb') as f:
                    self.lung_pickle_array = pickle.load(f)
                f.close()

                with open(self.data_path + '/keys_array.pickle', 'rb') as f:
                    self.keys_array = pickle.load(f)
                f.close()

                with open(self.data_path + '/original_size_array.pickle', 'rb') as f:
                    self.original_size_array = pickle.load(f)
                f.close()

                with open(self.data_path + '/spacing.pickle', 'rb') as f:
                    self.spacing = pickle.load(f)
                f.close()

                with open(self.data_path + '/size_.pickle', 'rb') as f:
                    self.size_ = pickle.load(f)
                f.close()

                with open(self.data_path + '/origin.pickle', 'rb') as f:
                    self.origin = pickle.load(f)
                f.close()

                # if self.masking is True:
                    # with open(self.data_path + '/pickle_dict/lung_mask_pickle_array.pickle', 'rb') as f:
                    #     self.lung_mask_pickle_array = pickle.load(f)
                    # f.close()

                if self.remove_wires is True:
                    if self.masking is True:
                        self.lung_mask_pickle_array = dict()
                        for i,files in enumerate(self.sorted_alphanumeric(os.listdir(self.lung_mask_path))):

                            if files.split('.')[-1]=='png':
                                temp_image = plt.imread(self.lung_mask_path+'/'+files)
                                if len(temp_image.shape)==3:
                                    temp_image = temp_image[:,:,0]
                            elif files.split('.')[-1]=='dcm' or files.split('.')[-1]=='gz':
                                temp_image = sitk.ReadImage(self.lung_mask_path+'/'+files)
                                #[10:] to remove Pred_mask_ from saved name, see test.py saving code
                                factor = np.asarray(self.spacing[files[10:]]) / [1, 1, 1]
                                factorSize = np.asarray(self.size_[files[10:]] * factor, dtype=float)
                                newSize = factorSize
                                newSize = newSize.astype(dtype=int)
                                newSize_tuple = tuple(newSize.tolist())

                                resampler = sitk.ResampleImageFilter()
                                resampler.SetReferenceImage(temp_image)
                                resampler.SetOutputSpacing([1, 1, 1])
                                resampler.SetSize(newSize_tuple)
                                resampler.SetInterpolator(sitk.sitkNearestNeighbor)
                                temp_image = resampler.Execute(temp_image)

                                temp_image = sitk.GetArrayFromImage(temp_image)
                                if len(temp_image.shape) == 3:
                                    temp_image = temp_image[0]

                            if files.split('.')[-1]=='png' or files.split('.')[-1]=='dcm' or files.split('.')[-1]=='gz':
                                temp_image = ndimage.zoom(temp_image, np.asarray(self.image_resolution) / np.asarray(temp_image.shape), order=0)
                                self.lung_mask_pickle_array[files[10:]] = temp_image

            self.data_list = np.array(self.keys_array)
            if self.remove_wires is True:
                for key in self.data_list:
                    if self.masking is True:
                        self.lung_pickle_array[key] = self.lung_pickle_array[key]*self.lung_mask_pickle_array[key]
            
            del self.keys_array
            print(len(self.data_list))
            if self.invert == True:
                for key in self.data_list:
                    self.lung_pickle_array[key] = 1.0 - self.lung_pickle_array[key]
                    
    def load_image_and_label(self, image_list):

        images = np.zeros((image_list.shape[0], 1, self.image_resolution[0], self.image_resolution[1]))
        labels = np.zeros((image_list.shape[0], self.image_resolution[0], self.image_resolution[1]))

        if self.dataloader_type == 'train':
            for i,image_id in enumerate(image_list):
                images[i][0], labels[i] = self.augment(self.lung_pickle_array[image_id], self.lung_mask_pickle_array[image_id])
        else:
            for i,image_id in enumerate(image_list):
                images[i][0] = self.lung_pickle_array[image_id]
                if self.dataloader_type != 'test':
                    labels[i] = self.lung_mask_pickle_array[image_id]
        
        return images, labels

    def augment(self, image, label, hflip_prob=0.75, vflip_prob=0.75, rotation_angle_list=np.arange(-90,90)):

        if np.random.rand(1)[0]>1-hflip_prob:
            image = np.flip(image, axis=1)
            label = np.flip(label, axis=1)
         
        if np.random.rand(1)[0]>1-vflip_prob:
            image = np.flip(image, axis=0)
            label = np.flip(label, axis=0)


        rotation_angle = np.random.choice(rotation_angle_list)
        image = rotate(image, angle=rotation_angle, order=1, mode='constant', reshape=False)
        label = rotate(label, angle=rotation_angle, order=1, mode='constant', reshape=False)

        return image, label

    def __getitem__(self, idx):

        if self.dataloader_type=='train':
            if self.batchsize >= len(self.data_list):
                temp_data_list = self.data_list
            else:  
                if (idx+1)*self.batchsize >= len(self.data_list):
                    temp_data_list = self.data_list[idx*self.batchsize : ]
                    np.random.shuffle(self.data_list)
                else:
                    temp_data_list = self.data_list[idx*self.batchsize : (idx+1)*self.batchsize]
                    if len(self.data_list)%self.batchsize == 0:
                        if idx == len(self.data_list)//self.batchsize:
                            np.random.shuffle(self.data_list)
        else:
            if self.batchsize >= len(self.data_list):
                temp_data_list = self.data_list
            else: 
                if (idx+1)*self.batchsize >= len(self.data_list):
                    temp_data_list = self.data_list[idx*self.batchsize : ]
                else:
                    temp_data_list = self.data_list[idx*self.batchsize : (idx+1)*self.batchsize]

        batch_images, batch_label = self.load_image_and_label(np.array(temp_data_list))
        batch_images_tensor = torch.from_numpy(batch_images).double().to(self.device)

        if self.dataloader_type != 'test':
            batch_label_tensor = torch.tensor(batch_label, dtype=torch.long, device=self.device)
            return batch_images_tensor, batch_label_tensor  
        else: 
            return batch_images_tensor
