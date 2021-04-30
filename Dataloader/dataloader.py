import numpy as np
import re
import pandas as pd
import matplotlib.pyplot as plt
import os
from scipy.ndimage.interpolation import rotate
from skimage.transform import AffineTransform, warp
import scipy.ndimage as ndimage
import shutil
import pickle
import torch
import datetime
from skimage import data, img_as_float
from skimage import io, exposure
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import cv2
import math
from PIL import Image

class Data_Loader(object):

    def __init__(self, data_path=None, dataloader_type='train', crossvalidation=0, do_augment=False, n_patches=30, patch_size=70):
        
        np.random.seed(0)
        
        self.data_path = data_path 
        self.label_path = data_path #change
        self.dataloader_type = dataloader_type
        self.crossvalidation = crossvalidation
        self.do_augment = do_augment
        self.n_patches = n_patches
        self.patch_size = patch_size

        self.generate_data_list()
 

    def sorted_alphanumeric(self, data):
        convert = lambda text: int(text) if text.isdigit() else text.lower()
        alphanum_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key) ] 
        return sorted(data, key=alphanum_key)


    def generate_data_list(self):
        
        self.name_list = np.array(self.sorted_alphanumeric(os.listdir(self.data_path)))

        if self.crossvalidation == 0:
            if self.dataloader_type == 'train':
                self.name_list = self.name_list[:self.name_list.shape[0]*3//5]

            elif self.dataloader_type == 'valid':
                self.name_list = self.name_list[self.name_list.shape[0]*3//5:self.name_list.shape[0]*4//5]

            elif self.dataloader_type == 'test':
                self.name_list = self.name_list[self.name_list.shape[0]*4//5:]

        elif self.crossvalidation == 1:
            if self.dataloader_type == 'train':
                self.name_list = self.name_list[self.name_list.shape[0]*1//5:self.name_list.shape[0]*4//5]

            elif self.dataloader_type == 'valid':
                self.name_list = self.name_list[self.name_list.shape[0]*4//5:]

            elif self.dataloader_type == 'test':
                self.name_list = self.name_list[:self.name_list.shape[0]*1//5]


        elif self.crossvalidation == 2:
            if self.dataloader_type == 'train':
                self.name_list = self.name_list[self.name_list.shape[0]*2//5:]

            elif self.dataloader_type == 'valid':
                self.name_list = self.name_list[:self.name_list.shape[0]*1//5]

            elif self.dataloader_type == 'test':
                self.name_list = self.name_list[self.name_list.shape[0]*1//5:self.name_list.shape[0]*2//5]

        elif self.crossvalidation == 3:
            if self.dataloader_type == 'train':
                self.name_list = np.concatenate((self.name_list[:self.name_list.shape[0]*1//5],self.name_list[self.name_list.shape[0]*3//5:]),axis=0)

            elif self.dataloader_type == 'valid':
                self.name_list = self.name_list[self.name_list.shape[0]*1//5:self.name_list.shape[0]*2//5]

            elif self.dataloader_type == 'test':
                self.name_list = self.name_list[self.name_list.shape[0]*2//5:self.name_list.shape[0]*3//5]


        elif self.crossvalidation == 4:
            if self.dataloader_type == 'train':
                self.name_list = np.concatenate((self.name_list[:self.name_list.shape[0]*2//5],self.name_list[self.name_list.shape[0]*4//5:]),axis=0)

            elif self.dataloader_type == 'valid':
                self.name_list = self.name_list[self.name_list.shape[0]*2//5:self.name_list.shape[0]*3//5]

            elif self.dataloader_type == 'test':
                self.name_list = self.name_list[self.name_list.shape[0]*3//5:self.name_list.shape[0]*4//5]


        self.label_dict = dict()

        self.final_label_dict = dict()


        for folders in self.name_list:
            self.label_dict[folders] = np.zeros(len(os.listdir(self.data_path + '/' + folders)))  #change
            self.final_label_dict[folders] = 0

        self.bag_dict = self.load_data()

        np.random.seed(0)


    def load_data(self):

        bag_dict = dict()

        for folder_name in self.name_list:

            number_of_timepoints = len(self.label_dict[folder_name])

            bag = np.zeros((number_of_timepoints, self.n_patches, 3, self.patch_size, self.patch_size), dtype=np.float64)

            
            for i, timepoints in enumerate(self.sorted_alphanumeric(os.listdir(self.data_path + '/' + folder_name))):
                for j, patch in enumerate(self.sorted_alphanumeric(os.listdir(self.data_path + '/' + folder_name + '/' + timepoints))):

                    with open(self.data_path + '/' + folder_name + '/' + timepoints + '/' + patch, 'rb') as f:
                        temp_patch = pickle.load(f)

                    if self.do_augment:
                        temp_patch = self.augment(temp_patch)

                    bag[i, j] = temp_patch

            bag_dict[folder_name] = bag

        return bag_dict


    def load_image_and_label(self, folder_name):
 
        bag = torch.from_numpy(self.bag_dict[folder_name])
        label = self.label_dict[folder_name].astype(int)  #change
        final_label = int(self.final_label_dict[folder_name])

        return bag, label, final_label


    def augment(self, image, hflip_prob=0.5, vflip_prob=0.5, rotation_angle_list=np.arange(-90,90)):

        if self.dataloader_type == 'train':

            if np.random.rand(1)[0]>1-hflip_prob:
                image = np.flip(image, axis=1)

            if np.random.rand(1)[0]>1-vflip_prob:
                image = np.flip(image, axis=0)

            rotation_angle = np.random.choice(rotation_angle_list)
            image = rotate(image, angle=rotation_angle, order=1, mode='constant', reshape=False)

            translation_probability = np.random.rand(1)[0]

            if translation_probability>0.66:
                transform = AffineTransform(translation=(-image.shape[0]//10,0))  # (-200,0) are x and y coordinate, change it see the effect
                image = warp(image, transform, mode="constant") #mode parameter is optional
                
            elif translation_probability>0.33:
                transform = AffineTransform(translation=(image.shape[0]//10,0))  # (-200,0) are x and y coordinate, change it see the effect
                image = warp(image, transform, mode="constant") #mode parameter is optional
                
            translation_probability = np.random.rand(1)[0]

            if translation_probability>0.66:
                transform = AffineTransform(translation=(0,-image.shape[0]//10))  # (-200,0) are x and y coordinate, change it see the effect
                image = warp(image, transform, mode="constant") #mode parameter is optional
                
            elif translation_probability>0.33:
                transform = AffineTransform(translation=(0,image.shape[0]//10))  # (-200,0) are x and y coordinate, change it see the effect
                image = warp(image, transform, mode="constant") #mode parameter is optional
                    
        return image


    def __getitem__(self, idx):

        temp_data_name = self.name_list[idx:idx+1]

        batch_images, batch_label, batch_final_label = self.load_image_and_label(temp_data_name[0])
        batch_images_tensor = batch_images.double()
        batch_label_tensor = torch.tensor(batch_label, dtype=torch.long)
        batch_final_label_tensor = torch.tensor(batch_final_label, dtype=torch.long)

        return batch_images_tensor, batch_label_tensor, batch_final_label_tensor


    def __len__(self):
        return len(self.name_list)

