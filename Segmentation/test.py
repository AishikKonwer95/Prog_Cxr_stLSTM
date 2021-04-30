from __future__ import print_function
import sys
import datetime
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torch.optim.lr_scheduler import StepLR
import torch.optim as optim
import numpy as np
import torch
import SimpleITK as sitk
from tqdm import tqdm
import argparse
import pickle
import os
import scipy.ndimage as ndimage
import shutil
import matplotlib.pyplot as plt
from skimage import io, morphology
import losses
import dataloader_cxr
from cxr_attention_resunet import Attention_UNet
from cxr_resunet import UNet


parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', type=int, default = 1)
parser.add_argument('--datapath', type=str)
parser.add_argument('--model_path', type=str)
parser.add_argument('--save_path', type=str)

args = parser.parse_args()

batch_size = args.batch_size
data_path = args.datapath
model_path = args.model_path
save_path = args.save_path

generate_mask = True

modeltype = 'resunet'
dataloader_type = 'test'
image_type = 'png'
n_classes = 2
classes = [0,1]
clubbed = []
model_depth = 5
use_attention = False
use_multiinput_architecture = True
invert = False
remove_wires=False
wf = 5

if classes[0] != 0:
    exclude_0 = True
else:
    exclude_0 = False

image_resolution = [512,512]

# gpu settings
use_cuda = torch.cuda.is_available()
print('gpu status =', use_cuda)
device = torch.device("cuda" if use_cuda else "cpu")
kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}

# using seed so to be deterministic
np.random.seed(0)
torch.manual_seed(0)
torch.cuda.empty_cache()
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = True

if not os.path.isdir(save_path):
    os.mkdir(save_path)
#else:
#    shutil.rmtree(save_path, ignore_errors=True)
#    os.mkdir(save_path)

def remove_small_regions(img, size):
    if remove_wires is True:
        img_ = np.zeros_like(img)
        """Morphologically removes small (less than size) connected regions of 0s or 1s."""
        for i in range(img.shape[0]):
            temp_image = img[i]
            temp_image = morphology.binary_erosion(temp_image)
            temp_image = ndimage.median_filter(temp_image,5)
            img_[i] = temp_image
        return img_

    else:
        img_ = np.zeros_like(img)
        """Morphologically removes small (less than size) connected regions of 0s or 1s."""
        for i in range(img.shape[0]):
            temp_image = img[i]
            temp_image = morphology.remove_small_objects(temp_image, size)
            temp_image = morphology.remove_small_holes(temp_image, size)
            temp_image = morphology.binary_erosion(temp_image)
            temp_image = morphology.binary_dilation(temp_image)
            temp_image = ndimage.median_filter(temp_image,7)
            img_[i] = temp_image
        return img_
            

def main():

    if use_multiinput_architecture is False:
        if modeltype == 'unet':
            model = UNet(n_classes=n_classes, padding=True, depth=model_depth, up_mode='upsample', batch_norm=True, residual=False).double().to(device)
        elif modeltype == 'resunet':
            model = UNet(n_classes=n_classes, padding=True, depth=model_depth, up_mode='upsample', batch_norm=True, residual=True).double().to(device)

    elif use_multiinput_architecture is True:
        if modeltype == 'unet':
            model = Attention_UNet(n_classes=n_classes, padding=True, up_mode='upconv', batch_norm=True, residual=False, wf=wf, use_attention=use_attention).double().to(device)
        elif modeltype == 'resunet':
            model = Attention_UNet(n_classes=n_classes, padding=True, up_mode='upconv', batch_norm=True, residual=True, wf=wf, use_attention=use_attention).double().to(device)


    checkpoint = torch.load(args.model_path, map_location=lambda storage, loc: storage)
    model.load_state_dict(checkpoint['model_state_dict'])

    valid_loader = dataloader_cxr.DataLoader(data_path, dataloader_type=dataloader_type, batchsize=batch_size, device=device, image_resolution=image_resolution, invert=invert, remove_wires=remove_wires)
    
    if len(valid_loader.data_list)%batch_size ==0:
        total_idx_valid = len(valid_loader.data_list)//batch_size
    else:
        total_idx_valid = len(valid_loader.data_list)//batch_size + 1
    
    model.eval() 
    prediction_array = np.zeros((len(valid_loader.data_list), image_resolution[0], image_resolution[1]))
    if valid_loader.dataloader_type != "test":
        input_mask_array = np.zeros((len(valid_loader.data_list), image_resolution[0], image_resolution[1]))

    valid_count = 0
    valid_dice_score = 0.0

    if 0 in classes:        
        valid_dice_score_0 = 0.0
    if 1 in classes:
        valid_dice_score_1 = 0.0

    for idx in range(total_idx_valid):
        with torch.no_grad():

            if valid_loader.dataloader_type != "test":
                batch_images_input, batch_label_input = valid_loader[idx]
            else:
                batch_images_input = valid_loader[idx]

            output = model(batch_images_input)

            if use_multiinput_architecture is False:
                if len(valid_loader.data_list)%batch_size ==0:
                    temp_image = torch.max(output, 1)[1].detach().cpu().numpy().astype(np.bool)
                    prediction_array[idx*batch_size:(idx+1)*batch_size] = remove_small_regions(temp_image, 0.02 * np.prod(image_resolution))
                    
                    if valid_loader.dataloader_type != "test":
                        input_mask_array[idx*batch_size:(idx+1)*batch_size] = batch_label_input.detach().cpu().numpy().astype(np.uint8)
                else:
                    if idx == len(valid_loader.data_list)//batch_size:
                        temp_image = torch.max(output, 1)[1].detach().cpu().numpy().astype(np.bool)
                        prediction_array[idx*batch_size:] = remove_small_regions(temp_image, 0.02 * np.prod(image_resolution))
                        
                        if valid_loader.dataloader_type != "test":
                            input_mask_array[idx*batch_size:] = batch_label_input.detach().cpu().numpy().astype(np.uint8)
                    else:
                        temp_image = torch.max(output, 1)[1].detach().cpu().numpy().astype(np.bool)
                        prediction_array[idx*batch_size:(idx+1)*batch_size] = remove_small_regions(temp_image, 0.02 * np.prod(image_resolution))
                        
                        if valid_loader.dataloader_type != "test":
                            input_mask_array[idx*batch_size:(idx+1)*batch_size] = batch_label_input.detach().cpu().numpy().astype(np.uint8)
            else:
                if len(valid_loader.data_list)%batch_size ==0:
                    temp_image = torch.max(output[-1], 1)[1].detach().cpu().numpy().astype(np.bool)
                    prediction_array[idx*batch_size:(idx+1)*batch_size] = remove_small_regions(temp_image, 0.02 * np.prod(image_resolution))
                    
                    if valid_loader.dataloader_type != "test":
                        input_mask_array[idx*batch_size:(idx+1)*batch_size] = batch_label_input.detach().cpu().numpy().astype(np.uint8)
                else:
                    if idx == len(valid_loader.data_list)//batch_size:
                        temp_image = torch.max(output[-1], 1)[1].detach().cpu().numpy().astype(np.bool)
                        prediction_array[idx*batch_size:] = remove_small_regions(temp_image, 0.02 * np.prod(image_resolution))
                        
                        if valid_loader.dataloader_type != "test":
                            input_mask_array[idx*batch_size:] = batch_label_input.detach().cpu().numpy().astype(np.uint8)
                    else:
                        temp_image = torch.max(output[-1], 1)[1].detach().cpu().numpy().astype(np.bool)
                        prediction_array[idx*batch_size:(idx+1)*batch_size] = remove_small_regions(temp_image, 0.02 * np.prod(image_resolution))
                        
                        if valid_loader.dataloader_type != "test":
                            input_mask_array[idx*batch_size:(idx+1)*batch_size] = batch_label_input.detach().cpu().numpy().astype(np.uint8)

            if valid_loader.dataloader_type != "test":

                if use_multiinput_architecture is False:
                    loss = losses.dice_loss(output, batch_label_input, exclude_0)
                else:
                    loss = losses.dice_loss(output[-1], batch_label_input, exclude_0)

                valid_count += batch_images_input.shape[0]

                if use_multiinput_architecture is False:
                    score = losses.dice_score(output, batch_label_input, exclude_0)
                else:
                    score = losses.dice_score(output[-1], batch_label_input, exclude_0)


                valid_dice_score += (score.sum().item() / score.size(0)) * batch_images_input.shape[0]  

                if 0 in classes and 1 in classes and len(classes) == 2 and len(clubbed) == 0:
                    valid_dice_score_0 += score[0].item() * batch_images_input.shape[0]  
                    valid_dice_score_1 += score[1].item() * batch_images_input.shape[0]

    if valid_loader.dataloader_type != "test":
        valid_dice_score = valid_dice_score/valid_count

        if 0 in classes:
            valid_dice_score_0 = valid_dice_score_0/valid_count
        if 1 in classes:
            valid_dice_score_1 = valid_dice_score_1/valid_count

    if generate_mask is True:

        for i,files in enumerate(valid_loader.data_list):
            temp_mask = prediction_array[i].astype(int)
            temp_mask = ndimage.zoom(temp_mask, np.asarray(valid_loader.original_size_array[files]) / np.asarray(temp_mask.shape), order=0)
            temp_mask = sitk.GetImageFromArray(temp_mask)
            temp_mask = sitk.Cast(temp_mask, sitk.sitkUInt8)

            resampler = sitk.ResampleImageFilter()
            resampler.SetReferenceImage(temp_mask)
            resampler.SetOutputSpacing(valid_loader.spacing[files])
            resampler.SetSize(valid_loader.size_[files])
            resampler.SetInterpolator(sitk.sitkNearestNeighbor)
            temp_mask = resampler.Execute(temp_mask)
            temp_mask.SetOrigin(valid_loader.origin[files])

            #temp_name = ''
            #for j in range(len(files.split('.'))-1):
                #if files.split('.')[j] != 'nii':
                    #temp_name = temp_name + files.split('.')[j] + '.'
            temp_name = files[:-4]
            sitk.WriteImage(temp_mask, save_path + '/Pred_mask_' + temp_name + '.nii.gz')

            # sitk.WriteImage(temp_mask, save_path + '/Pred_mask_' + files.split('.')[0] + '.nii.gz')

            # io.imsave(save_path + '/Pred_mask_' + files.split('.')[0] + '.png', temp_mask)

            # if valid_loader.dataloader_type != "test":
            #   temp_img_plus_mask = prediction_array[i].astype(int) + input_mask_array[i]*2
            #   temp_img_plus_mask = ndimage.zoom(temp_img_plus_mask, np.asarray(valid_loader.original_size_array[files]) / np.asarray(temp_img_plus_mask.shape), order=0)
                
            #   temp_img_plus_mask = sitk.GetImageFromArray(temp_img_plus_mask)

            #   resampler = sitk.ResampleImageFilter()
            #   resampler.SetReferenceImage(temp_img_plus_mask)
            #   resampler.SetOutputSpacing(valid_loader.spacing[files])
            #   resampler.SetSize(valid_loader.size_[files])
            #   resampler.SetInterpolator(sitk.sitkNearestNeighbor)
            #   temp_img_plus_mask = resampler.Execute(temp_img_plus_mask)
            #   temp_img_plus_mask.SetOrigin(valid_loader.origin[files])

            #   sitk.WriteImage(temp_img_plus_mask, save_path + '/Pred_merged_mask_' + files.split('.')[0] + '.nii.gz')

    if valid_loader.dataloader_type != "test":

        if 0 in classes and 1 in classes and len(clubbed) == 0:
            print('Valid Dice Score: ', valid_dice_score, ' Valid Dice Score 0: ', valid_dice_score_0, ' Valid Dice Score 1: ', valid_dice_score_1)

if __name__ == '__main__':
    main()

