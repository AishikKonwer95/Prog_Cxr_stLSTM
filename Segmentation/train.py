from __future__ import print_function
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
import shutil
import matplotlib.pyplot as plt
from cxr_resunet import UNet
from cxr_attention_resunet import Attention_UNet, init_weights
import losses
import dataloader_cxr

parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', type=int, default = 1)
parser.add_argument('--LR', type=float, default = 0.001)
parser.add_argument('--restart_training', type=str, default = 'true')
parser.add_argument('--datapath', type=str)
parser.add_argument('--checkpoint_plot_dir', type=str)
parser.add_argument('--model_path', type=str)
args = parser.parse_args()

batch_size = args.batch_size
LR = args.LR
data_path = args.datapath
model_path = args.model_path
experiment_dir = args.checkpoint_plot_dir

load_old_lists = False

modeltype = 'resunet'
use_focal = False
use_attention = False
use_multiinput_architecture = True
init_weights_xavier = False
invert = False
remove_wires=False
# lr best working is 0.00001 with schedular (10,0.95)

n_classes = 2
clubbed = []
classes = [0,1]
model_depth = 5
wf = 5

focal_alpha = 0.5
focal_gamma = 1.0

train_epoch = 300
print_every = 50000000
save_every = 1000   # epoch
valid_every = 5  # epoch
image_resolution = [512,512]

gamma0 = 1
gamma1 = 1

# lambda_l1 = 0.0
# lambda_l2 = 0.0
scheduler_step_size = 10
scheduler_gamma = 0.95

if classes[0] != 0:
    exclude_0 = True
else:
    exclude_0 = False

if not os.path.isdir(experiment_dir):
    os.mkdir(experiment_dir)

model_checkpoint_dir = os.path.join(experiment_dir,'checkpoint_dir')
plot_main_dir = os.path.join(experiment_dir,'Plots')
plots_dir = os.path.join(experiment_dir,'Plots/fig')
plots_pickle_dir = os.path.join(experiment_dir,'Plots/pickle') 

if args.restart_training == 'true':
    if os.path.isdir(model_checkpoint_dir):
        shutil.rmtree(model_checkpoint_dir, ignore_errors=True)
    if os.path.isdir(plot_main_dir):
        shutil.rmtree(plot_main_dir, ignore_errors=True)
    
    os.mkdir(model_checkpoint_dir)
    os.mkdir(plot_main_dir)
    os.mkdir(plots_dir)
    os.mkdir(plots_pickle_dir)

elif args.restart_training == 'false':
    
    if not os.path.isdir(model_checkpoint_dir):
        os.mkdir(model_checkpoint_dir)

    if not os.path.isdir(plot_main_dir):
        os.mkdir(plot_main_dir)
    
    if not os.path.isdir(plots_dir):
        os.mkdir(plots_dir)
    
    if not os.path.isdir(plots_pickle_dir):
        os.mkdir(plots_pickle_dir)


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

def main():
    if args.restart_training == 'true':
        if use_multiinput_architecture is False:
            if modeltype == 'unet':
                model = UNet(n_classes=n_classes, padding=True, depth=model_depth, wf=wf, up_mode='upsample', batch_norm=True, residual=False).double().to(device)
            elif modeltype == 'resunet':
                model = UNet(n_classes=n_classes, padding=True, depth=model_depth, wf=wf, up_mode='upsample', batch_norm=True, residual=True).double().to(device)
    
        elif use_multiinput_architecture is True:
            if modeltype == 'unet':
                model = Attention_UNet(n_classes=n_classes, padding=True, up_mode='upconv', batch_norm=True, residual=False, wf=wf, use_attention=use_attention).double().to(device)
            elif modeltype == 'resunet':
                model = Attention_UNet(n_classes=n_classes, padding=True, up_mode='upconv', batch_norm=True, residual=True, wf=wf, use_attention=use_attention).double().to(device)
            
        if init_weights_xavier is True:
            init_weights(model)

    else:
        if use_multiinput_architecture is False:
            if modeltype == 'unet':
                model = UNet(n_classes=n_classes, padding=True, depth=model_depth, wf=wf, up_mode='upsample', batch_norm=True, residual=False).double().to(device)
            elif modeltype == 'resunet':
                model = UNet(n_classes=n_classes, padding=True, depth=model_depth, wf=wf, up_mode='upsample', batch_norm=True, residual=True).double().to(device)

            checkpoint = torch.load(args.model_path, map_location=lambda storage, loc: storage)
            pretrained_dict = checkpoint['model_state_dict']

            model_dict = model.state_dict()
            # 1. filter out unnecessary keys
            pretrained_dict = {k: v for k, v in pretrained_dict.items() if k not in ['last.weight', 'last.bias']}
            # 2. overwrite entries in the existing state dict
            model_dict.update(pretrained_dict) 
            # 3. load the new state dict
            model.load_state_dict(model_dict)

        elif use_multiinput_architecture is True:
            if modeltype == 'unet':
                model = Attention_UNet(n_classes=n_classes, padding=True, up_mode='upconv', batch_norm=True, residual=False, wf=wf, use_attention=use_attention).double().to(device)
            elif modeltype == 'resunet':
                model = Attention_UNet(n_classes=n_classes, padding=True, up_mode='upconv', batch_norm=True, residual=True, wf=wf, use_attention=use_attention).double().to(device)
            
            checkpoint = torch.load(args.model_path, map_location=lambda storage, loc: storage)
            model.load_state_dict(checkpoint['model_state_dict'])

    train_loader = dataloader_cxr.DataLoader(data_path, dataloader_type='train', batchsize=batch_size, device=device, image_resolution=image_resolution, invert=invert, remove_wires=remove_wires)
    print('trainloader loaded')
    valid_loader = dataloader_cxr.DataLoader(data_path, dataloader_type='valid', batchsize=batch_size, device=device, image_resolution=image_resolution, invert=invert, remove_wires=remove_wires)
    print('validloader loaded')
    

    loss_list_train_epoch = [None] 
    dice_score_list_train_epoch = [None]
    epoch_data_list = [None]
    loss_list_train = [None]
    dice_score_list_train = [None]
    index_data_list = [None]

    loss_list_validation = [None]
    loss_list_validation_index = [None]
    dice_score_list_validation = [None]
    dice_score_list_validation_0 = [None]
    dice_score_list_validation_1 = [None]
    
    epoch_old = 0
    if load_old_lists == True:
      if args.restart_training == 'false':
          epoch_old = checkpoint['epochs']     
          
          if checkpoint['train_loss_list_epoch'][-1] == None:
              loss_list_train = [None]
              index_data_list = [None]
              dice_score_list_train = [None]
              dice_score_list_train_epoch = [None]
              loss_list_train_epoch = [None]
              epoch_data_list = [None] 

          else:
              loss_list_train = checkpoint['train_loss_list']
              index_data_list = checkpoint['train_loss_index']
              dice_score_list_train = checkpoint['train_dice_score_list']
              dice_score_list_train_epoch = checkpoint['train_dice_score_list_epoch']
              loss_list_train_epoch = checkpoint['train_loss_list_epoch']
              epoch_data_list = checkpoint['train_loss_index_epoch']
          
          if checkpoint['valid_loss_list'][-1] == None:
              loss_list_validation = [None]
              loss_list_validation_index = [None]

              dice_score_list_validation = [None]
              dice_score_list_validation_0 = [None]
              dice_score_list_validation_1 = [None]

          else:
              loss_list_validation = checkpoint['valid_loss_list']              
              loss_list_validation_index = checkpoint['valid_loss_index']
              dice_score_list_validation = checkpoint['valid_dice_score_list']  
              dice_score_list_validation_0 = checkpoint['valid_dice_score_list_0']
              dice_score_list_validation_1 = checkpoint['valid_dice_score_list_1']
              best_model_accuracy = np.max(dice_score_list_validation[1:])

    if len(train_loader.data_list)%batch_size ==0:
        total_idx_train = len(train_loader.data_list)//batch_size
    else:
        total_idx_train = len(train_loader.data_list)//batch_size + 1

    if len(valid_loader.data_list)%batch_size ==0:
        total_idx_valid = len(valid_loader.data_list)//batch_size
    else:
        total_idx_valid = len(valid_loader.data_list)//batch_size + 1
    
    if epoch_old != 0:
        power_factor = epoch_old//scheduler_step_size     
        LR_ = LR*(scheduler_gamma**power_factor)
    else:
        LR_ = LR

    LR_ = LR
    optimizer = optim.Adam(model.parameters(), lr=LR_)
    # optimizer = optim.SGD(model.parameters(), lr=LR_, momentum=0.9)
    scheduler = StepLR(optimizer, step_size=scheduler_step_size, gamma=scheduler_gamma)

    for epoch in range(epoch_old, train_epoch):

        if (epoch+1)%10 == 0:
            # l1 = l1*1.0
            # l2 = l2*1.0
            scheduler.step()
        

        running_loss = 0.0
        running_dice_score = 0.0
        running_train_count = 0
        
        epoch_loss = 0.0
        epoch_dice_score = 0.0
        train_count = 0

        model.train()  
        for idx in range(total_idx_train):

            optimizer.zero_grad()

            batch_images_input, batch_label_input = train_loader[idx]
            output = model(batch_images_input)

            if use_multiinput_architecture is False:
                if use_focal is False:
                    loss = losses.dice_loss(output, batch_label_input, exclude_0, weights=torch.Tensor([gamma0,gamma1]).double().to(device))
                else:
                    loss = losses.focal_tversky_loss(output, batch_label_input, exclude_0, weights=torch.Tensor([gamma0,gamma1]).double().to(device), alpha=focal_alpha, focal_gamma=focal_gamma)

            elif use_multiinput_architecture is True:
                loss = losses.focal_tversky_loss_deep_supervised(output, batch_label_input, exclude_0, weights=torch.Tensor([gamma0,gamma1]).double().to(device), alpha=focal_alpha, focal_gamma=focal_gamma)
              
            # l1 = 0
            # l2 = 0
            # for p in model.parameters():
            #     l1 = l1 + p.abs().sum()
            #     l2 = l2 + (p**2).sum()
            # loss = loss + lambda_l1 * l1 + lambda_l2 * l2

            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()*batch_images_input.shape[0]
            running_train_count += batch_images_input.shape[0]

            if use_multiinput_architecture is False:
                score = losses.dice_score(output, batch_label_input, exclude_0)
            else:
                score = losses.dice_score(output[-1], batch_label_input, exclude_0)

            running_dice_score += (score.sum().item() / score.size(0)) * batch_images_input.shape[0]
            epoch_dice_score += (score.sum().item() / score.size(0)) * batch_images_input.shape[0]

            epoch_loss += loss.item()*batch_images_input.shape[0]
            train_count += batch_images_input.shape[0]  

            if (idx+1) % print_every == 0:  # print every print_every folder of data
                print('Epoch Data Completed:%d Loss:%.6f Dice Score:%.6f' %
                        ((idx+1)*batch_size, running_loss / running_train_count, running_dice_score / running_train_count),' Time:',datetime.datetime.now())
                    
                index_data_list.append(epoch*total_idx_train*batch_size+(idx+1)*batch_size)
                loss_list_train.append(running_loss / running_train_count)
                dice_score_list_train.append(running_dice_score / running_train_count)

                running_loss = 0.0
                running_dice_score = 0.0
                running_train_count = 0
                    
                plt.plot(index_data_list[1:], loss_list_train[1:], label = "Training", color='green', marker='o', markerfacecolor='blue', markersize=5)
                plt.xlabel('Slices encountered') 
                plt.ylabel('Loss') 
                plt.savefig(plots_dir + '/train_plot_loss_running.png')
                plt.clf()

                plt.plot(index_data_list[1:], dice_score_list_train[1:], label = "Training", color='green', marker='o', markerfacecolor='blue', markersize=5)
                plt.xlabel('Slices encountered') 
                plt.ylabel('Dice Score') 
                plt.savefig(plots_dir + '/train_plot_dice_score_running.png')
                plt.clf()
                
                training_pickle = open(plots_pickle_dir + "/loss_list_train_running.npy",'wb')
                pickle.dump(loss_list_train,training_pickle)
                training_pickle.close()

                training_pickle = open(plots_pickle_dir + "/index_list_train_running.npy",'wb')
                pickle.dump(index_data_list,training_pickle)
                training_pickle.close()

                training_pickle = open(plots_pickle_dir + "/dice_score_list_train_running.npy",'wb')
                pickle.dump(dice_score_list_train,training_pickle)
                training_pickle.close()
            
        loss_list_train_epoch.append(epoch_loss/train_count)
        epoch_data_list.append(epoch+1)
        dice_score_list_train_epoch.append(epoch_dice_score/train_count)

        print('Epoch %d Training Loss: %.3f Dice Score: %.3f' % (epoch + 1, loss_list_train_epoch[-1], dice_score_list_train_epoch[-1]),' Time:',datetime.datetime.now() )

        plt.plot(epoch_data_list[1:], loss_list_train_epoch[1:], label = "Training", color='red', marker='o', markerfacecolor='yellow', markersize=5)
        plt.xlabel('Epoch') 
        plt.ylabel('Training Loss') 
        plt.savefig(plots_dir + '/train_loss_plot.png')
        plt.clf()

        plt.plot(epoch_data_list[1:], dice_score_list_train_epoch[1:], label = "Training", color='red', marker='o', markerfacecolor='yellow', markersize=5)
        plt.xlabel('Epoch') 
        plt.ylabel('Training Dice Score') 
        plt.savefig(plots_dir + '/train_dice_score_plot.png')
        plt.clf()
        
        # training_pickle = open(plots_pickle_dir + "/loss_list_train.npy",'wb')
        # pickle.dump(loss_list_train_epoch,training_pickle)
        # training_pickle.close()

        # training_pickle = open(plots_pickle_dir + "/epoch_list_train.npy",'wb')
        # pickle.dump(epoch_data_list,training_pickle)
        # training_pickle.close()

        # training_pickle = open(plots_pickle_dir + "/dice_score_list_train_epoch.npy",'wb')
        # pickle.dump(dice_score_list_train_epoch,training_pickle)
        # training_pickle.close()


        if (epoch+1) % save_every == 0:
            print('Saving model at %d epoch' % (epoch + 1),' Time:',datetime.datetime.now())  # save every save_every mini_batch of data
            torch.save({
            'epochs': epoch+1,
            'batchsize': batch_size,
            'train_loss_list': loss_list_train,
            'train_loss_list_epoch':loss_list_train_epoch,
            'train_dice_score_list': dice_score_list_train,
            'train_dice_score_list_epoch': dice_score_list_train_epoch,
            'train_loss_index': index_data_list,
            'train_loss_index_epoch': epoch_data_list,
            'valid_loss_list': loss_list_validation,
            'valid_dice_score_list': dice_score_list_validation,
            'valid_dice_score_list_0': dice_score_list_validation_0,
            'valid_dice_score_list_1': dice_score_list_validation_1,
            'valid_loss_index': loss_list_validation_index,
            'model_state_dict': model.state_dict(),
            }, model_checkpoint_dir + '/model_%d.pth' % (epoch + 1))
            

        if (epoch+1) % valid_every == 0:
            model.eval() 
            optimizer.zero_grad() 

            valid_count = 0
            total_loss_valid = 0.0
            valid_dice_score = 0.0
            if 0 in classes:
              valid_dice_score_0 = 0.0
            if 1 in classes:
              valid_dice_score_1 = 0.0

            for idx in range(total_idx_valid):
                with torch.no_grad():

                    batch_images_input, batch_label_input = valid_loader[idx]

                    output = model(batch_images_input)
                   
                    if use_multiinput_architecture is False:
                        loss = losses.dice_loss(output, batch_label_input, exclude_0)
                    else:
                        loss = losses.dice_loss(output[-1], batch_label_input, exclude_0)

                    total_loss_valid += loss.item()*batch_images_input.shape[0]  
                    valid_count += batch_images_input.shape[0]

                    if use_multiinput_architecture is False:
                        score = losses.dice_score(output, batch_label_input, exclude_0)
                    else:
                        score = losses.dice_score(output[-1], batch_label_input, exclude_0)

                    valid_dice_score += (score.sum().item() / score.size(0)) * batch_images_input.shape[0]  

                    if 0 in classes and 1 in classes and len(classes) == 2 and len(clubbed) == 0:
                        valid_dice_score_0 += score[0].item() * batch_images_input.shape[0]  
                        valid_dice_score_1 += score[1].item() * batch_images_input.shape[0]
                    elif 1 in classes and len(classes)==1:
                        valid_dice_score_1 += score[0].item() * batch_images_input.shape[0]

            loss_list_validation.append(total_loss_valid/valid_count)
            dice_score_list_validation.append(valid_dice_score/valid_count)

            if 0 in classes:
                dice_score_list_validation_0.append(valid_dice_score_0/valid_count)
            if 1 in classes:      
                dice_score_list_validation_1.append(valid_dice_score_1/valid_count)

            loss_list_validation_index.append(epoch+1)

            print('Epoch %d Valid Loss: %.3f' % (epoch + 1, loss_list_validation[-1]),' Time:',datetime.datetime.now() )

            if 0 in classes and 1 in classes and len(classes) == 2 and len(clubbed) == 0:
                print('Valid Dice Score: ', dice_score_list_validation[-1], ' Valid Dice Score 0: ', dice_score_list_validation_0[-1], ' Valid Dice Score 1: ', dice_score_list_validation_1[-1])
            elif 1 in classes and len(classes)==1:
                print('Valid Dice Score: ', dice_score_list_validation[-1], ' Valid Dice Score 1: ', dice_score_list_validation_1[-1])

            plt.plot(loss_list_validation_index[1:], loss_list_validation[1:], label = "Validation", color='red', marker='o', markerfacecolor='yellow', markersize=5)
            plt.xlabel('Epoch') 
            plt.ylabel('Validation Loss') 
            plt.savefig(plots_dir + '/valid_loss_plot.png')
            plt.clf()

            plt.plot(loss_list_validation_index[1:], dice_score_list_validation[1:], label = "Validation", color='red', marker='o', markerfacecolor='yellow', markersize=5)
            plt.xlabel('Epoch') 
            plt.ylabel('Validation Dice Score') 
            plt.savefig(plots_dir + '/valid_dice_score_plot.png')
            plt.clf()

            if 0 in classes:
                plt.plot(loss_list_validation_index[1:], dice_score_list_validation_0[1:], label = "Validation", color='red', marker='o', markerfacecolor='yellow', markersize=5)
                plt.xlabel('Epoch') 
                plt.ylabel('Validation Dice Score') 
                plt.savefig(plots_dir + '/valid_dice_score_0_plot.png')
                plt.clf()

            if 1 in classes:  
                plt.plot(loss_list_validation_index[1:], dice_score_list_validation_1[1:], label = "Validation", color='red', marker='o', markerfacecolor='yellow', markersize=5)
                plt.xlabel('Epoch') 
                plt.ylabel('Validation Dice Score') 
                plt.savefig(plots_dir + '/valid_dice_score_1_plot.png')
                plt.clf()

            # validation_pickle = open(plots_pickle_dir + "/loss_list_validation.npy",'wb')
            # pickle.dump(loss_list_validation,validation_pickle)
            # validation_pickle.close()

            # validation_pickle = open(plots_pickle_dir + "/index_list_validation.npy",'wb')
            # pickle.dump(loss_list_validation_index,validation_pickle)
            # validation_pickle.close()

            # validation_pickle = open(plots_pickle_dir + "/dice_score_list_validation.npy",'wb')
            # pickle.dump(dice_score_list_validation,validation_pickle)
            # validation_pickle.close()

            if len(loss_list_validation) >= 3:
                if dice_score_list_validation[-1] > best_model_accuracy:
                    best_model_accuracy = dice_score_list_validation[-1]
                    torch.save({
                    'epochs': epoch+1,
                    'batchsize': batch_size,
                    'train_loss_list': loss_list_train,
                    'train_loss_list_epoch':loss_list_train_epoch,
                    'train_dice_score_list': dice_score_list_train,
                    'train_dice_score_list_epoch': dice_score_list_train_epoch,
                    'train_loss_index': index_data_list,
                    'train_loss_index_epoch': epoch_data_list,
                    'valid_loss_list': loss_list_validation,
                    'valid_dice_score_list': dice_score_list_validation,
                    'valid_dice_score_list_0': dice_score_list_validation_0,
                    'valid_dice_score_list_1': dice_score_list_validation_1,
                    'valid_loss_index': loss_list_validation_index,
                    'model_state_dict': model.state_dict(),
                    }, model_checkpoint_dir + '/model_best.pth')

            else:
                best_model_accuracy = dice_score_list_validation[-1]
                torch.save({
                'epochs': epoch+1,
                'batchsize': batch_size,
                'train_loss_list': loss_list_train,
                'train_loss_list_epoch':loss_list_train_epoch,
                'train_dice_score_list': dice_score_list_train,
                'train_dice_score_list_epoch': dice_score_list_train_epoch,
                'train_loss_index': index_data_list,
                'train_loss_index_epoch': epoch_data_list,
                'valid_loss_list': loss_list_validation,
                'valid_dice_score_list': dice_score_list_validation,
                'valid_dice_score_list_0': dice_score_list_validation_0,
                'valid_dice_score_list_1': dice_score_list_validation_1,
                'valid_loss_index': loss_list_validation_index,
                'model_state_dict': model.state_dict(),
                }, model_checkpoint_dir + '/model_best.pth')


                
if __name__ == '__main__':
    main()

