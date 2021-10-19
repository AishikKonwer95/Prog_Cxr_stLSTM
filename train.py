from __future__ import print_function
import sys
# sys.path.append('../lib')
import datetime
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torch.optim.lr_scheduler import StepLR
import torch.optim as optim
import numpy as np
import pickle
import model
import dataloader
import matplotlib.pyplot as plt
import argparse
import os
import shutil
import requests
import io
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from sklearn import metrics
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2


parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', type=int)
parser.add_argument('--LR', type=float)
parser.add_argument('--restart_training', type=str)
parser.add_argument('--datapath', type=str)
parser.add_argument('--checkpoint_plot_dir', type=str)
parser.add_argument('--model_path', type=str)
args = parser.parse_args()

# parameters and hyper params
# can be set by terminal
batch_size = args.batch_size
LR = args.LR
data_path = args.datapath
# set manually 

experiment_dir = args.checkpoint_plot_dir

if not os.path.isdir(experiment_dir):
    os.mkdir(experiment_dir)

model_checkpoint_dir = os.path.join(experiment_dir,'checkpoint_dir')
plot_main_dir = os.path.join(experiment_dir,'Plots')
plots_dir = os.path.join(experiment_dir,'Plots/fig')
plots_pickle_dir = os.path.join(experiment_dir,'Plots/pickle') 

train_epoch = 1000

input_res = 512
resolution = 128
n_tiles = 40

print_every = 10  # idx
save_every = 10000     # epoch
valid_every = 1    # epoch

freeze = False
gamma_ = 1
sampling = 'none'
use_last_data = False
# scheduler_step_size = 1
# scheduler_gamma = 0.9

crossvalidation = 5
classes = 3

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

def seed_everything(seed):
    # random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

seed_everything(0)

train_on_gpu = torch.cuda.is_available()

if not train_on_gpu:
    print('CUDA is not available. Training on CPU...')
else:
    print("CUDA is available. Training on GPU...")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def main():
    # initialise the models

    net = model.final_model(c_out=classes, n_tiles=n_tiles, tile_size=resolution).double().to(device)

    """Dataset"""

    train_loader_ = dataloader.Data_Loader(data_path, sampling=sampling, dataloader_type='train', crossvalidation=crossvalidation, classes=classes, transform=train_transform, n_tiles=n_tiles)
    valid_loader_ = dataloader.Data_Loader(data_path, sampling=sampling, dataloader_type='valid', crossvalidation=crossvalidation, classes=classes, transform=valid_transform, n_tiles=n_tiles)

    criterion = nn.CrossEntropyLoss(weight=torch.Tensor([1,gamma_]).double().to(device))
    # criterion = nn.CrossEntropyLoss()

    train_loader = DataLoader(train_loader_, batch_size=batch_size, shuffle=True, num_workers=1, drop_last=True)
    valid_loader = DataLoader(valid_loader_, batch_size=batch_size, shuffle=False, num_workers=1)

    loss_list_train_epoch = [None] 
    accuracy_list_train_epoch = [None]
    epoch_data_list = [None]

    loss_list_train = [None]
    accuracy_list_train = [None]
    index_data_list = [None]

    loss_list_validation = [None]

    accuracy_list_validation = [None]
    accuracy_list_validation_0 = [None]
    accuracy_list_validation_1 = [None]

    loss_list_validation_index = [None]
    
    epoch_old = 0

    if args.restart_training == 'false':
        checkpoint = torch.load(args.model_path, map_location=lambda storage, loc: storage)
        pretrained_dict = checkpoint['model_state_dict']
        model_dict = net.state_dict()

        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict) 
        net.load_state_dict(model_dict)

        # net.load_state_dict(checkpoint['model_state_dict'])
        print('loaded')


    if use_last_data is True:
        if args.restart_training == 'false':
            checkpoint = torch.load(args.model_path, map_location=lambda storage, loc: storage)
            epoch_old = checkpoint['epochs']     
            
            if checkpoint['train_loss_list_epoch'][-1] == None:
                loss_list_train = [None]
                index_data_list = [None]
                accuracy_list_train = [None]
                accuracy_list_train_epoch = [None]
                loss_list_train_epoch = [None]
                epoch_data_list = [None] 

            else:
                loss_list_train = checkpoint['train_loss_list']
                index_data_list = checkpoint['train_loss_index']
                accuracy_list_train = checkpoint['train_accuracy_list']
                accuracy_list_train_epoch = checkpoint['train_accuracy_list_epoch']
                loss_list_train_epoch = checkpoint['train_loss_list_epoch']
                epoch_data_list = checkpoint['train_loss_index_epoch']


            if checkpoint['valid_loss_list'][-1] == None:
                loss_list_validation = [None]
                loss_list_validation_index = [None]

                accuracy_list_validation = [None]
                accuracy_list_validation_0 = [None]
                accuracy_list_validation_1 = [None]

            else:

                loss_list_validation = checkpoint['valid_loss_list']              
                loss_list_validation_index = checkpoint['valid_loss_index']
                accuracy_list_validation = checkpoint['valid_accuracy_list']  

                accuracy_list_validation = checkpoint['valid_accuracy_list']
                best_accuracy = np.max(accuracy_list_validation[1:])
                accuracy_list_validation_0 = checkpoint['valid_accuracy_list_0']
                accuracy_list_validation_1 = checkpoint['valid_accuracy_list_1']


    if len(train_loader_.data_list)%batch_size ==0:
        total_idx_train = len(train_loader_.data_list)//batch_size
    else:
        total_idx_train = len(train_loader_.data_list)//batch_size + 1

    if len(valid_loader_.data_list)%batch_size ==0:
        total_idx_valid = len(valid_loader_.data_list)//batch_size
    else:
        total_idx_valid = len(valid_loader_.data_list)//batch_size + 1
    

    optimizer = optim.Adam(net.parameters(), lr=LR, betas=(0.9, 0.999))
    # optimizer = optim.SGD(net.parameters(), lr=LR, momentum=0.9)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer, max_lr=LR, div_factor=10, pct_start=1 / train_epoch, steps_per_epoch=len(train_loader), epochs=train_epoch)


    for epoch in range(epoch_old, train_epoch):

        epoch_loss = 0.0
        epoch_accuracy = 0.0
        train_count = 0

        running_loss = 0.0
        running_accuracy = 0.0
        running_train_count = 0
        
        net.train()  
        for idx, batch_images_label_list in enumerate(tqdm(train_loader)):

            optimizer.zero_grad()
            batch_images_input = batch_images_label_list[0].to(device)
            batch_label_input = batch_images_label_list[1].to(device)
            batch_final_label_input = batch_images_label_list[2].to(device)

            output = net(batch_images_input)
            loss = criterion(output, batch_label_input)
            
            loss.backward()
            optimizer.step()
            scheduler.step()

            epoch_loss += loss.item()*batch_images_input.shape[0]
            train_count += batch_images_input.shape[0]

            correct = np.where(np.argmax(output.cpu().detach().numpy(), axis=1)==batch_final_label_input.cpu().detach().numpy())[0].shape[0]
            epoch_accuracy += correct                
            # running_accuracy += correct


        loss_list_train_epoch.append(epoch_loss/train_count)
        epoch_data_list.append(epoch+1)
        accuracy_list_train_epoch.append(epoch_accuracy/train_count)

        print('Epoch %d Training Loss: %.3f Accuracy: %.3f' % (epoch + 1, loss_list_train_epoch[-1], accuracy_list_train_epoch[-1]),' Time:',datetime.datetime.now() )

        plt.plot(epoch_data_list[1:], loss_list_train_epoch[1:], label = "Training", color='red', marker='o', markerfacecolor='yellow', markersize=5)
        plt.xlabel('Epoch') 
        plt.ylabel('Training Loss') 
        plt.savefig(plots_dir + '/train_plot_loss.png')
        plt.clf()

        plt.plot(epoch_data_list[1:], accuracy_list_train_epoch[1:], label = "Training", color='red', marker='o', markerfacecolor='yellow', markersize=5)
        plt.xlabel('Epoch') 
        plt.ylabel('Training Accuracy') 
        plt.savefig(plots_dir + '/train_plot_accuracy.png')
        plt.clf()
        

        if (epoch+1) % save_every == 0:
            print('Saving model at %d epoch' % (epoch + 1),' Time:',datetime.datetime.now())  # save every save_every mini_batch of data
            torch.save({
            'epochs': epoch+1,
            'batchsize': batch_size,
            'train_loss_list': loss_list_train,
            'train_loss_list_epoch':loss_list_train_epoch,
            'train_accuracy_list': accuracy_list_train,
            'train_accuracy_list_epoch': accuracy_list_train_epoch,
            'train_loss_index': index_data_list,
            'train_loss_index_epoch': epoch_data_list,
            'valid_loss_list': loss_list_validation,
            'valid_accuracy_list': accuracy_list_validation,
            'valid_accuracy_list_0': accuracy_list_validation_0,
            'valid_accuracy_list_1': accuracy_list_validation_1,
            'valid_loss_index': loss_list_validation_index,
            'model_state_dict': net.state_dict(),
            }, model_checkpoint_dir + '/model_%d.pth' % (epoch + 1))


        if (epoch+1) % valid_every == 0:
            net.eval() 
            optimizer.zero_grad() 
            total_loss_valid = 0.0
            False_positive = 0.0
            False_negative = 0.0
            valid_accuracy_0=0.0
            valid_accuracy_1=0.0
            valid_count_0 = 0
            valid_count_1 = 0
            # prediction_raw_array = np.zeros((len(valid_loader.data_list),2))
            ground_truth_array = np.zeros(len(valid_loader_.data_list))
         
            for idx, batch_images_label_list in enumerate(tqdm(valid_loader)):
                with torch.no_grad():

                    batch_images_input = batch_images_label_list[0].to(device)
                    batch_label_input = batch_images_label_list[1].to(device)

                    output = net(batch_images_input)            

                    loss = criterion(output, batch_label_input)

                    total_loss_valid += loss.item()*batch_images_input.shape[0]  

                    zero_label_index = np.where(batch_label_input.cpu().detach().numpy() == 0)[0]
                    one_label_index = np.where(batch_label_input.cpu().detach().numpy() == 1)[0]  

                    valid_count_0 += zero_label_index.shape[0]
                    valid_count_1 += one_label_index.shape[0]

                    valid_accuracy_0 += np.where(np.argmax(output.cpu().detach().numpy()[zero_label_index], axis=1)==batch_label_input.cpu().detach().numpy()[zero_label_index])[0].shape[0]
                    valid_accuracy_1 += np.where(np.argmax(output.cpu().detach().numpy()[one_label_index], axis=1)==batch_label_input.cpu().detach().numpy()[one_label_index])[0].shape[0]

                    False_positive += np.where(np.argmax(output.cpu().detach().numpy()[zero_label_index], axis=1)!=batch_label_input.cpu().detach().numpy()[zero_label_index])[0].shape[0]
                    False_negative += np.where(np.argmax(output.cpu().detach().numpy()[one_label_index], axis=1)!=batch_label_input.cpu().detach().numpy()[one_label_index])[0].shape[0]

            precision = valid_accuracy_1/(valid_accuracy_1 + False_positive + 1e-5)
            recall = valid_accuracy_1/(valid_accuracy_1 + False_negative + 1e-5)
            TNR = valid_accuracy_0/(valid_accuracy_0 + False_positive + 1e-5)   #specificity
            FPR  = 1 - TNR
            TPR = valid_accuracy_1/(valid_accuracy_1 + False_negative + 1e-5)  # sensitivity/recall/TPR
            F1_score = 2*precision*recall/(precision + recall + 1e-5)

            valid_accuracy_0 = valid_accuracy_0/valid_count_0
            valid_accuracy_1 = valid_accuracy_1/valid_count_1
            valid_accuracy = (valid_accuracy_0 + valid_accuracy_1)/2

            loss_list_validation.append(total_loss_valid / (valid_count_0 + valid_count_1))
            loss_list_validation_index.append(epoch+1)

            accuracy_list_validation.append(valid_accuracy)
            accuracy_list_validation_0.append(valid_accuracy_0)
            accuracy_list_validation_1.append(valid_accuracy_1)


            print('Epoch %d Valid Loss: %.3f' % (epoch + 1, loss_list_validation[-1]),' Time:',datetime.datetime.now() )
            print('Valid Accuracy: ', valid_accuracy, ' Valid Accuracy 0: ', valid_accuracy_0,' Valid Accuracy 1: ', valid_accuracy_1)
            # print('Presicion: ',precision, ' Recall: ',recall, ' F1 Score: ', F1_score, ' TPR: ', TPR, ' TNR: ', TNR)       

            # if epoch >= 2:
            plt.plot(loss_list_validation_index[1:], loss_list_validation[1:], label = "Validation", color='red', marker='o', markerfacecolor='yellow', markersize=5)
            plt.xlabel('Epoch') 
            plt.ylabel('Validation Loss') 
            plt.savefig(plots_dir + '/valid_loss_plot.png')
            plt.clf()


            plt.plot(loss_list_validation_index[1:], accuracy_list_validation[1:], label = "Validation", color='red', marker='o', markerfacecolor='yellow', markersize=5)
            plt.xlabel('Epoch') 
            plt.ylabel('Validation Accuracy') 
            plt.savefig(plots_dir + '/valid_accuracy_plot.png')
            plt.clf()
            plt.plot(loss_list_validation_index[1:], accuracy_list_validation_0[1:], label = "Validation", color='red', marker='o', markerfacecolor='yellow', markersize=5)
            plt.xlabel('Epoch') 
            plt.ylabel('Validation Accuracy') 
            plt.savefig(plots_dir + '/valid_accuracy_plot_0.png')
            plt.clf()
            plt.plot(loss_list_validation_index[1:], accuracy_list_validation_1[1:], label = "Validation", color='red', marker='o', markerfacecolor='yellow', markersize=5)
            plt.xlabel('Epoch') 
            plt.ylabel('Validation Accuracy') 
            plt.savefig(plots_dir + '/valid_accuracy_plot_1.png')
            plt.clf()

            if len(loss_list_validation) >= 3:
                if accuracy_list_validation[-1] > best_accuracy:
                    best_accuracy = accuracy_list_validation[-1]
                    torch.save({
                    'epochs': epoch+1,
                    'batchsize': batch_size,
                    'train_loss_list': loss_list_train,
                    'train_loss_list_epoch':loss_list_train_epoch,
                    'train_accuracy_list': accuracy_list_train,
                    'train_accuracy_list_epoch': accuracy_list_train_epoch,
                    'train_loss_index': index_data_list,
                    'train_loss_index_epoch': epoch_data_list,
                    'valid_loss_list': loss_list_validation,
                    'valid_accuracy_list': accuracy_list_validation,
                    'valid_accuracy_list_0': accuracy_list_validation_0,
                    'valid_accuracy_list_1': accuracy_list_validation_1,
                    'valid_loss_index': loss_list_validation_index,
                    'model_state_dict': net.state_dict(),
                    }, model_checkpoint_dir + '/best_model.pth')

            else:
                best_accuracy = accuracy_list_validation[-1]
                torch.save({
                'epochs': epoch+1,
                'batchsize': batch_size,
                'train_loss_list': loss_list_train,
                'train_loss_list_epoch':loss_list_train_epoch,
                'train_accuracy_list': accuracy_list_train,
                'train_accuracy_list_epoch': accuracy_list_train_epoch,
                'train_loss_index': index_data_list,
                'train_loss_index_epoch': epoch_data_list,
                'valid_loss_list': loss_list_validation,
                'valid_accuracy_list': accuracy_list_validation,
                'valid_accuracy_list_0': accuracy_list_validation_0,
                'valid_accuracy_list_1': accuracy_list_validation_1,
                'valid_loss_index': loss_list_validation_index,
                'model_state_dict': net.state_dict(),
                }, model_checkpoint_dir + '/best_model.pth')


                
if __name__ == '__main__':
    main()

