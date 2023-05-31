# -*- coding: utf-8 -*-
"""
Created on Fri Apr 29 16:00:43 2022

@author: Chi Ding
"""

import torch
from torch.utils.data import DataLoader
import torch.nn as nn
from utils import *

import scipy.io as io
import numpy as np
import os
import platform

from argparse import ArgumentParser

parser = ArgumentParser(description='Learnable Optimization Algorithms (LOA)')

parser.add_argument('--start_epoch', type=int, default=0, help='epoch number of start training')
parser.add_argument('--end_epoch', type=int, default=200, help='epoch number of end training')
parser.add_argument('--start_phase', type=int, default=3, help='phase number of start training')
parser.add_argument('--end_phase', type=int, default=15, help='phase number of end training')
parser.add_argument('--layer_num', type=int, default=15, help='phase number of LDA-Net')
parser.add_argument('--learning_rate', type=float, default=1e-4, help='learning rate')
parser.add_argument('--group_num', type=int, default=1, help='group number for training')
parser.add_argument('--cs_ratio', type=int, default=25, help='from {1, 4, 10, 25, 40, 50}')
parser.add_argument('--batch_size', type=int, default=64, help='batch size for loading data')
parser.add_argument('--gpu_list', type=str, default='0', help='gpu index')
parser.add_argument('--init', type=bool, default=True, help='initialization True by default')

parser.add_argument('--matrix_dir', type=str, default='sampling_matrix', help='sampling matrix directory')
parser.add_argument('--model_dir', type=str, default='model', help='trained or pre-trained model directory')
parser.add_argument('--data_dir', type=str, default='data', help='training data directory')
parser.add_argument('--log_dir', type=str, default='log', help='log directory')

args = parser.parse_args()

#%% experiement setup
start_epoch = args.start_epoch
end_epoch = args.end_epoch
start_phase = args.start_phase
end_phase = args.end_phase
learning_rate = args.learning_rate
layer_num = args.layer_num
group_num = args.group_num
cs_ratio = args.cs_ratio
gpu_list = args.gpu_list
batch_size = args.batch_size
init = args.init

PhaseList = np.arange(3,16,2)

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = gpu_list
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

ratio_dict = {1: 10, 4: 43, 10: 109, 25: 272, 30: 327, 40: 436, 50: 545}

n_input = ratio_dict[cs_ratio]
n_output = 1089
nrtrain = 88912   # number of training blocks
#%% Load Data
print('Load Data...')

if os.path.exists('Q_init/Q_init_%d.npy' % cs_ratio):
    Qinit = np.load('Q_init/Q_init_%d.npy' % cs_ratio)
else:
    Qinit = compute_initialization_matrix(cs_ratio)
    np.save('Q_init/Q_init_%d.npy' % cs_ratio, Qinit)

Phi = torch.from_numpy(load_Phi(cs_ratio)).type(torch.FloatTensor).to(device)
Qinit = torch.from_numpy(Qinit).type(torch.FloatTensor).to(device)

Training_data_Name = 'data/Training_Data_Img91.mat'
Training_data = io.loadmat(Training_data_Name)

# labels are transformations of 88912 images, or dictionaries
Training_inputs = Training_data['inputs']

# labels are 88912 original images
Training_labels = Training_data['labels']


#%% dataloader
if (platform.system() == 'Windows'):
    rand_loader = DataLoader(dataset = RandomDataset(Training_labels, nrtrain), 
                             batch_size=batch_size, num_workers=0,shuffle=True)
else:
    rand_loader = DataLoader(dataset=RandomDataset(Training_labels, nrtrain), 
                             batch_size=batch_size, num_workers=8,shuffle=True)
#%% initialize model
model = LDA(layer_num, start_phase)
model = nn.DataParallel(model)
model.to(device)

print_flag = 1   # print parameter number

if print_flag:
    num_count = 0
    for para in model.parameters():
        num_count += 1
        print('Layer %d' % num_count)
        print(para.size())

optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

model_dir = "./%s/LDA_layer_%d_group_%d_ratio_%d_lr_%.4f" % \
    (args.model_dir, layer_num, group_num, cs_ratio, learning_rate)

log_file_name = "./%s/LDA_layer_%d_group_%d_ratio_%d_lr_%.4f.txt" % \
    (args.log_dir, layer_num, group_num, cs_ratio, learning_rate)

# if not start from beginning load pretrained models
if start_epoch > 0:
    # start from checkpoint
    model.load_state_dict(torch.load('%s/net_params_epoch%d_phase%d.pkl' % \
                                     (model_dir, start_epoch, start_phase), 
                                     map_location=device))

if not os.path.exists(model_dir):
    os.makedirs(model_dir)
    
if not os.path.exists(args.log_dir):
    os.makedirs(args.log_dir)
#%% training
for PhaseNo in range(start_phase, end_phase+1, 2):
    # add new phases
    model.module.set_PhaseNo(PhaseNo)
    if PhaseNo == 3:
        end_epoch = 500
    else:
        end_epoch = args.end_epoch
    for epoch_i in range(start_epoch+1, end_epoch+1):
        progress = 0
        for data in rand_loader:
            progress += 1
                
            batch_x = data
            batch_x = batch_x.to(device)
            
            Phix = torch.mm(batch_x, torch.transpose(Phi, 0, 1))
            
            x_output = model(Phix, Phi, Qinit)
            
            # compute and print loss
            loss_all = torch.mean(torch.pow(x_output - batch_x, 2))
            
            # Zero gradients, perform a backward pass, and update the weights.
            optimizer.zero_grad()
            loss_all.backward()
            optimizer.step()
            
            if progress % 20 == 0:
                output_data = "[Phase %02d] [Epoch %02d/%02d] Total Loss: %.4f" % \
                    (PhaseNo, epoch_i, end_epoch, loss_all.item()) \
                    + "\t progress: %02f" % (progress * batch_size/ nrtrain * 100) + "%\n"
                print(output_data)
            
        output_file = open(log_file_name, 'a')
        output_file.write(output_data)
        output_file.close()
    
        if epoch_i % 50 == 0:
            # save the parameters
            torch.save(model.state_dict(), "./%s/net_params_epoch%d_phase%d.pkl" % \
                       (model_dir, epoch_i, PhaseNo))
                
    # after finish training current phases, introduce new phases and start from epoch 0
    start_epoch = 0