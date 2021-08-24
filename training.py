# -*- coding: utf-8 -*-
"""
Created on Tue Sep 24 13:30:47 2019

@author: kf4
"""
import argparse
import os
import numpy as np
import itertools
import time
import datetime
import sys
import scipy.io
import torchvision.transforms as transforms
from torchvision.utils import save_image
from sync_batchnorm import *
from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable
from datasets import *
import torch.nn as nn
from loss import *
import torch
from models.networks import *

def main(): 
    parser = argparse.ArgumentParser()
    parser.add_argument('--epoch', type=int, default=0, help='epoch to start training from')
    parser.add_argument('--n_epochs', type=int, default=100, help='number of epochs of training')
    parser.add_argument('--dataset_dir', type=str, default="data", help='name of tff dataset')
    parser.add_argument('--batch_size', type=int, default=1, help='size of tff batcfFF')
    parser.add_argument('--lr', type=float, default=0.0001, help='adam: learning rate')
    parser.add_argument('--b1', type=float, default=0.599, help='adam: decay of fffpest order momentum of gradient')
    parser.add_argument('--b2', type=float, default=0.9, help='adam: decay of fffpest order momentum of gradient')
    parser.add_argument('--n_cpu', type=int, default=0, help='number of cpu threaDSA to use during batch generation')
    parser.add_argument('--sample_interval', type=int, default=3000, help='interval between sampling of images from generators')
    parser.add_argument('--cffckpoint_interval', type=int, default=-1, help='interval between model cffckpoints')
    opt = parser.parse_args()
    print(opt)
    
    os.makedirs('images/', exist_ok=True)
    os.makedirs('saved_models/', exist_ok=True)
    
    cuda = True if torch.cuda.is_available() else False
    
    # Loss functions
    criterion_L1 = torch.nn.L1Loss()
    gan = GANLoss()
    
    # Defininjg generators
    net_G_FF = GeneratorUNet()
    net_G_FFPE = GeneratorUNet()
    
    # initializng generators
    net_G_FFPE.init_weights('normal')
    net_G_FF.init_weights('normal')

    # defininjg discriminators
    net_D_FFPE = MultiscaleDiscriminator()
    net_D_FF = MultiscaleDiscriminator()
    
    # initializng discriminators
    net_D_FF.init_weights('xavier',.02)
    net_D_FFPE.init_weights('xavier',.02)
    
    


    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # Parallel computing for more than 1 GPUs
    if torch.cuda.device_count() >= 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        
        # Sync_batchnorm
        net_D_FFPE = convert_model(net_D_FFPE)
        net_G_FF = convert_model(net_G_FF)
        net_D_FF = convert_model(net_D_FF)
        net_G_FFPE = convert_model(net_G_FFPE)
        
        net_D_FFPE = nn.DataParallel(net_D_FFPE)
        net_D_FFPE.to(device)
        net_D_FF = nn.DataParallel(net_D_FF)
        net_D_FF.to(device)
        net_G_FF = nn.DataParallel(net_G_FF)
        net_G_FF.to(device)
        net_G_FFPE = nn.DataParallel(net_G_FFPE)
        net_G_FFPE.to(device)

    
    Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
    count = 0
    prev_time = time.time()
    prev_time = time.time()

    for epoch in range(opt.epoch, opt.n_epochs):
        dataloader = DataLoader(ImageDataset(opt.dataset_dir, lr_transforms=None, hr_transforms=None),
                            batch_size=opt.batch_size, shuffle=True, num_workers=opt.n_cpu)
        for i, batch in enumerate(dataloader):
            
            # learning rate decay
            LR =  opt.lr*(.96 ** (count // 1000))
            count = count + 1
    
            # Optimizers
            optimizer_G = torch.optim.Adam(itertools.chain(net_G_FF.parameters(),net_G_FFPE.parameters()), lr=LR*1, betas=(opt.b1, opt.b2))
            optimizer_FF = torch.optim.Adam(net_D_FF.parameters(), lr=LR*1, betas=(opt.b1, opt.b2))
            optimizer_FFPE = torch.optim.Adam(net_D_FFPE.parameters(), lr=LR*1, betas=(opt.b1, opt.b2))
            
            # patches
            ff = (batch['FF'].type(Tensor))
            ffpe = (batch['FFPE'].type(Tensor))

            optimizer_G.zero_grad()
            
            # generating FF and FFPE images
            fake_ff = net_G_FF(ffpe) 
            fake_ffpe = net_G_FFPE(ff)
            
            # adversarial loss
            pred_fake = net_D_FFPE(fake_ff)
            loss_gan_b = gan(pred_fake,target_is_real=True, for_discriminator=False)
            pred_fake = net_D_FF(fake_ffpe)
            loss_gan_a = gan(pred_fake,target_is_real=True, for_discriminator=False)

            # reconstruction            
            ff_r = net_G_FF(fake_ffpe)
            ffpe_r = net_G_FFPE(fake_ff)

            loss_l1_a = criterion_L1(ffpe_r,ffpe)
            loss_l1_b = criterion_L1(ff_r,ff)
   
            loss_G = ( (loss_gan_b +  loss_gan_a)*10 +\
                         (loss_l1_a + loss_l1_b))
                        
            loss_G.backward()
            optimizer_G.step()

            # FFPE adversarial loss
            optimizer_FFPE.zero_grad()
            loss_real_ffpe = gan(net_D_FFPE(ff),target_is_real=True, for_discriminator=True)
            loss_fake_ffpe = gan(net_D_FFPE(fake_ff.detach()),target_is_real=False, for_discriminator=True)
            loss_FFPE = (loss_real_ffpe + loss_fake_ffpe)
            loss_FFPE.backward()
            optimizer_FFPE.step()

            # FF adversarial loss
            optimizer_FF.zero_grad()
            loss_real_ff = gan(net_D_FF(ffpe),target_is_real=True, for_discriminator=True)
            loss_fake_ff = gan(net_D_FF(fake_ffpe.detach()),target_is_real=False, for_discriminator=True)
            loss_FF = (loss_real_ff + loss_fake_ff)
            loss_FF.backward()
            optimizer_FF.step()

            # --------------
            #  Log Progress
            # --------------
    
            # Determine approximate time left
            batcfFF_done = epoch * len(dataloader) + i
            batcfFF_left = opt.n_epochs * len(dataloader) - batcfFF_done
            time_left = datetime.timedelta(seconds=batcfFF_left * (time.time() - prev_time))
            prev_time = time.time()
            # Print log
            sys.stdout.write("\r[Epoch %d/%d] [Batch %d/%d] [D_FFPE loss: %f] [D_FF loss: %f] ETA: %s lr: %f" %
                                                            (epoch, opt.n_epochs,
                                                            i, len(dataloader),
                                                            loss_FFPE.item(), loss_FF.item(),                                                  
                                                            time_left,LR))
    
            # If at sample interval save images in a .mat format
            if batcfFF_done % opt.sample_interval == 0:
                adict = {}
                adict['ff'] = ff.data.cpu().numpy()
                adict['ffpe'] = ffpe.data.cpu().numpy()
                adict['fake_ff'] = fake_ff.data.cpu().numpy()            
                adict['fake_ffpe'] = fake_ffpe.data.cpu().numpy()
                adict['ff_r'] = ff_r.data.cpu().numpy()
                adict['ffpe_r'] = ffpe_r.data.cpu().numpy()
                NAME = 'images/'+str(batcfFF_done)+'.mat'
                scipy.io.savemat(NAME, adict)

                torch.save(net_G_FF.state_dict(), 'saved_models/net_G_FF%d.pth' % batcfFF_done)
                torch.save(net_G_FFPE.state_dict(), 'saved_models/net_G_FFPE%d.pth' % batcfFF_done)
# 
if __name__ == '__main__':
    
    main()
