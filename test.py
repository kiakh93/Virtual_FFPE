import argparse
import os
import numpy as np
import scipy.io
from PIL import Image
import torchvision.transforms as transforms
from torchvision.utils import save_image
from sync_batchnorm import convert_model
from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable
from datasets_test import *
import torch.nn as nn
from loss import *
import torch
from models.networks import *

def main(): 
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_dir', type=str, default="data_test", help='name of the dataset')
    parser.add_argument('--n_cpu', type=int, default=0, help='number of cpu to use during batch generation')
    parser.add_argument('--batch_size', type=int, default=1, help='size of tff batcfFF')
    opt = parser.parse_args()
    print(opt)

    os.makedirs('output/', exist_ok=True)
    
    cuda = True if torch.cuda.is_available() else False

    net_G_FFPE = GeneratorUNet()
#    net_G_FFPE.load_state_dict(torch.load('saved_models/model.pth'))
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if torch.cuda.device_count() >= 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")

        net_G_FFPE = nn.DataParallel(net_G_FFPE)
        net_G_FFPE.to(device)
        net_G_FFPE.load_state_dict(torch.load('saved_models/model.pth'))
        torch.set_grad_enabled(False)
        torch.cuda._lazy_init()
        net_G_FFPE = net_G_FFPE.eval() 

   
    Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor


    dataloader = DataLoader(ImageDataset(opt.dataset_dir, lr_transforms=None, hr_transforms=None),
                        batch_size=opt.batch_size, shuffle=False, num_workers=opt.n_cpu)
    for i, batch in enumerate(dataloader):
        
        ff = (batch['FF'].type(Tensor))
        name = (batch['name'][0])

        # defining a tensor with the same size as input ff image
        fake_ffpe = (ff*0).data.cpu().numpy()
        
        # size of each crop is 800x800 
        fact = 800
        
        _,_,r,c = ff.shape
        rn = np.ceil(r/fact)
        cn = np.ceil(c/fact)
        ff_temp = torch.zeros(1,3,int(rn)*fact+1,int(cn)*fact+1)
        fake_ffpe = torch.zeros(1,3,int(rn)*fact+1,int(cn)*fact+1).data.cpu().numpy()
        ff_temp[:,:,:r,:c] = ff
        # cropping the image and apply the model to each crop
        I = range(0,int(rn)*fact+1,fact)
        J = range(0,int(cn)*fact+1,fact)
        for ii in range(len(I)-1):
            for jj in range(len(J) - 1):
                fake_ffpe[0:1,:,I[ii]:I[ii+1],J[jj]:J[jj+1]] = (net_G_FFPE(ff_temp[0:1,:,I[ii]:I[ii+1],J[jj]:J[jj+1]].type(Tensor)).squeeze(0)*.5+.5).data.cpu().numpy()
    
    im = np.zeros((int(rn)*fact+1,int(cn)*fact+1,3))
    for ind in range(3):
        im[:,:,ind] = fake_ffpe[0:1,ind,:,:]
    im[im>1] = 1
    im[im<0] = 0
    
    im = Image.fromarray(np.uint8(im[:r,:c,:]*255))
    NAME = 'output/'+name
    im.save(NAME)
#                
if __name__ == '__main__':
    
    main()
