import random
import os
import numpy as np
import scipy.io as sio
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms

class ImageDataset(Dataset):
    def __init__(self, root, lr_transforms=None, hr_transforms=None):
#        RAND = random.randint(24,48)
        
        # Frozen samples directory
        self.filesI = os.path.join(root, 'FF')
        self.list_I = os.listdir(self.filesI)
        
        # FFPE samples directory
        self.filesT = os.path.join(root, 'FFPE')
        self.list_T = os.listdir(self.filesT)

        

        
    def __getitem__(self, index):

        I_name = os.path.join(self.filesI,self.list_I[index%len(self.list_I)])
        img = Image.open(I_name)

        T_name = os.path.join(self.filesT,self.list_I[index%len(self.list_T)])
        img_T = Image.open(T_name)

        Transform = [  
                        transforms.ToTensor(),
                        transforms.Normalize((.5,0.5,0.5), (0.5,0.5,0.5))
                        ]
        # Converting the images to tensor and normalizing them
        self.transform = transforms.Compose(Transform)
        

        # choosing 384x384 randomly selected areas from the pathces
        CRx = random.randint(0,115)
        CRy = random.randint(0,115)
        fact = 384
        
        name = self.list_I[index%len(self.list_I)];
        img = (self.transform(img)) 
        img_T = (self.transform(img_T))
        img_T = img_T[:,CRx:CRx+fact,CRy:CRy+fact]
        img = img[:,CRx:CRx+fact,CRy:CRy+fact]

        return {'FF': img, 'FFPE': img_T,'name': name}

    def __len__(self):
        return len(self.list_T)
