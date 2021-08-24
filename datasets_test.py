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
        self.filesI = os.path.join(root, '')
        self.list_I = os.listdir(self.filesI)

    def __getitem__(self, index):

        I_name = os.path.join(self.filesI,self.list_I[index%len(self.list_I)])
        img = Image.open(I_name)

        Transform = [  
                        transforms.ToTensor(),
                        transforms.Normalize((.5,0.5,0.5), (0.5,0.5,0.5))
                        ]
        # Converting the images to tensor and normalizing them
        self.transform = transforms.Compose(Transform)

        name = self.list_I[index%len(self.list_I)];
        img = (self.transform(img)) 

        return {'FF': img,'name': name}

    def __len__(self):
        return len(self.list_I)
