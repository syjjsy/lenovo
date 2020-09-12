import os
from PIL import Image
import h5py
from torch.utils.data import Dataset
import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib as mpl
import torch
import math
import random
from PIL import Image, ImageOps, ImageEnhance
try:
    import accimage
except ImportError:
    accimage = None
import numpy as np
import numbers
import types
import collections
import warnings
import torch
import torchvision.transforms as transforms
import numpy as np
import random


class KittiLoader(Dataset):
    def __init__(self, root_dir, mode, transform=None):
        # print('xxxxxxxxxxxxxxxxxxxx',root_dir)
        # print('xxxxxxxxxx')
        left_dir = os.path.join(root_dir, 'input/')
        self.left_paths = sorted([os.path.join(left_dir, fname) for fname\
                           in os.listdir(left_dir)]) 

        bg_dir = os.path.join(root_dir, 'gt256/')
        self.bg_paths = sorted([os.path.join(bg_dir, fname) for fname\
                           in os.listdir(bg_dir)])  


        self.transform = transform
        self.mode = mode


    def __len__(self):
        return len(self.left_paths)

    def __getitem__(self, idx):
        # left_image = Image.open(self.left_paths[idx])
        
        bg_image = Image.open(self.bg_paths[idx])
        
        # transa=transforms.Compose([transforms.Resize([192,192]),])#24*8########################################
        # transa=transforms.Compose([transforms.Resize([224,224]),])#28*8########################################
        # bg_image=transa(bg_image)##########################################################################
        
        # print('ffffffffff',self.bg_paths[idx])
        # print(bg_image.size[0])

        # print('xxxxxxxxxxxxx',self.left_paths[idx])
        left_image=h5py.File(self.left_paths[idx])
        for group2 in left_image.keys():
            left_image=left_image[group2][:] 
            #left_image=np.transpose(left_image,(1,0))
            left_image = Image.fromarray(left_image)
        # sample = {'left_image': left_image, 'right_image': right_image}




        sample = {'left_image': left_image,'bg_image': bg_image}
        # sample = {'left_image': left_image, 'D2_image': D2_image,'mask_image':mask_image}

        if self.transform:
            sample = self.transform(sample)
            return sample
        else:
            return sample
