from __future__ import print_function, division
import os
from PIL import Image
import torch
import torch.utils.data
import torchvision
from skimage import io
from torch.utils.data import Dataset
import random
import numpy as np


class Images_Dataset(Dataset):
    """Class for getting data as a Dict
    Args:
        images_dir = path of input images
        labels_dir = path of labeled images
        transformI = Input Images transformation (default: None)
        transformM = Input Labels transformation (default: None)
    Output:
        sample : Dict of images and labels"""

    def __init__(self, images_dir, labels_dir, transformI = None, transformM = None):

        self.labels_dir = labels_dir
        self.images_dir = images_dir
        self.transformI = transformI
        self.transformM = transformM

    def __len__(self):
        return len(self.images_dir)

    def __getitem__(self, idx):

        for i in range(len(self.images_dir)):
            image = io.imread(self.images_dir[i])
            label = io.imread(self.labels_dir[i])
            if self.transformI:
                image = self.transformI(image)
            if self.transformM:
                label = self.transformM(label)
            sample = {'images': image, 'labels': label}

        return sample


class Images_Dataset_folder(torch.utils.data.Dataset):
    """Class for getting individual transformations and data
    Args:
        images_dir = path of input images
        labels_dir = path of labeled images
        transformI = Input Images transformation (default: None)
        transformM = Input Labels transformation (default: None)
    Output:
        tx = Transformed images
        lx = Transformed labels"""

    def __init__(self, images_dir, labels_dir,transformI = None, transformM = None):
        self.images = sorted(os.listdir(images_dir))
        self.labels = sorted(os.listdir(labels_dir))
        self.images_dir = images_dir
        self.labels_dir = labels_dir
        self.transformI = transformI
        self.transformM = transformM

        if self.transformI:
            self.tx = self.transformI
        else:
            self.tx = torchvision.transforms.Compose([
              #  torchvision.transforms.Resize((128,128)),
                torchvision.transforms.CenterCrop(96),
                torchvision.transforms.RandomRotation((-10,10)),
               # torchvision.transforms.RandomHorizontalFlip(),
                torchvision.transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
            ])

        if self.transformM:
            self.lx = self.transformM
        else:
            self.lx = torchvision.transforms.Compose([
              #  torchvision.transforms.Resize((128,128)),
                torchvision.transforms.CenterCrop(96),
                torchvision.transforms.RandomRotation((-10,10)),
                torchvision.transforms.Grayscale(),
                torchvision.transforms.ToTensor(),
                #torchvision.transforms.Lambda(lambda x: torch.cat([x, 1 - x], dim=0))
            ])

    def __len__(self):

        return len(self.images)

    def __getitem__(self, i):
        i1 = Image.open(self.images_dir + self.images[i])
        l1 = Image.open(self.labels_dir + self.labels[i])

        seed=np.random.randint(0,2**32) # make a seed with numpy generator 

        # apply this seed to img tranfsorms
        random.seed(seed) 
        torch.manual_seed(seed)
        img = self.tx(i1)
        
        # apply this seed to target/label tranfsorms  
        random.seed(seed) 
        torch.manual_seed(seed)
        label = self.lx(l1)

        

        return img, label

