"""Dataset class template

This module provides a template for users to implement custom datasets.
You can specify '--dataset_mode template' to use this dataset.
The class name should be consistent with both the filename and its dataset_mode option.
The filename should be <dataset_mode>_dataset.py
The class name should be <Dataset_mode>Dataset.py
You need to implement the following functions:
    -- <modify_commandline_options>:　Add dataset-specific options and rewrite default values for existing options.
    -- <__init__>: Initialize this dataset class.
    -- <__getitem__>: Return a data point and its metadata information.
    -- <__len__>: Return the number of images.
"""
from data.base_dataset import BaseDataset, get_transform, get_params
from PIL import Image
import os
import torch
from torchvision import transforms
import numpy as np


class ViShaOffDataset(BaseDataset):
    """A template dataset class for you to implement custom datasets."""
    def __init__(self, opt):
        """Initialize this dataset class.

        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions

        A few things can be done here.
        - save the options (have been done in BaseDataset)
        - get image paths and meta information of the dataset.
        - define the image transformation.
        """
        # save the option and dataset root
        BaseDataset.__init__(self, opt)
        # get the image paths of your dataset;
        self.root = opt.dataroot
        groups = []
        imgs = []
        for folder in os.listdir(self.root):
            if os.path.isdir(os.path.join(self.root, folder)):
                images = [os.path.join(folder, img_name) for img_name in os.listdir(os.path.join(self.root, folder)) if img_name.endswith('.jpg')]
                images.sort(key=lambda f: int(''.join(filter(str.isdigit, f))))
                imgs.extend(images)
                groups.extend(images[0:(len(images)-1)])
        self.imgs = imgs
        self.groups = groups

    def __getitem__(self, index):
        """Return a data point and its metadata information.

        Parameters:
            index -- a random integer for data indexing

        Returns:
            a dictionary of data with their names. It usually contains the data itself and its metadata information.
        """
        image_path = self.groups[index]
        idx = self.imgs.index(image_path)

        image1 = Image.open(os.path.join(self.root, self.imgs[idx])).convert('RGB')
        w, h = image1.size
        transform_params = get_params(self.opt, image1.size)
        i_transform = get_transform(self.opt, transform_params)
        fir_img = i_transform(image1)

        image2 = Image.open(os.path.join(self.root, self.imgs[idx+1])).convert('RGB')
        sec_img = i_transform(image2)

        gt1 = Image.open(os.path.join(self.root, '../labels', self.imgs[idx][:-4]+'.png')).convert('L')
        fir_gt = i_transform(gt1)
        gt2 = Image.open(os.path.join(self.root, '../labels', self.imgs[idx+1][:-4]+'.png')).convert('L')
        sec_gt = i_transform(gt2)
        
        flow12 = np.load(os.path.join(self.root, '../flow12', self.imgs[idx+1][:-4]+'.npy'))
        flow12 = flow12.transpose(2,0,1)
        flow12 = torch.from_numpy(flow12)
        
        flow21 = np.load(os.path.join(self.root, '../flow21', self.imgs[idx][:-4]+'.npy'))
        flow21 = flow21.transpose(2,0,1)
        flow21 = torch.from_numpy(flow21)

        return {'A1': fir_img, 'B1': fir_gt, 'A2': sec_img, 'B2': sec_gt, 'flow12': flow12, 'flow21': flow21, 'ori_w':w, 'ori_h':h, 'A_paths': [self.imgs[idx], self.imgs[idx+1]]}

    def __len__(self):
        """Return the total number of images."""
        return len(self.groups)
