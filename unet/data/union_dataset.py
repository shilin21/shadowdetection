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
from data.base_dataset import BaseDataset, get_transform
from PIL import Image
import numpy as np
import os
import torch

def make_union_dataset(root, isTrain):
    groups = []
    imgs = []
    if isTrain:
        mpath = os.path.join('ViSha', 'train', 'images')
        for folder in os.listdir(os.path.join(root, mpath)):
            if os.path.isdir(os.path.join(root, mpath, folder)):
                images = [os.path.join(mpath, folder, img_name) for img_name in 
                          os.listdir(os.path.join(root, mpath, folder)) if img_name.endswith('.jpg')]
                images.sort(key=lambda f: int(''.join(filter(str.isdigit, f))))
                imgs.extend(images)
                groups.extend(images[0:(len(images)-1)])
        mpath = os.path.join('small_shadow', 'train')
        for folder in os.listdir(os.path.join(root, mpath)):
            if os.path.isdir(os.path.join(root, mpath, folder)):
                images = [os.path.join(mpath, folder, img_name) for img_name in 
                          os.listdir(os.path.join(root, mpath, folder)) if img_name.endswith('.jpg')]
                images.sort(key=lambda f: int(''.join(filter(str.isdigit, f))))
                imgs.extend(images)
                groups.extend(images[0:(len(images)-1)])
    else:
        mpath = os.path.join('ViSha', 'test', 'images')
        for folder in os.listdir(os.path.join(root, mpath)):
            if os.path.isdir(os.path.join(root, mpath, folder)):
                images = [os.path.join(mpath, folder, img_name) for img_name in 
                          os.listdir(os.path.join(root, mpath, folder)) if img_name.endswith('.jpg')]
                images.sort(key=lambda f: int(''.join(filter(str.isdigit, f))))
                imgs.extend(images)
                groups.extend(images[0:(len(images)-1)])
        mpath = os.path.join('small_shadow', 'train')
        for folder in os.listdir(os.path.join(root, mpath)):
            if os.path.isdir(os.path.join(root, mpath, folder)):
                images = [os.path.join(mpath, folder, img_name) for img_name in 
                          os.listdir(os.path.join(root, mpath, folder)) if img_name.endswith('.jpg')]
                images.sort(key=lambda f: int(''.join(filter(str.isdigit, f))))
                imgs.extend(images)
                groups.extend(images[0:(len(images)-1)])
    return imgs, groups


class UnionDataset(BaseDataset):
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
        self.opt = opt
        self.imgs, self.groups = make_union_dataset(self.root, opt.isTrain)
        # define the default transform function. You can use <base_dataset.get_transform>; You can also define your custom transform function
        self.transform = get_transform(opt)

    def __getitem__(self, index):
        """Return a data point and its metadata information.

        Parameters:
            index -- a random integer for data indexing

        Returns:
            a dictionary of data with their names. It usually contains the data itself and its metadata information.
        """
        image_path = self.groups[index]
        idx = self.imgs.index(image_path)
        if image_path.split('/')[0] == 'ViSha':
            subname = image_path.split('/')[1]
            fir_img = Image.open(os.path.join(self.root, self.imgs[idx])).convert('RGB')
            w, h = fir_img.size
            fir_img = self.transform(fir_img)
            sec_img = Image.open(os.path.join(self.root, self.imgs[idx+1])).convert('RGB')
            sec_img = self.transform(sec_img)
            fir_gt = Image.open(os.path.join(self.root, 'ViSha', subname, 'labels', 
                                             self.imgs[idx].split('/')[-2],
                                             self.imgs[idx].split('/')[-1][:-4]+'.png')).convert('L')
            fir_gt = self.transform(fir_gt)
            sec_gt = Image.open(os.path.join(self.root, 'ViSha', subname, 'labels', 
                                             self.imgs[idx+1].split('/')[-2],
                                             self.imgs[idx+1].split('/')[-1][:-4]+'.png')).convert('L')
            sec_gt = self.transform(sec_gt)
            flow12 = np.load(os.path.join(self.root, 'ViSha', subname, 'flow12', 
                                          self.imgs[idx+1].split('/')[-2],
                                          self.imgs[idx+1].split('/')[-1][:-4]+'.npy'))
            flow12 = flow12.transpose(2,0,1)
            flow12 = torch.from_numpy(flow12)
            flow21 = np.load(os.path.join(self.root, 'ViSha', subname, 'flow21', 
                                          self.imgs[idx].split('/')[-2],
                                          self.imgs[idx].split('/')[-1][:-4]+'.npy'))
            flow21 = flow21.transpose(2,0,1)
            flow21 = torch.from_numpy(flow21)
            
            return {'A1': fir_img, 'B1': fir_gt, 'A2': sec_img, 'B2': sec_gt, 
                    'flow12': flow12, 'flow21': flow21, 'ori_w': w, 'ori_h': h, 
                    'A_paths': [self.imgs[idx], self.imgs[idx+1]]}
        
        elif image_path.split('/')[0] == 'small_shadow':
            fir_img = Image.open(os.path.join(self.root, self.imgs[idx])).convert('RGB')
            w, h = fir_img.size
            fir_img = self.transform(fir_img)
            sec_img = Image.open(os.path.join(self.root, self.imgs[idx+1])).convert('RGB')
            sec_img = self.transform(sec_img)
            fir_gt = torch.zeros(1,self.opt.load_size,self.opt.load_size)
            sec_gt = torch.zeros(1,self.opt.load_size,self.opt.load_size)
            flow12 = np.load(os.path.join(self.root, 'flow_512', 
                                          self.imgs[idx].split('/')[-2],
                                          'flow_'+self.imgs[idx].split('/')[-1][:-4]+'.npy'))
            flow12 = flow12.transpose(2,0,1)
            flow12 = torch.from_numpy(flow12)
            flow21 = np.load(os.path.join(self.root, 'flow_512_bw', 
                                          self.imgs[idx].split('/')[-2],
                                          'flow_'+self.imgs[idx].split('/')[-1][:-4]+'.npy'))
            flow21 = flow21.transpose(2,0,1)
            flow21 = torch.from_numpy(flow21)
            
            return {'A1': fir_img, 'B1': fir_gt, 'A2': sec_img, 'B2': sec_gt, 
                    'flow12': flow12, 'flow21': flow21, 'ori_w': w, 'ori_h': h, 
                    'A_paths': [self.imgs[idx], self.imgs[idx+1]]}

    def __len__(self):
        """Return the total number of images."""
        return len(self.groups)
