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
from data.base_dataset import BaseDataset, get_params, get_transform
from PIL import Image
import numpy as np
import os
import torch


def make_dataset(root, isTrain):
    if isTrain:
        img_list = [os.path.splitext(f)[0] for f in os.listdir(os.path.join(root, 'SBUTrain4KRecoveredSmall', 'ShadowImages')) if f.endswith('.jpg')]
        return [
            (os.path.join(root, 'SBUTrain4KRecoveredSmall', 'ShadowImages', img_name + '.jpg'), os.path.join(root, 'SBUTrain4KRecoveredSmall', 'ShadowMasks', img_name + '.png'))
            for img_name in img_list]
    else:
        img_list = [os.path.splitext(f)[0] for f in os.listdir(os.path.join(root, 'SBU-Test', 'ShadowImages')) if f.endswith('.jpg')]
        return [
            (os.path.join(root, 'SBU-Test', 'ShadowImages', img_name + '.jpg'), os.path.join(root, 'SBU-Test', 'ShadowMasks', img_name + '.png'))
            for img_name in img_list]


class VideoSBUDataset(BaseDataset):
    """A video & sbu dataset class to get images together with video frames."""
    @staticmethod
    def modify_commandline_options(parser, is_train):
        """Add new dataset-specific options, and rewrite default values for existing options.

        Parameters:
            parser          -- original option parser
            is_train (bool) -- whether training phase or test phase. You can use this flag to add training-specific or test-specific options.

        Returns:
            the modified parser.
        """
        parser.add_argument('--seq_len', type=int, default=1, help='length of frame sequence in each sample')
        return parser

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
        self.sbuimg_path = make_dataset(os.path.join(self.root, '../../SBU'), opt.isTrain)
        self.seq_len = opt.seq_len
        groups = []
        imgs = []
        for folder in os.listdir(self.root):
            if os.path.isdir(os.path.join(self.root, folder)):
                images = [os.path.join(folder, img_name) for img_name in os.listdir(os.path.join(self.root, folder)) if img_name.endswith('.jpg')]
                images.sort(key=lambda f: int(''.join(filter(str.isdigit, f))))
                imgs.extend(images)
                groups.extend(images[0:(len(images)-self.seq_len)])
        self.imgs = imgs
        self.groups = groups
        # define the default transform function. You can use <base_dataset.get_transform>; You can also define your custom transform function
        self.transform = get_transform(opt)

    def __getitem__(self, index):
        """Return a data point and its metadata information.

        Parameters:
            index -- a random integer for data indexing

        Returns:
            a dictionary of data with their names. It usually contains the data itself and its metadata information.

        Step 1: get a random image path: e.g., path = self.image_paths[index]
        Step 2: load your data from the disk: e.g., image = Image.open(path).convert('RGB').
        Step 3: convert your data to a PyTorch tensor. You can use helpder functions such as self.transform. e.g., data = self.transform(image)
        Step 4: return a data point as a dictionary.
        """
        img_path, gt_path = self.sbuimg_path[index % len(self.sbuimg_path)]
        A = Image.open(img_path).convert('RGB')
        B = Image.open(gt_path).convert('L')
        w, h = A.size
        A = self.transform(A)
        B = self.transform(B)
        image_path = self.groups[index]
        idx = self.imgs.index(image_path)
        cur_imgs = []
        fol_imgs = []
        flows12 = []
        flows21 = []
        for i in range(self.seq_len):
            image = Image.open(os.path.join(self.root, self.imgs[idx+i])).convert('RGB')
            if self.transform:
                image = self.transform(image)
            cur_imgs.append(image)
        for i in range(self.seq_len):
            image = Image.open(os.path.join(self.root, self.imgs[idx+1+i])).convert('RGB')
            if self.transform:
                image = self.transform(image)
            fol_imgs.append(image)
        for i in range(self.seq_len):
            flow = np.load(os.path.join(self.root, '../../flow_512', self.imgs[idx+i].split('/')[0], 'flow_'+self.imgs[idx+i].split('/')[1][:-4]+'.npy'))
            flow = flow.transpose(2,0,1)
            flow = torch.from_numpy(flow)
            flows12.append(flow)
        for i in range(self.seq_len):
            flow = np.load(os.path.join(self.root, '../../flow_512_bw', self.imgs[idx+i].split('/')[0], 'flow_'+self.imgs[idx+i].split('/')[1][:-4]+'.npy'))
            flow = flow.transpose(2,0,1)
            flow = torch.from_numpy(flow)
            flows21.append(flow)
        x = torch.stack(cur_imgs).squeeze()
        y = torch.stack(fol_imgs).squeeze()
        f12 = torch.stack(flows12).squeeze()
        f21 = torch.stack(flows21).squeeze()
        return {'A': A, 'B': B, 'data_f1': x, 'data_f2': y, 'flow12': f12, 'flow21': f21, 'A_paths': img_path}

    def __len__(self):
        """Return the total number of images."""
        return len(self.groups)
