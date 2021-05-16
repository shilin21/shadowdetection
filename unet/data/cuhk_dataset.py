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
import os


def make_dataset(root, isTrain):
    if isTrain:

        img_txt = open(os.path.join(root, 'train.txt'))

        img_name = []

        for img_list in img_txt:
            x = img_list.split()
            img_name.append([os.path.join(root, x[0]), (os.path.join(root, x[1]))])

        img_txt.close()

        return img_name


    else:

        img_txt = open(os.path.join(root, 'test.txt'))

        img_name = []

        for img_list in img_txt:
            x = img_list.split()
            img_name.append([os.path.join(root, x[0]), (os.path.join(root, x[1]))])

        img_txt.close()

        return img_name


class CUHKDataset(BaseDataset):
    """Custom CUHK dataset for using CUHKshadow dataset."""
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
        self.root = opt.dataroot
        # get the image paths of your dataset;
        self.image_paths = make_dataset(self.root, opt.isTrain)
        self.batch_size = opt.batch_size
        

    def __getitem__(self, index):
        """Return a data point and its metadata information.

        Parameters:
            index -- a random integer for data indexing

        Returns:
            a dictionary of data with their names. It usually contains the data itself and its metadata information.
        """
        img_path, gt_path = self.image_paths[index % len(self.image_paths)]
        A = Image.open(img_path).convert('RGB')
        B = Image.open(gt_path).convert('L')
        w, h = A.size
        
        # apply the same transform to both A and B
        transform_params = get_params(self.opt, A.size)
        A_transform = get_transform(self.opt, transform_params)
        B_transform = get_transform(self.opt, transform_params)
        
        A = A_transform(A)
        B = B_transform(B)
        
        return {'A': A, 'B': B, 'A_paths': img_path, 'B_paths': gt_path, 'ori_w': w, 'ori_h': h}

    def __len__(self):
        """Return the total number of images."""
        #return len(self.image_paths) + self.batch_size - (len(self.image_paths) % self.batch_size)
        return len(self.image_paths)
