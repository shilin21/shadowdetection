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
import torchvision.transforms as transforms
from PIL import Image
import os


def make_dataset(root, isTrain, masktype):
    if isTrain:
        img_list = [os.path.splitext(f)[0] for f in os.listdir(os.path.join(root, 'SBUTrain4KRecoveredSmall', 'ShadowImages')) if f.endswith('.jpg')]
        return [
            (os.path.join(root, 'SBUTrain4KRecoveredSmall', 'ShadowImages', img_name + '.jpg'), os.path.join(root, 'SBUTrain4KRecoveredSmall', masktype, img_name + '.png'))
            for img_name in img_list]
    else:
        img_list = [os.path.splitext(f)[0] for f in os.listdir(os.path.join(root, 'SBU-Test', 'ShadowImages')) if f.endswith('.jpg')]
        return [
            (os.path.join(root, 'SBU-Test', 'ShadowImages', img_name + '.jpg'), os.path.join(root, 'SBU-Test', 'ShadowMasks', img_name + '.png'))
            for img_name in img_list]


class SBUDataset(BaseDataset):
    """A SBU dataset class to use SBU."""
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
        self.image_paths = make_dataset(self.root, opt.isTrain, opt.masktype)
        self.masktype = opt.masktype
        

    def __getitem__(self, index):
        """Return a data point and its metadata information.

        Parameters:
            index -- a random integer for data indexing

        Returns:
            a dictionary of data with their names. It usually contains the data itself and its metadata information.
        """
        img_path, gt_path = self.image_paths[index]
        A = Image.open(img_path).convert('RGB')
        B = Image.open(gt_path).convert('L')
        w, h = A.size
        
        # apply the same transform to both A and B
        transform_params = get_params(self.opt, A.size)
        A_transform = get_transform(self.opt, transform_params)
        B_transform = get_transform(self.opt, transform_params)
        
        A = A_transform(A)
        A = transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))(A)
        B = B_transform(B)
        if self.masktype != 'ShadowMasks':
            B = B * 255 / 16
        
        return {'A': A, 'B': B, 'A_paths': img_path, 'B_paths': gt_path, 'ori_w': w, 'ori_h': h}

    def __len__(self):
        """Return the total number of images."""
        return len(self.image_paths)
