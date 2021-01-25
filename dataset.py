import os

import torch
import torch.utils.data as data
from PIL import Image


def make_dataset(root, data='SBU', sub=''):
    if data == 'SBU':
        img_list = [os.path.splitext(f)[0] for f in os.listdir(os.path.join(root, 'ShadowImages')) if f.endswith('.jpg')]
        return [
            (os.path.join(root, 'ShadowImages', img_name + '.jpg'), os.path.join(root, 'ShadowMasks', img_name + '.png'))
            for img_name in img_list]
    if data == 'CUHK':
        if (sub == 'KITTI') or (sub == 'MAP'):
            img_list = [os.path.splitext(f)[0] for f in os.listdir(os.path.join(root, 'shadow_' + sub)) if f.endswith('.png')]
            return [
                (os.path.join(root, 'shadow_' + sub, img_name + '.png'), os.path.join(root, 'mask_' + sub, img_name + '.png'))
                for img_name in img_list]
        else:
            img_list = [os.path.splitext(f)[0] for f in os.listdir(os.path.join(root, 'shadow_' + sub)) if f.endswith('.jpg')]
            return [
                (os.path.join(root, 'shadow_' + sub, img_name + '.jpg'), os.path.join(root, 'mask_' + sub, img_name + '.png'))
                for img_name in img_list]


class ImageFolder(data.Dataset):
    def __init__(self, root, data, sub, joint_transform=None, transform=None, target_transform=None):
        self.root = root
        self.imgs = make_dataset(root, data, sub)
        self.joint_transform = joint_transform
        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, index):
        img_path, gt_path = self.imgs[index]
        img = Image.open(img_path).convert('RGB')
        target = Image.open(gt_path)
        if self.joint_transform is not None:
            img, target = self.joint_transform(img, target)
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return len(self.imgs)
    

class VideoframeDataset(data.Dataset):
    """extract sequences frames from video dataset"""
    def __init__(self, root, seq_len=1, transform=None):
        """
        root: path to videos
        seq_len: length of frame sequence in each sample
        transform: optional, callable, apply image transform
        """
        self.root = root
        self.seq_len = seq_len
        self.transform = transform
        groups = []
        imgs = []
        for folder in os.listdir(root):
            if os.path.isdir(os.path.join(root, folder)):
                images = [os.path.join(folder, img_name) for img_name in os.listdir(os.path.join(root, folder)) if img_name.endswith('.jpg')]
                images.sort(key=lambda f: int(''.join(filter(str.isdigit, f))))
                imgs.extend(images)
                groups.extend(images[0:(len(images)-seq_len)])
        self.imgs = imgs
        self.groups = groups

    def __len__(self):
        return len(self.groups)

    def __getitem__(self, index):
        image_path = self.groups[index]
        idx = self.imgs.index(image_path)
        cur_imgs = []
        fol_imgs = []
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
        x = torch.stack(cur_imgs)
        y = torch.stack(fol_imgs)
        # for i in range(self.seq_len):
        #     image = self.imgs[idx+i]
        #     cur_imgs.append(image)
        # for i in range(self.seq_len):
        #     image = self.imgs[idx+1+i]
        #     fol_imgs.append(image)
        # x = cur_imgs
        # y = fol_imgs

        return x,y