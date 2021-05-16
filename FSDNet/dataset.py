import os
import os.path

import torch.utils.data as data
from PIL import Image


def make_dataset(root, is_train):
    if is_train:

        img_txt = open(os.path.join(root, 'train.txt'))

        img_name = []

        for img_list in img_txt:
            x = img_list.split()
            img_name.append([os.path.join(root, x[0]), (os.path.join(root, x[1]))])

        img_txt.close()

        return img_name


    else:

        img_txt = open(os.path.join(root, 'val.txt'))

        img_name = []

        for img_list in img_txt:
            x = img_list.split()
            img_name.append([os.path.join(root, x[0]), (os.path.join(root, x[1]))])

        img_txt.close()

        return img_name


class ImageFolder(data.Dataset):
    def __init__(self, root, joint_transform=None, transform=None, target_transform=None, is_train=True, batch_size=4):
        self.root = root
        self.imgs = make_dataset(root, is_train)
        self.joint_transform = joint_transform
        self.transform = transform
        self.target_transform = target_transform
        self.batch_size = batch_size

    def __getitem__(self, index):
        img_path, gt_path = self.imgs[index % len(self.imgs)]
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
        return len(self.imgs) + self.batch_size - (len(self.imgs) % self.batch_size)


def make_visha(root, is_train):
    if is_train:
        video_root = os.path.join(root, 'train')
        video_list = [video for video in os.listdir(os.path.join(video_root, 'images')) if os.path.isdir(os.path.join(video_root, 'images', video))]
        img_name = []
        for v in video_list:
            for img in os.listdir(os.path.join(video_root, 'images', v)):
                if img.endswith('.jpg'):
                    img_name.append([os.path.join(video_root, 'images', v, img), os.path.join(video_root, 'labels', v, img[:-4]+'.png')])
        return img_name
        
    else:
        video_root = os.path.join(root, 'test')
        video_list = [video for video in os.listdir(os.path.join(video_root, 'images')) if os.path.isdir(os.path.join(video_root, 'images', video))]
        img_name = []
        for v in video_list:
            for img in os.listdir(os.path.join(video_root, 'images', v)):
                if img.endswith('.jpg'):
                    img_name.append([os.path.join(video_root, 'images', v, img), os.path.join(video_root, 'labels', v, img[:-4]+'.png')])
        return img_name


class ViShaFolder(data.Dataset):
    def __init__(self, root, joint_transform=None, transform=None, target_transform=None, is_train=True, batch_size=4):
        self.root = root
        self.imgs = make_visha(root, is_train)
        self.joint_transform = joint_transform
        self.transform = transform
        self.target_transform = target_transform
        self.batch_size = batch_size
        
    def __getitem__(self, index):
        img_path, gt_path = self.imgs[index % len(self.imgs)]
        img = Image.open(img_path).convert('RGB')
        target = Image.open(gt_path).convert('L')
        if self.joint_transform is not None:
            img, target = self.joint_transform(img, target)
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target
    
    def __len__(self):
        return len(self.imgs) + self.batch_size - (len(self.imgs) % self.batch_size)
        
        
