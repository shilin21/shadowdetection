import imageio
import argparse
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os

import torch
from easydict import EasyDict
from torchvision import transforms
from transforms import sep_transforms

from utils.flow_utils import flow_to_image, resize_flow
from utils.torch_utils import restore_model
from models.pwclite import PWCLite


def check_mkdir(dir_name):
    if not os.path.exists(dir_name):
        os.mkdir(dir_name)

class TestHelper():
    def __init__(self, cfg):
        self.cfg = EasyDict(cfg)
        self.device = torch.device('cuda:1') if torch.cuda.is_available() else torch.device('cpu')
        self.model = self.init_model()
        self.input_transform = transforms.Compose([
            sep_transforms.Zoom(*self.cfg.test_shape),
            sep_transforms.ArrayToTensor(),
            transforms.Normalize(mean=[0, 0, 0], std=[255, 255, 255]),
        ])

    def init_model(self):
        model = PWCLite(self.cfg.model)
        # print('Number fo parameters: {}'.format(model.num_parameters()))
        model = model.to(self.device)
        model = restore_model(model, self.cfg.pretrained_model)
        model.eval()
        return model

    def run(self, imgs):
        imgs = [self.input_transform(img).unsqueeze(0) for img in imgs]
        img_pair = torch.cat(imgs, 1).to(self.device)
        return self.model(img_pair)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model', default='checkpoints/Sintel/pwclite_ar.tar')
    parser.add_argument('-s', '--test_shape', default=[448, 1024], type=int, nargs=2)
    parser.add_argument('-n', '--n_frames', type=int, default=2)
    parser.add_argument('-i', '--img_list', default='/nfs/bigcornea/add_disk0/shilinhu/SBUTimeLapse/frames')
    parser.add_argument('-p', '--savepath', default='/nfs/bigcornea/add_disk0/shilinhu/SBUTimeLapse/flow21')
    parser.add_argument('-r', '--result_shape', type=int, default=512)
    args = parser.parse_args()

    cfg = {
        'model': {
            'upsample': True,
            'n_frames': args.n_frames,
            'reduce_dense': True
        },
        'pretrained_model': args.model,
        'test_shape': args.test_shape,
    }

    ts = TestHelper(cfg)

    #imgs = [imageio.imread(img, pilmode='RGB').astype(np.float32) for img in args.img_list]
    #h, w = imgs[0].shape[:2]
    
    video_list = []
    for tmp in os.listdir(args.img_list):
        if os.path.isdir(os.path.join(args.img_list, tmp)):
            video_list.append(os.path.join(args.img_list, tmp))
    for video in video_list:
        img_list = []
        for img_name in os.listdir(video):
            if img_name.endswith('.png'):
                img_list.append(os.path.join(video, img_name))
        img_list.sort(key=lambda f: int(''.join(filter(str.isdigit, f.split('/')[-1]))))
        list_len = len(img_list)
        for i in range(list_len-1):
            print('processing %dth image in %s' % (i, video))
            imgs = [imageio.imread(img, pilmode='RGB').astype(np.float32) for img in img_list[i:i+2]]
            #h, w = imgs[0].shape[:2]
            imgs.reverse()
            flow_12 = ts.run(imgs)['flows_fw'][0]
            flow_12 = resize_flow(flow_12, (args.result_shape, args.result_shape))
            np_flow_12 = flow_12[0].detach().cpu().numpy().transpose([1, 2, 0])
            #vis_flow = flow_to_image(np_flow_12)
            check_mkdir(os.path.join(args.savepath, video.split('/')[-1]))
            #Image.fromarray(vis_flow).save(os.path.join(args.savepath, video.split('/')[-1], 
            #                           'flow_' + img_list[i].split('/')[-1]))
            np.save(os.path.join(args.savepath, video.split('/')[-1], img_list[i].split('/')[-1].split('.')[0]), np_flow_12)

    #flow_12 = ts.run(imgs)['flows_fw'][0]

    #flow_12 = resize_flow(flow_12, (h, w))
    #np_flow_12 = flow_12[0].detach().cpu().numpy().transpose([1, 2, 0])

    #vis_flow = flow_to_image(np_flow_12)

    #fig = plt.figure()
    #plt.imshow(vis_flow)
    #plt.show()
   
