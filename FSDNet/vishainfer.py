import os
import time
import numpy as np

import torch
from PIL import Image
from torch.autograd import Variable
from torchvision import transforms

from nets import FSDNet
from misc import check_mkdir, crf_refine
from models.deeplab import *


os.environ['CUDA_VISIBLE_DEVICES'] = '0'

torch.manual_seed(2019)
torch.cuda.set_device(0)

ckpt_path = './ckpt'
exp_name = 'FSDNetvisha'
args = {
    'snapshot': '50000',
    'backbone': 'mobilenet',  # 'resnet', 'xception', 'drn', 'mobilenet'],
    'out_stride': 16,  # 8 or 16
    'sync_bn': None,  # whether to use sync bn (default: auto)
    'freeze_bn': False,
    'crf': 0
}

transform = transforms.Compose([
    transforms.Resize([512,512]),
    transforms.ToTensor() ])


to_pil = transforms.ToPILImage()

#to_test = {'CUHKshadow': '/nfs/bigcornea/add_disk0/shilinhu/CUHKshadow'}
#to_test = {'small_shadow': '/nfs/bigcornea/add_disk0/shilinhu/small_shadow/train/sv277'}
to_test = {'ViSha': '/nfs/bigcornea/add_disk0/shilinhu/ViSha'}


def main():

    net = FSDNet(num_classes=1,
                     backbone=args['backbone'],
                     output_stride=args['out_stride'],
                     sync_bn=args['sync_bn'],
                     freeze_bn=args['freeze_bn']).cuda()

    if len(args['snapshot']) > 0:
        print('load snapshot \'%s\' for testing' % args['snapshot'])
        net.load_state_dict(torch.load(os.path.join(ckpt_path, exp_name, args['snapshot'] + '.pth'),
                                       map_location=lambda storage, loc: storage.cuda(0)))

    net.eval()
    total_time = 0
    with torch.no_grad():
        for name, root in to_test.items():
            
            video_root = os.path.join(root, 'test')
            video_list = [video for video in os.listdir(os.path.join(video_root, 'images')) if os.path.isdir(os.path.join(video_root, 'images', video))]
            img_name = []
            for v in video_list:
                for img in os.listdir(os.path.join(video_root, 'images', v)):
                    if img.endswith('.jpg'):
                        img_name.append(os.path.join(video_root, 'images', v, img))

            for idx, image_name in enumerate(img_name):
                
                check_mkdir(
                    os.path.join(ckpt_path, exp_name, 'prediction_' + args['snapshot']))

                img = Image.open(os.path.join(root, image_name))
                #img = Image.open(os.path.join(root, image_name + '.jpg'))
                w, h = img.size
                img_var = Variable(transform(img).unsqueeze(0)).cuda()

                start_time = time.time()

                res = net(img_var)

                torch.cuda.synchronize()

                total_time = total_time + time.time() - start_time

                print('predicting: %d / %d, avg_time: %.5f' % (idx+1,len(img_name),total_time/(idx+1)))

                result = transforms.Resize((h, w))(to_pil(res.data.squeeze(0).cpu()))
                
                if args['crf'] == 1:
                    result = np.array(result)
                    result = crf_refine(np.array(img.convert('RGB')), result)
                    sub_name = image_name.split('/')
                    check_mkdir(
                        os.path.join(ckpt_path, exp_name, 'prediction_' + args['snapshot'], 
                                     sub_name[-3] + '_w'))
                    check_mkdir(
                        os.path.join(ckpt_path, exp_name, 'prediction_' + args['snapshot'], 
                                     sub_name[-3] + '_w', sub_name[-2]))
                    Image.fromarray(result).save(
                        os.path.join(ckpt_path, exp_name, 'prediction_' + args['snapshot'], 
                                     sub_name[-3] + '_w', sub_name[-2], sub_name[-1]))
                else:
                    sub_name = image_name.split('/')
                #sub_name = root.split('/')

                    check_mkdir(
                        os.path.join(ckpt_path, exp_name, 'prediction_' + args['snapshot'], sub_name[-3]))
                #check_mkdir(
                #    os.path.join(ckpt_path, exp_name, 'prediction_' + args['snapshot'], sub_name[-1]))

                    check_mkdir(
                        os.path.join(ckpt_path, exp_name, 'prediction_' + args['snapshot'], 
                                     sub_name[-3], sub_name[-2]))

                    result.save(
                        os.path.join(ckpt_path, exp_name, 'prediction_' + args['snapshot'], 
                                     sub_name[-3], sub_name[-2], sub_name[-1]))
                
                #result.save(
                #    os.path.join(ckpt_path, exp_name, 'prediction_' + args['snapshot'], 
                #                 sub_name[-1], image_name+'.png'), "PNG")



if __name__ == '__main__':
    main()
