import numpy as np
import os
import argparse

import torch
from PIL import Image
from torch.autograd import Variable
from torchvision import transforms

from config import sbu_testing_root
from misc import check_mkdir, crf_refine
from model import BDRAR

torch.cuda.set_device(0)

a = argparse.ArgumentParser()
a.add_argument("--pathIn", type=str, help="input path")
a.add_argument("--pathOut", type=str, help="output path")
a.add_argument("--intype", type=str, default='.jpg', help="input image type")
a.add_argument("--rootpath", type=str, default='/data/add_disk0/shilinhu', help="root path")
a.add_argument("--expname", type=str, default='test_bdrar', help="example name")
a.add_argument("--snapshot", type=str, default='3000.pth', help="checkpoint path")
a.add_argument("--scale", type=int, default=416, help="image scale")
a.add_argument("--crf", type=int, default=1, help="whether use crf or not, 1 is use")
args = a.parse_args()

#ckpt_path = './ckpt'
#exp_name = 'BDRAR'
#args = {
#    'snapshot': '3000',
#    'scale': 416
#}

img_transform = transforms.Compose([
    transforms.Resize(args.scale),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

to_test = {args.pathIn: os.path.join(args.rootpath, args.expname, args.pathIn)}
to_pil = transforms.ToPILImage()


def main():
    net = BDRAR().cuda()

    if len(args.snapshot) > 0:
        print('load snapshot \'%s\' for testing' % args.snapshot)
        net.load_state_dict(torch.load(os.path.join(args.rootpath, args.expname, args.snapshot)))

    net.eval()
    with torch.no_grad():
        for name, root in to_test.items():
            img_list = [os.path.splitext(f)[0] for f in os.listdir(root) if f.endswith(args.intype)]
            #img_list = [img_name for img_name in os.listdir(root) if img_name.endswith('.jpg')]
            for idx, img_name in enumerate(img_list):
                print('predicting for %s: %d / %d' % (name, idx + 1, len(img_list)))
                check_mkdir(
                    os.path.join(args.pathOut))
                img = Image.open(os.path.join(root, img_name + args.intype))
                w, h = img.size
                img_var = Variable(img_transform(img).unsqueeze(0)).cuda()
                res = net(img_var)
                prediction = np.array(transforms.Resize((h, w))(to_pil(res.data.squeeze(0).cpu())))
                if args.crf == 1:
                    prediction = crf_refine(np.array(img.convert('RGB')), prediction)

                Image.fromarray(prediction).save(
                    os.path.join(args.pathOut, img_name + '.png'), "PNG")


if __name__ == '__main__':
    main()
