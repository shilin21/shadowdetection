import datetime
import os
from PIL import Image
import random
import numpy as np
import argparse

import torch
from torch import nn
from torch import optim
from torch.autograd import Variable
from torch.backends import cudnn
from torch.utils.data import DataLoader
from torchvision import transforms
from torch.nn import functional as F

import joint_transforms
from config import *
from dataset import ImageFolder
from misc import AvgMeter, check_mkdir, crf_refine
from model import BDRAR

cudnn.benchmark = True

ckpt_path = './ckpt'
exp_name = 'BDRAR'
con_path = '/data/add_disk0/shilinhu/small_shadow/train'

# batch size of 8 with resolution of 416*416 is exactly OK for the GTX 1080Ti GPU
#args = {
#    'iter_num': 4000,
#    'train_batch_size': 8,
#    'last_iter': 0,
#    'lr': 5e-3,
#    'lr_decay': 0.9,
#    'weight_decay': 5e-4,
#    'momentum': 0.9,
#    'snapshot': '',
#    'scale': 416
#}

a = argparse.ArgumentParser()
a.add_argument("--cuda", type=int, default=0, help="cuda device number")
a.add_argument("--data", type=str, default='SBU', help="train dataset")
a.add_argument("--sub", type=str, default='', help="CUHK subset")
a.add_argument("--iter_num", type=int, default=4000, help="iteration number")
a.add_argument("--train_batch_size", type=int, default=8, help="train batch size")
a.add_argument("--last_iter", type=int, default=0, help="start training at this iteration")
a.add_argument("--lr", type=float, default=5e-3, help="learning rate")
a.add_argument("--lr_decay", type=float, default=0.9, help="learning rate decay rate")
a.add_argument("--weight_decay", type=float, default=5e-4, help="weight decay rate")
a.add_argument("--momentum", type=float, default=0.9, help="momentum rate")
a.add_argument("--snapshot", type=str, default='', help="loading snapshot")
a.add_argument("--scale", type=int, default=416, help="image scale")
a.add_argument("--con_type", type=str, default='size', help="consistency loss type")
a.add_argument("--con_batch", type=int, default=4, help="train consistency batch scale")
a.add_argument("--alpha", type=float, default=1.0, help="importance of consistency loss")
args = a.parse_args()

device = torch.device("cuda:%d" % (args.cuda) if torch.cuda.is_available() else "cpu")

joint_transform = joint_transforms.Compose([
    joint_transforms.RandomHorizontallyFlip(),
    joint_transforms.Resize((args.scale, args.scale))
])
val_joint_transform = joint_transforms.Compose([
    joint_transforms.Resize((args.scale, args.scale))
])
img_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])
target_transform = transforms.ToTensor()
to_pil = transforms.ToPILImage()

if args.data == 'SBU':
    train_set = ImageFolder(sbu_training_root, args.data, args.sub, joint_transform, img_transform, target_transform)
if args.data == 'CUHK':
    train_set = ImageFolder(cuhk_training_root, args.data, args.sub, joint_transform, img_transform, target_transform)
train_loader = DataLoader(train_set, batch_size=args.train_batch_size, num_workers=8, shuffle=True)

bce_logit = nn.BCEWithLogitsLoss().to(device)
log_path = os.path.join(ckpt_path, exp_name, str(datetime.datetime.now()) + '.txt')

con_transform = transforms.Compose([
    transforms.Resize((args.scale, args.scale)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])


def main():
    net = BDRAR().to(device).train()

    optimizer = optim.SGD([
        {'params': [param for name, param in net.named_parameters() if name[-4:] == 'bias'],
         'lr': 2 * args.lr},
        {'params': [param for name, param in net.named_parameters() if name[-4:] != 'bias'],
         'lr': args.lr, 'weight_decay': args.weight_decay}
    ], momentum=args.momentum)

    if len(args.snapshot) > 0:
        print('training resumes from \'%s\'' % args.snapshot)
        net.load_state_dict(torch.load(os.path.join(ckpt_path, exp_name, args.snapshot + '.pth')))
        optimizer.load_state_dict(torch.load(os.path.join(ckpt_path, exp_name, args.snapshot + '_optim.pth')))
        optimizer.param_groups[0]['lr'] = 2 * args.lr
        optimizer.param_groups[1]['lr'] = args.lr

    check_mkdir(ckpt_path)
    check_mkdir(os.path.join(ckpt_path, exp_name))
    open(log_path, 'w').write(str(args) + '\n\n')
    train(net, optimizer)


def train(net, optimizer):
    curr_iter = args.last_iter
    while True:
        train_loss_record, loss_fuse_record, loss1_h2l_record = AvgMeter(), AvgMeter(), AvgMeter()
        loss2_h2l_record, loss3_h2l_record, loss4_h2l_record = AvgMeter(), AvgMeter(), AvgMeter()
        loss1_l2h_record, loss2_l2h_record, loss3_l2h_record = AvgMeter(), AvgMeter(), AvgMeter()
        loss4_l2h_record = AvgMeter()
        loss_con_record = AvgMeter()

        for i, data in enumerate(train_loader):
            optimizer.param_groups[0]['lr'] = 2 * args.lr * (1 - float(curr_iter) / args.iter_num
                                                                ) ** args.lr_decay
            optimizer.param_groups[1]['lr'] = args.lr * (1 - float(curr_iter) / args.iter_num
                                                            ) ** args.lr_decay
            
            # get consistency loss data
            videos = [v for v in os.listdir(con_path) if os.path.isdir(os.path.join(con_path, v))]
            v_path = os.path.join(con_path, random.choice(videos))
            con_imgs = [i for i in os.listdir(v_path) if i.endswith('.jpg')]
            con_imgs.sort(key=lambda f: int(''.join(filter(str.isdigit, f))))
            i_list1 = [random.randrange(len(con_imgs)-1) for _ in range(args.con_batch)]
            i_list2 = [i+1 for i in i_list1]
            i_path1 = [os.path.join(v_path, con_imgs[i]) for i in i_list1]
            i_path2 = [os.path.join(v_path, con_imgs[i]) for i in i_list2]
            c_inputs1 = [con_transform(Image.open(i).convert('RGB')) for i in i_path1]
            c_inputs2 = [con_transform(Image.open(i).convert('RGB')) for i in i_path2]
            c_inputs1 = torch.stack(c_inputs1)
            c_inputs2 = torch.stack(c_inputs2)
            assert c_inputs1.size() == c_inputs2.size()
            con_batch_size = c_inputs1.size(0)
            
            inputs, labels = data
            batch_size = inputs.size(0)
            labels = Variable(labels).to(device)
            
            inputs = torch.cat((inputs, c_inputs1), 0)
            inputs = Variable(inputs).to(device)
            

            optimizer.zero_grad()

            fuse_predict, predict1_h2l, predict2_h2l, predict3_h2l, predict4_h2l, \
            predict1_l2h, predict2_l2h, predict3_l2h, predict4_l2h = net(inputs)

            loss_fuse = bce_logit(fuse_predict[:batch_size], labels)
            loss1_h2l = bce_logit(predict1_h2l[:batch_size], labels)
            loss2_h2l = bce_logit(predict2_h2l[:batch_size], labels)
            loss3_h2l = bce_logit(predict3_h2l[:batch_size], labels)
            loss4_h2l = bce_logit(predict4_h2l[:batch_size], labels)
            loss1_l2h = bce_logit(predict1_l2h[:batch_size], labels)
            loss2_l2h = bce_logit(predict2_l2h[:batch_size], labels)
            loss3_l2h = bce_logit(predict3_l2h[:batch_size], labels)
            loss4_l2h = bce_logit(predict4_l2h[:batch_size], labels)
            
            # calculate consistency loss
            if args.con_type == 'size':
                con_sigmoid1 = F.sigmoid(fuse_predict[batch_size:])
                con_size1 = con_sigmoid1.view(con_batch_size,-1).mean(dim=1)
                # get consistency result
                c_inputs2 = Variable(c_inputs2, requires_grad=False).to(device)
                with torch.no_grad():
                    con_predict2, _, _, _, _, _, _, _, _ = net(c_inputs2)
                    con_sigmoid2 = F.sigmoid(con_predict2)
                    con_size2 = con_sigmoid2.view(con_batch_size,-1).mean(dim=1)

                loss_con = F.l1_loss(con_size1, con_size2)
            if args.con_type == 'pixel':
                con_sigmoid1 = F.sigmoid(fuse_predict[batch_size:])
                # get consistency result
                c_inputs2 = Variable(c_inputs2, requires_grad=False).to(device)
                with torch.no_grad():
                    con_predict2, _, _, _, _, _, _, _, _ = net(c_inputs2)
                    con_sigmoid2 = F.sigmoid(con_predict2)
                
                loss_con = F.mse_loss(con_sigmoid1, con_sigmoid2)

            loss = loss_fuse + loss1_h2l + loss2_h2l + loss3_h2l + loss4_h2l + loss1_l2h + \
                   loss2_l2h + loss3_l2h + loss4_l2h + (loss_con * args.alpha)
            loss.backward()

            optimizer.step()

            train_loss_record.update(loss.data, batch_size)
            loss_fuse_record.update(loss_fuse.data, batch_size)
            loss1_h2l_record.update(loss1_h2l.data, batch_size)
            loss2_h2l_record.update(loss2_h2l.data, batch_size)
            loss3_h2l_record.update(loss3_h2l.data, batch_size)
            loss4_h2l_record.update(loss4_h2l.data, batch_size)
            loss1_l2h_record.update(loss1_l2h.data, batch_size)
            loss2_l2h_record.update(loss2_l2h.data, batch_size)
            loss3_l2h_record.update(loss3_l2h.data, batch_size)
            loss4_l2h_record.update(loss4_l2h.data, batch_size)
            loss_con_record.update(loss_con.data, con_batch_size)

            curr_iter += 1

            log = '[iter %d], [train loss %.5f], [loss_fuse %.5f], [loss1_h2l %.5f], [loss2_h2l %.5f], ' \
                  '[loss3_h2l %.5f], [loss4_h2l %.5f], [loss1_l2h %.5f], [loss2_l2h %.5f], ' \
                  '[loss3_l2h %.5f], [loss4_l2h %.5f], [lr %.13f], [loss_con %.5f]' % \
                  (curr_iter, train_loss_record.avg, loss_fuse_record.avg, loss1_h2l_record.avg,
                   loss2_h2l_record.avg, loss3_h2l_record.avg, loss4_h2l_record.avg, 
                   loss1_l2h_record.avg, loss2_l2h_record.avg, loss3_l2h_record.avg, 
                   loss4_l2h_record.avg, optimizer.param_groups[1]['lr'], loss_con_record.avg)
            print(log)
            open(log_path, 'a').write(log + '\n')

            if curr_iter > 1500 and curr_iter % 500 == 0:
                torch.save(net.state_dict(), 
                           os.path.join(ckpt_path, exp_name, 'baseline1_%s_alpha%.1f_batch%d%d_%d.pth' % (args.con_type, args.alpha, args.train_batch_size, args.con_batch, curr_iter)))
            if curr_iter >= args.iter_num:
                return


if __name__ == '__main__':
    main()
