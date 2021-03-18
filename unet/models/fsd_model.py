"""Model class template

This module provides a template for users to implement custom models.
You can specify '--model template' to use this model.
The class name should be consistent with both the filename and its model option.
The filename should be <model>_dataset.py
The class name should be <Model>Dataset.py
It implements a simple image-to-image translation baseline based on regression loss.
Given input-output pairs (data_A, data_B), it learns a network netG that can minimize the following L1 loss:
    min_<netG> ||netG(data_A) - data_B||_1
You need to implement the following functions:
    <modify_commandline_options>:　Add model-specific options and rewrite default values for existing options.
    <__init__>: Initialize this model class.
    <set_input>: Unpack input data and perform data pre-processing.
    <forward>: Run forward pass. This will be called by both <optimize_parameters> and <test>.
    <optimize_parameters>: Update network weights; it will be called in every training iteration.
"""
import torch
from .base_model import BaseModel
from . import networks


class FSDModel(BaseModel):
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        """Add new model-specific options and rewrite default values for existing options.

        Parameters:
            parser -- the option parser
            is_train -- if it is training phase or test phase. You can use this flag to add training-specific or test-specific options.

        Returns:
            the modified parser.
        """
        parser.set_defaults(dataset_mode='aligned', output_nc=1, lr=5e-3, batch_size=6, n_epochs=1, n_epochs_decay=39, preprocess='resize', load_size=512, no_epoch=True, save_by_iter=True, load_iter=50000)
        parser.add_argument('--weight_decay', type=float, default=5e-4, help='weight decay for bdrar sgd optimizer')
        parser.add_argument('--momentum', type=float, default=0.9, help='momentum for bdrar sgd optimizer')
        parser.add_argument('--num_classes', type=int, default=1, help='number of classes')
        parser.add_argument('--backbone', type=str, default='mobilenet', help='backbone net type')
        parser.add_argument('--output_stride', type=int, default=16, help='number of output stride')
        parser.add_argument('--sync_bn', default=None, help='synchronized batchnorm or not')
        parser.add_argument('--freeze_bn', default=False, help='freeze bacthnorm or not')
        parser.add_argument('--iter_num', type=int, default=50000, help='number of iterations')
        parser.add_argument('--lr_decay', type=float, default=0.9, help='learning rate decay rate')

        return parser

    def __init__(self, opt):
        """Initialize this model class.

        Parameters:
            opt -- training/test options

        A few things can be done here.
        - (required) call the initialization function of BaseModel
        - define loss function, visualization images, model names, and optimizers
        """
        BaseModel.__init__(self, opt)  # call the initialization method of BaseModel
        # specify the training losses you want to print out. The program will call base_model.get_current_losses to plot the losses to the console and save them to the disk.
        self.loss_names = ['fsd']
        # specify the images you want to save and display. The program will call base_model.get_current_visuals to save and display these images.
        self.visual_names = ['data_A', 'data_B', 'output']
        # specify the models you want to save to the disk. The program will call base_model.save_networks and base_model.load_networks to save and load networks.
        # you can use opt.isTrain to specify different behaviors for training and test. For example, some networks will not be used during test, and you don't need to load them.
        self.model_names = ['FSD']
        # define networks; you can use opt.isTrain to specify different behaviors for training and test.
        self.netFSD = networks.define_fsd(opt.num_classes, opt.backbone, opt.output_stride, opt.sync_bn, opt.freeze_bn, gpu_ids=self.gpu_ids)
        if self.isTrain:  # only defined during training time
            # define your loss functions. You can use losses provided by torch.nn such as torch.nn.L1Loss.
            self.criterionLoss = torch.nn.L1Loss()
            # define and initialize optimizers. You can define one optimizer for each network.
            self.train_params = [{'params': self.netFSD.module.get_1x_lr_params(), 'lr': opt.lr},
                    {'params': self.netFSD.module.get_10x_lr_params(), 'lr': opt.lr * 10}]
            self.optimizer = torch.optim.SGD(self.train_params, momentum=opt.momentum, weight_decay=opt.weight_decay)
            self.optimizers = [self.optimizer]

        # Our program will automatically call <model.setup> to define schedulers, load networks, and print networks

    def set_input(self, input):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.

        Parameters:
            input: a dictionary that contains the data itself and its metadata information.
        """
        AtoB = self.opt.direction == 'AtoB'  # use <direction> to swap data_A and data_B
        self.data_A = input['A' if AtoB else 'B'].to(self.device)  # get image data A
        self.data_B = input['B' if AtoB else 'A'].to(self.device)  # get image data B
        self.image_paths = input['A_paths' if AtoB else 'B_paths']  # get image paths

    def forward(self):
        """Run forward pass. This will be called by both functions <optimize_parameters> and <test>."""
        self.output = self.netFSD(self.data_A)  # generate output image given the input data_A

    def backward(self):
        """Calculate losses, gradients, and update network weights; called in every training iteration"""
        # calculate loss given the input and intermediate results
        self.loss_fsd = self.criterionLoss(self.output, self.data_B)
        self.loss_fsd.backward()

    def optimize_parameters(self):
        """Update network weights; it will be called in every training iteration."""
        self.forward()               # first call forward to calculate intermediate results
        self.optimizer.zero_grad()   # clear network G's existing gradients
        self.backward()              # calculate gradients for network G
        self.optimizer.step()        # update gradients for network G
