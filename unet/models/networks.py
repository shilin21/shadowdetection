import torch
import torch.nn as nn
from torch.nn import init
import functools
from torch.optim import lr_scheduler
from .resnext import ResNeXt101, ResNeXt101_5out
import torch.nn.functional as F
from .sync_batchnorm.batchnorm import SynchronizedBatchNorm2d
from .backbone import build_backbone
from .backbone import resnet, xception, drn, mobilenet
from .irnn import irnn

###############################################################################
# Helper Functions
###############################################################################


class Identity(nn.Module):
    def forward(self, x):
        return x


def get_norm_layer(norm_type='instance'):
    """Return a normalization layer

    Parameters:
        norm_type (str) -- the name of the normalization layer: batch | instance | none

    For BatchNorm, we use learnable affine parameters and track running statistics (mean/stddev).
    For InstanceNorm, we do not use learnable affine parameters. We do not track running statistics.
    """
    if norm_type == 'batch':
        norm_layer = functools.partial(nn.BatchNorm2d, affine=True, track_running_stats=True)
    elif norm_type == 'instance':
        norm_layer = functools.partial(nn.InstanceNorm2d, affine=False, track_running_stats=False)
    elif norm_type == 'none':
        def norm_layer(x): return Identity()
    else:
        raise NotImplementedError('normalization layer [%s] is not found' % norm_type)
    return norm_layer


def get_scheduler(optimizer, opt):
    """Return a learning rate scheduler

    Parameters:
        optimizer          -- the optimizer of the network
        opt (option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions．　
                              opt.lr_policy is the name of learning rate policy: linear | step | plateau | cosine

    For 'linear', we keep the same learning rate for the first <opt.n_epochs> epochs
    and linearly decay the rate to zero over the next <opt.n_epochs_decay> epochs.
    For other schedulers (step, plateau, and cosine), we use the default PyTorch schedulers.
    See https://pytorch.org/docs/stable/optim.html for more details.
    """
    if opt.lr_policy == 'linear':
        def lambda_rule(epoch):
            lr_l = 1.0 - max(0, epoch + opt.epoch_count - opt.n_epochs) / float(opt.n_epochs_decay + 1)
            return lr_l
        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
    elif opt.lr_policy == 'step':
        scheduler = lr_scheduler.StepLR(optimizer, step_size=opt.lr_decay_iters, gamma=0.1)
    elif opt.lr_policy == 'plateau':
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, threshold=0.01, patience=5)
    elif opt.lr_policy == 'cosine':
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=opt.n_epochs, eta_min=0)
    else:
        return NotImplementedError('learning rate policy [%s] is not implemented', opt.lr_policy)
    return scheduler


def init_weights(net, init_type='normal', init_gain=0.02):
    """Initialize network weights.

    Parameters:
        net (network)   -- network to be initialized
        init_type (str) -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        init_gain (float)    -- scaling factor for normal, xavier and orthogonal.

    We use 'normal' in the original pix2pix and CycleGAN paper. But xavier and kaiming might
    work better for some applications. Feel free to try yourself.
    """
    def init_func(m):  # define the initialization function
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=init_gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:  # BatchNorm Layer's weight is not a matrix; only normal distribution applies.
            init.normal_(m.weight.data, 1.0, init_gain)
            init.constant_(m.bias.data, 0.0)

    print('initialize network with %s' % init_type)
    net.apply(init_func)  # apply the initialization function <init_func>


def init_net(net, init_type='normal', init_gain=0.02, gpu_ids=[]):
    """Initialize a network: 1. register CPU/GPU device (with multi-GPU support); 2. initialize the network weights
    Parameters:
        net (network)      -- the network to be initialized
        init_type (str)    -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        gain (float)       -- scaling factor for normal, xavier and orthogonal.
        gpu_ids (int list) -- which GPUs the network runs on: e.g., 0,1,2

    Return an initialized network.
    """
    if len(gpu_ids) > 0:
        assert(torch.cuda.is_available())
        net.to(gpu_ids[0])
        net = torch.nn.DataParallel(net, gpu_ids)  # multi-GPUs
    init_weights(net, init_type, init_gain=init_gain)
    return net


def define_G(input_nc, output_nc, ngf, netG, norm='batch', use_dropout=False, init_type='normal', init_gain=0.02, gpu_ids=[]):
    """Create a generator

    Parameters:
        input_nc (int) -- the number of channels in input images
        output_nc (int) -- the number of channels in output images
        ngf (int) -- the number of filters in the last conv layer
        netG (str) -- the architecture's name: resnet_9blocks | resnet_6blocks | unet_256 | unet_128
        norm (str) -- the name of normalization layers used in the network: batch | instance | none
        use_dropout (bool) -- if use dropout layers.
        init_type (str)    -- the name of our initialization method.
        init_gain (float)  -- scaling factor for normal, xavier and orthogonal.
        gpu_ids (int list) -- which GPUs the network runs on: e.g., 0,1,2

    Returns a generator

    Our current implementation provides two types of generators:
        U-Net: [unet_128] (for 128x128 input images) and [unet_256] (for 256x256 input images)
        The original U-Net paper: https://arxiv.org/abs/1505.04597

        Resnet-based generator: [resnet_6blocks] (with 6 Resnet blocks) and [resnet_9blocks] (with 9 Resnet blocks)
        Resnet-based generator consists of several Resnet blocks between a few downsampling/upsampling operations.
        We adapt Torch code from Justin Johnson's neural style transfer project (https://github.com/jcjohnson/fast-neural-style).


    The generator has been initialized by <init_net>. It uses RELU for non-linearity.
    """
    net = None
    norm_layer = get_norm_layer(norm_type=norm)

    if netG == 'resnet_9blocks':
        net = ResnetGenerator(input_nc, output_nc, ngf, norm_layer=norm_layer, use_dropout=use_dropout, n_blocks=9)
    elif netG == 'resnet_6blocks':
        net = ResnetGenerator(input_nc, output_nc, ngf, norm_layer=norm_layer, use_dropout=use_dropout, n_blocks=6)
    elif netG == 'unet_128':
        net = UnetGenerator(input_nc, output_nc, 7, ngf, norm_layer=norm_layer, use_dropout=use_dropout)
    elif netG == 'unet_256':
        net = UnetGenerator(input_nc, output_nc, 8, ngf, norm_layer=norm_layer, use_dropout=use_dropout)
    else:
        raise NotImplementedError('Generator model name [%s] is not recognized' % netG)
    return init_net(net, init_type, init_gain, gpu_ids)


def define_bdrar(init_type='normal', init_gain=0.02, gpu_ids=[]):
    net = BDRAR()
    return init_net(net, init_type, init_gain, gpu_ids)


def define_mtmt(base_model_cfg='resnext101', ema=False, init_type='normal', init_gain=0.02, gpu_ids=[]):
    if not ema:
        net = TUN_bone(base_model_cfg, *extra_layer(base_model_cfg, ResNeXt101_5out()))
        return init_net(net, init_type, init_gain, gpu_ids)
    else:
        net = TUN_bone(base_model_cfg, *extra_layer(base_model_cfg, ResNeXt101_5out()))
        return init_net(net, init_type, init_gain, gpu_ids)
    

# extra part
def extra_layer(base_model_cfg, vgg):
    if base_model_cfg == 'vgg':
        config = config_vgg
    elif base_model_cfg == 'resnet':
        config = config_resnet
    elif base_model_cfg == 'resnext101':
        config = config_resnext101
    merge1_layers = MergeLayer1(config['merge1'])
    merge2_layers = MergeLayer2(config['merge2'])

    return vgg, merge1_layers, merge2_layers


def define_fsd(num_classes, backbone, output_stride, sync_bn, freeze_bn, init_type='normal', init_gain=0.02, gpu_ids=[]):
    net = FSDNet(num_classes=num_classes, backbone=backbone, output_stride=output_stride, sync_bn=sync_bn, freeze_bn=freeze_bn)
    return init_net(net, init_type, init_gain, gpu_ids)


def build_backbone(backbone, output_stride, BatchNorm):
    if backbone == 'resnet':
        return resnet.ResNet101(output_stride, BatchNorm)
    elif backbone == 'xception':
        return xception.AlignedXception(output_stride, BatchNorm)
    elif backbone == 'drn':
        return drn.drn_d_54(BatchNorm)
    elif backbone == 'mobilenet':
        return mobilenet.MobileNetV2(output_stride, BatchNorm)
    else:
        raise NotImplementedError
    

##############################################################################
# Classes
##############################################################################
class ResnetGenerator(nn.Module):
    """Resnet-based generator that consists of Resnet blocks between a few downsampling/upsampling operations.

    We adapt Torch code and idea from Justin Johnson's neural style transfer project(https://github.com/jcjohnson/fast-neural-style)
    """

    def __init__(self, input_nc, output_nc, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False, n_blocks=6, padding_type='reflect'):
        """Construct a Resnet-based generator

        Parameters:
            input_nc (int)      -- the number of channels in input images
            output_nc (int)     -- the number of channels in output images
            ngf (int)           -- the number of filters in the last conv layer
            norm_layer          -- normalization layer
            use_dropout (bool)  -- if use dropout layers
            n_blocks (int)      -- the number of ResNet blocks
            padding_type (str)  -- the name of padding layer in conv layers: reflect | replicate | zero
        """
        assert(n_blocks >= 0)
        super(ResnetGenerator, self).__init__()
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        model = [nn.ReflectionPad2d(3),
                 nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0, bias=use_bias),
                 norm_layer(ngf),
                 nn.ReLU(True)]

        n_downsampling = 2
        for i in range(n_downsampling):  # add downsampling layers
            mult = 2 ** i
            model += [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=2, padding=1, bias=use_bias),
                      norm_layer(ngf * mult * 2),
                      nn.ReLU(True)]

        mult = 2 ** n_downsampling
        for i in range(n_blocks):       # add ResNet blocks

            model += [ResnetBlock(ngf * mult, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout, use_bias=use_bias)]

        for i in range(n_downsampling):  # add upsampling layers
            mult = 2 ** (n_downsampling - i)
            model += [nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2),
                                         kernel_size=3, stride=2,
                                         padding=1, output_padding=1,
                                         bias=use_bias),
                      norm_layer(int(ngf * mult / 2)),
                      nn.ReLU(True)]
        model += [nn.ReflectionPad2d(3)]
        model += [nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0)]
        model += [nn.Tanh()]

        self.model = nn.Sequential(*model)

    def forward(self, input):
        """Standard forward"""
        return self.model(input)


class ResnetBlock(nn.Module):
    """Define a Resnet block"""

    def __init__(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        """Initialize the Resnet block

        A resnet block is a conv block with skip connections
        We construct a conv block with build_conv_block function,
        and implement skip connections in <forward> function.
        Original Resnet paper: https://arxiv.org/pdf/1512.03385.pdf
        """
        super(ResnetBlock, self).__init__()
        self.conv_block = self.build_conv_block(dim, padding_type, norm_layer, use_dropout, use_bias)

    def build_conv_block(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        """Construct a convolutional block.

        Parameters:
            dim (int)           -- the number of channels in the conv layer.
            padding_type (str)  -- the name of padding layer: reflect | replicate | zero
            norm_layer          -- normalization layer
            use_dropout (bool)  -- if use dropout layers.
            use_bias (bool)     -- if the conv layer uses bias or not

        Returns a conv block (with a conv layer, a normalization layer, and a non-linearity layer (ReLU))
        """
        conv_block = []
        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)

        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias), norm_layer(dim), nn.ReLU(True)]
        if use_dropout:
            conv_block += [nn.Dropout(0.5)]

        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)
        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias), norm_layer(dim)]

        return nn.Sequential(*conv_block)

    def forward(self, x):
        """Forward function (with skip connections)"""
        out = x + self.conv_block(x)  # add skip connections
        return out


class UnetGenerator(nn.Module):
    """Create a Unet-based generator"""

    def __init__(self, input_nc, output_nc, num_downs, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False):
        """Construct a Unet generator
        Parameters:
            input_nc (int)  -- the number of channels in input images
            output_nc (int) -- the number of channels in output images
            num_downs (int) -- the number of downsamplings in UNet. For example, # if |num_downs| == 7,
                                image of size 128x128 will become of size 1x1 # at the bottleneck
            ngf (int)       -- the number of filters in the last conv layer
            norm_layer      -- normalization layer

        We construct the U-Net from the innermost layer to the outermost layer.
        It is a recursive process.
        """
        super(UnetGenerator, self).__init__()
        # construct unet structure
        unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, input_nc=None, submodule=None, norm_layer=norm_layer, innermost=True)  # add the innermost layer
        for i in range(num_downs - 5):          # add intermediate layers with ngf * 8 filters
            unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, input_nc=None, submodule=unet_block, norm_layer=norm_layer, use_dropout=use_dropout)
        # gradually reduce the number of filters from ngf * 8 to ngf
        unet_block = UnetSkipConnectionBlock(ngf * 4, ngf * 8, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock(ngf * 2, ngf * 4, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock(ngf, ngf * 2, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        self.model = UnetSkipConnectionBlock(output_nc, ngf, input_nc=input_nc, submodule=unet_block, outermost=True, norm_layer=norm_layer)  # add the outermost layer

    def forward(self, input):
        """Standard forward"""
        return self.model(input)


class UnetSkipConnectionBlock(nn.Module):
    """Defines the Unet submodule with skip connection.
        X -------------------identity----------------------
        |-- downsampling -- |submodule| -- upsampling --|
    """

    def __init__(self, outer_nc, inner_nc, input_nc=None,
                 submodule=None, outermost=False, innermost=False, norm_layer=nn.BatchNorm2d, use_dropout=False):
        """Construct a Unet submodule with skip connections.

        Parameters:
            outer_nc (int) -- the number of filters in the outer conv layer
            inner_nc (int) -- the number of filters in the inner conv layer
            input_nc (int) -- the number of channels in input images/features
            submodule (UnetSkipConnectionBlock) -- previously defined submodules
            outermost (bool)    -- if this module is the outermost module
            innermost (bool)    -- if this module is the innermost module
            norm_layer          -- normalization layer
            use_dropout (bool)  -- if use dropout layers.
        """
        super(UnetSkipConnectionBlock, self).__init__()
        self.outermost = outermost
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d
        if input_nc is None:
            input_nc = outer_nc
        downconv = nn.Conv2d(input_nc, inner_nc, kernel_size=4,
                             stride=2, padding=1, bias=use_bias)
        downrelu = nn.LeakyReLU(0.2, True)
        downnorm = norm_layer(inner_nc)
        uprelu = nn.ReLU(True)
        upnorm = norm_layer(outer_nc)

        if outermost:
            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1)
            down = [downconv]
            up = [uprelu, upconv, nn.Tanh()]
            model = down + [submodule] + up
        elif innermost:
            upconv = nn.ConvTranspose2d(inner_nc, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1, bias=use_bias)
            down = [downrelu, downconv]
            up = [uprelu, upconv, upnorm]
            model = down + up
        else:
            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1, bias=use_bias)
            down = [downrelu, downconv, downnorm]
            up = [uprelu, upconv, upnorm]

            if use_dropout:
                model = down + [submodule] + up + [nn.Dropout(0.5)]
            else:
                model = down + [submodule] + up

        self.model = nn.Sequential(*model)

    def forward(self, x):
        if self.outermost:
            return self.model(x)
        else:   # add skip connections
            return torch.cat([x, self.model(x)], 1)
    

class BDRAR(nn.Module):
    def __init__(self):
        super(BDRAR, self).__init__()
        resnext = ResNeXt101()
        self.layer0 = resnext.layer0
        self.layer1 = resnext.layer1
        self.layer2 = resnext.layer2
        self.layer3 = resnext.layer3
        self.layer4 = resnext.layer4

        self.down4 = nn.Sequential(
            nn.Conv2d(2048, 32, 1, bias=False), nn.BatchNorm2d(32), nn.ReLU()
        )
        self.down3 = nn.Sequential(
            nn.Conv2d(1024, 32, 1, bias=False), nn.BatchNorm2d(32), nn.ReLU()
        )
        self.down2 = nn.Sequential(
            nn.Conv2d(512, 32, 1, bias=False), nn.BatchNorm2d(32), nn.ReLU()
        )
        self.down1 = nn.Sequential(
            nn.Conv2d(256, 32, 1, bias=False), nn.BatchNorm2d(32), nn.ReLU()
        )

        self.refine3_hl = nn.Sequential(
            nn.Conv2d(64, 32, 1, bias=False), nn.BatchNorm2d(32), nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=1, groups=32, bias=False), nn.BatchNorm2d(32), nn.ReLU(),
            nn.Conv2d(32, 32, 1, bias=False), nn.BatchNorm2d(32)
        )
        self.refine2_hl = nn.Sequential(
            nn.Conv2d(64, 32, 1, bias=False), nn.BatchNorm2d(32), nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=1, groups=32, bias=False), nn.BatchNorm2d(32), nn.ReLU(),
            nn.Conv2d(32, 32, 1, bias=False), nn.BatchNorm2d(32)
        )
        self.refine1_hl = nn.Sequential(
            nn.Conv2d(64, 32, 1, bias=False), nn.BatchNorm2d(32), nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=1, groups=32, bias=False), nn.BatchNorm2d(32), nn.ReLU(),
            nn.Conv2d(32, 32, 1, bias=False), nn.BatchNorm2d(32)
        )
        self.attention3_hl = _AttentionModule()
        self.attention2_hl = _AttentionModule()
        self.attention1_hl = _AttentionModule()

        self.refine2_lh = nn.Sequential(
            nn.Conv2d(64, 32, 1, bias=False), nn.BatchNorm2d(32), nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=1, groups=32, bias=False), nn.BatchNorm2d(32), nn.ReLU(),
            nn.Conv2d(32, 32, 1, bias=False), nn.BatchNorm2d(32)
        )
        self.refine4_lh = nn.Sequential(
            nn.Conv2d(64, 32, 1, bias=False), nn.BatchNorm2d(32), nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=1, groups=32, bias=False), nn.BatchNorm2d(32), nn.ReLU(),
            nn.Conv2d(32, 32, 1, bias=False), nn.BatchNorm2d(32)
        )
        self.refine3_lh = nn.Sequential(
            nn.Conv2d(64, 32, 1, bias=False), nn.BatchNorm2d(32), nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=1, groups=32, bias=False), nn.BatchNorm2d(32), nn.ReLU(),
            nn.Conv2d(32, 32, 1, bias=False), nn.BatchNorm2d(32)
        )
        self.attention2_lh = _AttentionModule()
        self.attention3_lh = _AttentionModule()
        self.attention4_lh = _AttentionModule()

        self.fuse_attention = nn.Sequential(
            nn.Conv2d(64, 16, 3, padding=1, bias=False), nn.BatchNorm2d(16), nn.ReLU(),
            nn.Conv2d(16, 2, 1)
        )

        self.predict = nn.Sequential(
            nn.Conv2d(32, 8, 3, padding=1, bias=False), nn.BatchNorm2d(8), nn.ReLU(),
            nn.Dropout(0.1), nn.Conv2d(8, 1, 1)
        )

        for m in self.modules():
            if isinstance(m, nn.ReLU):#or isinstance(m, nn.Dropout)
                m.inplace = True

    def forward(self, x):
        layer0 = self.layer0(x)
        layer1 = self.layer1(layer0)
        layer2 = self.layer2(layer1)
        layer3 = self.layer3(layer2)
        layer4 = self.layer4(layer3)

        down4 = self.down4(layer4)
        down3 = self.down3(layer3)
        down2 = self.down2(layer2)
        down1 = self.down1(layer1)

        down4 = F.upsample(down4, size=down3.size()[2:], mode='bilinear')
        refine3_hl_0 = F.relu(self.refine3_hl(torch.cat((down4, down3), 1)) + down4, True)
        refine3_hl_0 = (1 + self.attention3_hl(torch.cat((down4, down3), 1))) * refine3_hl_0
        refine3_hl_1 = F.relu(self.refine3_hl(torch.cat((refine3_hl_0, down3), 1)) + refine3_hl_0, True)
        refine3_hl_1 = (1 + self.attention3_hl(torch.cat((refine3_hl_0, down3), 1))) * refine3_hl_1

        refine3_hl_1 = F.upsample(refine3_hl_1, size=down2.size()[2:], mode='bilinear')
        refine2_hl_0 = F.relu(self.refine2_hl(torch.cat((refine3_hl_1, down2), 1)) + refine3_hl_1, True)
        refine2_hl_0 = (1 + self.attention2_hl(torch.cat((refine3_hl_1, down2), 1))) * refine2_hl_0
        refine2_hl_1 = F.relu(self.refine2_hl(torch.cat((refine2_hl_0, down2), 1)) + refine2_hl_0, True)
        refine2_hl_1 = (1 + self.attention2_hl(torch.cat((refine2_hl_0, down2), 1))) * refine2_hl_1

        refine2_hl_1 = F.upsample(refine2_hl_1, size=down1.size()[2:], mode='bilinear')
        refine1_hl_0 = F.relu(self.refine1_hl(torch.cat((refine2_hl_1, down1), 1)) + refine2_hl_1, True)
        refine1_hl_0 = (1 + self.attention1_hl(torch.cat((refine2_hl_1, down1), 1))) * refine1_hl_0
        refine1_hl_1 = F.relu(self.refine1_hl(torch.cat((refine1_hl_0, down1), 1)) + refine1_hl_0, True)
        refine1_hl_1 = (1 + self.attention1_hl(torch.cat((refine1_hl_0, down1), 1))) * refine1_hl_1

        down2 = F.upsample(down2, size=down1.size()[2:], mode='bilinear')
        refine2_lh_0 = F.relu(self.refine2_lh(torch.cat((down1, down2), 1)) + down1, True)
        refine2_lh_0 = (1 + self.attention2_lh(torch.cat((down1, down2), 1))) * refine2_lh_0
        refine2_lh_1 = F.relu(self.refine2_lh(torch.cat((refine2_lh_0, down2), 1)) + refine2_lh_0, True)
        refine2_lh_1 = (1 + self.attention2_lh(torch.cat((refine2_lh_0, down2), 1))) * refine2_lh_1

        down3 = F.upsample(down3, size=down1.size()[2:], mode='bilinear')
        refine3_lh_0 = F.relu(self.refine3_lh(torch.cat((refine2_lh_1, down3), 1)) + refine2_lh_1, True)
        refine3_lh_0 = (1 + self.attention3_lh(torch.cat((refine2_lh_1, down3), 1))) * refine3_lh_0
        refine3_lh_1 = F.relu(self.refine3_lh(torch.cat((refine3_lh_0, down3), 1)) + refine3_lh_0, True)
        refine3_lh_1 = (1 + self.attention3_lh(torch.cat((refine3_lh_0, down3), 1))) * refine3_lh_1

        down4 = F.upsample(down4, size=down1.size()[2:], mode='bilinear')
        refine4_lh_0 = F.relu(self.refine4_lh(torch.cat((refine3_lh_1, down4), 1)) + refine3_lh_1, True)
        refine4_lh_0 = (1 + self.attention4_lh(torch.cat((refine3_lh_1, down4), 1))) * refine4_lh_0
        refine4_lh_1 = F.relu(self.refine4_lh(torch.cat((refine4_lh_0, down4), 1)) + refine4_lh_0, True)
        refine4_lh_1 = (1 + self.attention4_lh(torch.cat((refine4_lh_0, down4), 1))) * refine4_lh_1

        refine3_hl_1 = F.upsample(refine3_hl_1, size=down1.size()[2:], mode='bilinear')
        predict4_hl = self.predict(down4)
        predict3_hl = self.predict(refine3_hl_1)
        predict2_hl = self.predict(refine2_hl_1)
        predict1_hl = self.predict(refine1_hl_1)

        predict1_lh = self.predict(down1)
        predict2_lh = self.predict(refine2_lh_1)
        predict3_lh = self.predict(refine3_lh_1)
        predict4_lh = self.predict(refine4_lh_1)

        fuse_attention = F.sigmoid(self.fuse_attention(torch.cat((refine1_hl_1, refine4_lh_1), 1)))
        fuse_predict = torch.sum(fuse_attention * torch.cat((predict1_hl, predict4_lh), 1), 1, True)

        predict4_hl = F.upsample(predict4_hl, size=x.size()[2:], mode='bilinear')
        predict3_hl = F.upsample(predict3_hl, size=x.size()[2:], mode='bilinear')
        predict2_hl = F.upsample(predict2_hl, size=x.size()[2:], mode='bilinear')
        predict1_hl = F.upsample(predict1_hl, size=x.size()[2:], mode='bilinear')
        predict1_lh = F.upsample(predict1_lh, size=x.size()[2:], mode='bilinear')
        predict2_lh = F.upsample(predict2_lh, size=x.size()[2:], mode='bilinear')
        predict3_lh = F.upsample(predict3_lh, size=x.size()[2:], mode='bilinear')
        predict4_lh = F.upsample(predict4_lh, size=x.size()[2:], mode='bilinear')
        fuse_predict = F.upsample(fuse_predict, size=x.size()[2:], mode='bilinear')

        return fuse_predict, predict1_hl, predict2_hl, predict3_hl, predict4_hl, predict1_lh, predict2_lh, predict3_lh, predict4_lh
    
    
class _AttentionModule(nn.Module):
    def __init__(self):
        super(_AttentionModule, self).__init__()
        self.block1 = nn.Sequential(
            nn.Conv2d(64, 64, 1, bias=False), nn.BatchNorm2d(64), nn.ReLU(),
            nn.Conv2d(64, 64, 3, dilation=2, padding=2, groups=32, bias=False), nn.BatchNorm2d(64), nn.ReLU(),
            nn.Conv2d(64, 64, 1, bias=False), nn.BatchNorm2d(64)
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(64, 64, 1, bias=False), nn.BatchNorm2d(64), nn.ReLU(),
            nn.Conv2d(64, 64, 3, dilation=3, padding=3, groups=32, bias=False), nn.BatchNorm2d(64), nn.ReLU(),
            nn.Conv2d(64, 64, 1, bias=False), nn.BatchNorm2d(64)
        )
        self.block3 = nn.Sequential(
            nn.Conv2d(64, 64, 1, bias=False), nn.BatchNorm2d(64), nn.ReLU(),
            nn.Conv2d(64, 64, 3, dilation=4, padding=4, groups=32, bias=False), nn.BatchNorm2d(64), nn.ReLU(),
            nn.Conv2d(64, 32, 1, bias=False), nn.BatchNorm2d(32)
        )
        self.down = nn.Sequential(
            nn.Conv2d(64, 32, 1, bias=False), nn.BatchNorm2d(32)
        )

    def forward(self, x):
        block1 = F.relu(self.block1(x) + x, True)
        block2 = F.relu(self.block2(block1) + block1, True)
        block3 = F.sigmoid(self.block3(block2) + self.down(block2))
        return block3
    
    
class ConvertLayer(nn.Module):
    def __init__(self, list_k):
        super(ConvertLayer, self).__init__()
        up0, up1, up2 = [], [], []
        for i in range(len(list_k[0])):
            up0.append(nn.Sequential(nn.Conv2d(list_k[0][i], list_k[1][i], 1, 1, bias=False), nn.BatchNorm2d(list_k[1][i]), nn.ReLU(inplace=True)))

        self.convert0 = nn.ModuleList(up0)

    def forward(self, list_x):
        resl = []
        for i in range(len(list_x)):
            resl.append(self.convert0[i](list_x[i]))
        return resl
    
    
class ResidualBlockLayer(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResidualBlockLayer, self).__init__()
        self.residual_block1 = nn.Sequential(nn.Conv2d(in_channels, in_channels//4, 1, 1,bias=False),
                      nn.Conv2d(in_channels//4, in_channels//4, 3, 1, 1, bias=False),
                      nn.Conv2d(in_channels//4, in_channels, 1, 1, bias=False))
        self.residual_block2 = nn.Sequential(nn.Conv2d(in_channels, in_channels//4, 1, 1,bias=False),
                      nn.Conv2d(in_channels//4, in_channels//4, 3, 1, 1, bias=False),
                      nn.Conv2d(in_channels//4, in_channels, 1, 1, bias=False))
        self.out = nn.Conv2d(in_channels, out_channels, 1, 1, bias=False)
        self.relu = nn.ReLU()

    def forward(self, x):
        x_tmp = self.relu(x + self.residual_block1(x))
        x_tmp = self.relu(x_tmp + self.residual_block2(x_tmp))
        x_tmp = self.out(x_tmp)
        return x_tmp

    
#DSS merge
class MergeLayer1(nn.Module): # list_k: [[32, 0, 32, 3, 1], [64, 0, 64, 3, 1], [64, 0, 64, 5, 2], [64, 0, 64, 5, 2], [64, 0, 64, 7, 3]]

    def __init__(self, list_k):
        super(MergeLayer1, self).__init__()
        self.list_k = list_k
        trans, up, DSS = [], [], []

        for i, ik in enumerate(list_k):
            up.append(nn.Sequential(nn.Conv2d(ik[0], ik[2], ik[3], 1, ik[4]), nn.BatchNorm2d(ik[2]), nn.ReLU(inplace=True),
                                    nn.Conv2d(ik[2], ik[2], ik[3], 1, ik[4]), nn.BatchNorm2d(ik[2]), nn.ReLU(inplace=True),
                                    nn.Conv2d(ik[2], ik[2], ik[3], 1, ik[4]), nn.BatchNorm2d(ik[2]), nn.ReLU(inplace=True)))
            if i > 0 and i < len(list_k)-1: # i represent number
                DSS.append(nn.Sequential(nn.Conv2d(ik[0]*(i+1), ik[0], 1, 1, 1),
                           nn.BatchNorm2d(ik[0]), nn.ReLU(inplace=True)))

        trans.append(nn.Sequential(nn.Conv2d(64, 32, 1, 1, bias=False), nn.ReLU(inplace=True)))
        # self.shadow_score = nn.Conv2d(list_k[0][2], 1, 3, 1, 1)
        self.shadow_score = nn.Sequential(
            nn.Conv2d(list_k[1][2], list_k[1][2]//4, 3, 1, 1), nn.BatchNorm2d(list_k[1][2]//4), nn.ReLU(inplace=True),
            nn.Dropout(0.1), nn.Conv2d(list_k[1][2]//4, 1, 1)
        )
        # self.edge_score = nn.Conv2d(list_k[0][2], 1, 3, 1, 1)
        self.edge_score = nn.Sequential(
            nn.Conv2d(list_k[0][2], list_k[0][2]//4, 3, 1, 1), nn.BatchNorm2d(list_k[0][2]//4), nn.ReLU(inplace=True),
            nn.Dropout(0.1), nn.Conv2d(list_k[0][2]//4, 1, 1)
        )

        self.up = nn.ModuleList(up)
        self.relu = nn.ReLU()
        self.trans = nn.ModuleList(trans)
        self.DSS = nn.ModuleList(DSS)

        # subitizing section
        self.number_per_fc = nn.Linear(list_k[1][2], 1) #64->1
        torch.nn.init.constant_(self.number_per_fc.weight, 0)

    def forward(self, list_x, x_size):
        up_edge, up_sal, edge_feature, sal_feature, U_tmp = [], [], [], [], []

        num_f = len(list_x)
        tmp = self.up[num_f - 1](list_x[num_f - 1])
        sal_feature.append(tmp)
        U_tmp.append(tmp)
        up_sal.append(F.interpolate(self.shadow_score(tmp), x_size, mode='bilinear', align_corners=True))
        # layer5
        up_tmp_x2 = F.interpolate(U_tmp[0], list_x[3].size()[2:], mode='bilinear', align_corners=True)
        U_tmp.append(self.DSS[0](torch.cat([up_tmp_x2, list_x[3]], dim=1)))
        tmp = self.up[3](U_tmp[-1])
        sal_feature.append(tmp)
        up_sal.append(F.interpolate(self.shadow_score(tmp), x_size, mode='bilinear', align_corners=True))
        # layer4
        up_tmp_x2 = F.interpolate(U_tmp[1], list_x[2].size()[2:], mode='bilinear', align_corners=True)
        up_tmp_x4 = F.interpolate(U_tmp[0], list_x[2].size()[2:], mode='bilinear', align_corners=True)
        U_tmp.append(self.DSS[1](torch.cat([up_tmp_x4, up_tmp_x2, list_x[2]], dim=1)))
        tmp = self.up[2](U_tmp[-1])
        sal_feature.append(tmp)
        up_sal.append(F.interpolate(self.shadow_score(tmp), x_size, mode='bilinear', align_corners=True))
        # layer3
        up_tmp_x2 = F.interpolate(U_tmp[2], list_x[1].size()[2:], mode='bilinear', align_corners=True)
        up_tmp_x4 = F.interpolate(U_tmp[1], list_x[1].size()[2:], mode='bilinear', align_corners=True)
        up_tmp_x8 = F.interpolate(U_tmp[0], list_x[1].size()[2:], mode='bilinear', align_corners=True)
        U_tmp.append(self.DSS[2](torch.cat([up_tmp_x8, up_tmp_x4, up_tmp_x2, list_x[1]], dim=1)))
        tmp = self.up[1](U_tmp[-1])
        sal_feature.append(tmp)
        up_sal.append(F.interpolate(self.shadow_score(tmp), x_size, mode='bilinear', align_corners=True))

        vector = F.adaptive_avg_pool2d(sal_feature[0], output_size=1)
        vector = vector.view(vector.size(0), -1)
        up_subitizing = self.number_per_fc(vector)

        # edge layer fuse
        U_tmp = list_x[0] + F.interpolate((self.trans[-1](sal_feature[0])), list_x[0].size()[2:], mode='bilinear', align_corners=True)
        tmp = self.up[0](U_tmp)
        edge_feature.append(tmp)

        up_edge.append(F.interpolate(self.edge_score(tmp), x_size, mode='bilinear', align_corners=True))
        return up_edge, edge_feature, up_sal, sal_feature, up_subitizing
    
    
class MergeLayer2(nn.Module): # list_k [[32], [64, 64, 64, 64]]
    def __init__(self, list_k):
        super(MergeLayer2, self).__init__()
        self.list_k = list_k
        trans, up, score = [], [], []
        for i in list_k[0]:
            tmp = []
            tmp_up = []
            tmp_score = []
            feature_k = [[3, 1], [5, 2], [5, 2], [7, 3]]
            for idx, j in enumerate(list_k[1]):
                tmp.append(nn.Sequential(nn.Conv2d(j, i, 1, 1, bias=False), nn.BatchNorm2d(i), nn.ReLU(inplace=True)))
                tmp_up.append(
                    nn.Sequential(nn.Conv2d(i, i, feature_k[idx][0], 1, feature_k[idx][1]), nn.BatchNorm2d(i), nn.ReLU(inplace=True),
                                  nn.Conv2d(i, i, feature_k[idx][0], 1, feature_k[idx][1]), nn.BatchNorm2d(i), nn.ReLU(inplace=True),
                                  nn.Conv2d(i, i, feature_k[idx][0], 1, feature_k[idx][1]), nn.BatchNorm2d(i), nn.ReLU(inplace=True)))
            trans.append(nn.ModuleList(tmp))
            up.append(nn.ModuleList(tmp_up))
        # self.sub_score = nn.Conv2d(list_k[0][0], 1, 3, 1, 1)
        self.sub_score = nn.Sequential(
            nn.Conv2d(list_k[0][0], list_k[0][0]//4, 3, 1, 1), nn.BatchNorm2d(list_k[0][0]//4), nn.ReLU(inplace=True),
            nn.Dropout(0.1), nn.Conv2d(list_k[0][0]//4, 1, 1)
        )

        self.trans, self.up = nn.ModuleList(trans), nn.ModuleList(up)
        
        self.relu = nn.ReLU()

    def forward(self, list_x, list_y, x_size):
        up_score, tmp_feature = [], []
        list_y = list_y[::-1]

        for i, i_x in enumerate(list_x):
            for j, j_x in enumerate(list_y):
                tmp = F.interpolate(self.trans[i][j](j_x), i_x.size()[2:], mode='bilinear', align_corners=True) + i_x
                tmp_f = self.up[i][j](tmp)
                up_score.append(F.interpolate(self.sub_score(tmp_f), x_size, mode='bilinear', align_corners=True))
                tmp_feature.append(tmp_f)


        tmp_fea = tmp_feature[0]
        for i_fea in range(len(tmp_feature) - 1):
            tmp_fea = self.relu(torch.add(tmp_fea, F.interpolate((tmp_feature[i_fea + 1]), tmp_feature[0].size()[2:],
                                                                 mode='bilinear', align_corners=True)))
        up_score.append(F.interpolate(self.sub_score(tmp_fea), x_size, mode='bilinear', align_corners=True))

        return up_score
    
    
class TUN_bone(nn.Module):
    def __init__(self, base_model_cfg, base, merge1_layers, merge2_layers):
        super(TUN_bone, self).__init__()
        self.base_model_cfg = base_model_cfg
        if self.base_model_cfg == 'vgg':
            self.base = base
            # self.base_ex = nn.ModuleList(base_ex)
            self.merge1 = merge1_layers
            self.merge2 = merge2_layers

        elif self.base_model_cfg == 'resnext101':
            self.convert = ConvertLayer(config_resnext101['convert'])
            self.base = base
            self.merge1 = merge1_layers
            self.merge2 = merge2_layers

    def forward(self, x):
        x_size = x.size()[2:]
        conv2merge = self.base(x)
        if self.base_model_cfg == 'resnext101':
            conv2merge = self.convert(conv2merge)
        up_edge, edge_feature, up_sal, sal_feature, up_subitizing = self.merge1(conv2merge, x_size)
        up_sal_final = self.merge2(edge_feature, sal_feature, x_size)
        return up_edge, up_sal, up_subitizing, up_sal_final
    

def conv1x1(in_channels, out_channels, stride=1):
    return nn.Conv2d(in_channels, out_channels, kernel_size=1,
                     stride=stride, padding=0, bias=False)


class Spacial_IRNN(nn.Module):
    def __init__(self, in_channels, alpha=1.0):
        super(Spacial_IRNN, self).__init__()
        self.left_weight = nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, groups=in_channels, padding=0)
        self.right_weight = nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, groups=in_channels, padding=0)
        self.up_weight = nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, groups=in_channels, padding=0)
        self.down_weight = nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, groups=in_channels, padding=0)
        self.left_weight.weight = nn.Parameter(torch.tensor([[[[alpha]]]] * in_channels))
        self.right_weight.weight = nn.Parameter(torch.tensor([[[[alpha]]]] * in_channels))
        self.up_weight.weight = nn.Parameter(torch.tensor([[[[alpha]]]] * in_channels))
        self.down_weight.weight = nn.Parameter(torch.tensor([[[[alpha]]]] * in_channels))

    def forward(self, input):
        return irnn()(input, self.up_weight.weight, self.right_weight.weight, self.down_weight.weight,
                      self.left_weight.weight, self.up_weight.bias, self.right_weight.bias, self.down_weight.bias,
                      self.left_weight.bias)


class Attention(nn.Module):
    def __init__(self, in_channels):
        super(Attention, self).__init__()
        self.out_channels = int(in_channels / 2)
        self.conv1 = nn.Conv2d(in_channels, self.out_channels, kernel_size=3, padding=1, stride=1)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(self.out_channels, self.out_channels, kernel_size=3, padding=1, stride=1)
        self.relu2 = nn.ReLU()
        self.conv3 = nn.Conv2d(self.out_channels, 4, kernel_size=1, padding=0, stride=1)
        self.sigmod = nn.Sigmoid()

    def forward(self, x):
        out = self.conv1(x)
        out = self.relu1(out)
        out = self.conv2(out)
        out = self.relu2(out)
        out = self.conv3(out)
        out = self.sigmod(out)
        return out

    
class DSC_Module(nn.Module):
    def __init__(self, in_channels, out_channels, attention=1, alpha=1.0):
        super(DSC_Module, self).__init__()
        self.out_channels = out_channels
        self.irnn1 = Spacial_IRNN(self.out_channels, alpha)
        self.irnn2 = Spacial_IRNN(self.out_channels, alpha)
        self.conv_in = conv1x1(in_channels, in_channels)
        self.conv2 = conv1x1(in_channels * 4, in_channels)
        self.conv3 = conv1x1(in_channels * 4, in_channels)
        self.relu2 = nn.ReLU(True)
        self.attention = attention
        if self.attention:
            self.attention_layer = Attention(in_channels)

    def forward(self, x):
        if self.attention:
            weight = self.attention_layer(x)
        out = self.conv_in(x)
        top_up, top_right, top_down, top_left = self.irnn1(out)

        # direction attention
        if self.attention:
            top_up.mul(weight[:, 0:1, :, :])
            top_right.mul(weight[:, 1:2, :, :])
            top_down.mul(weight[:, 2:3, :, :])
            top_left.mul(weight[:, 3:4, :, :])
        out = torch.cat([top_up, top_right, top_down, top_left], dim=1)
        out = self.conv2(out)
        top_up, top_right, top_down, top_left = self.irnn2(out)

        # direction attention
        if self.attention:
            top_up.mul(weight[:, 0:1, :, :])
            top_right.mul(weight[:, 1:2, :, :])
            top_down.mul(weight[:, 2:3, :, :])
            top_left.mul(weight[:, 3:4, :, :])

        out = torch.cat([top_up, top_right, top_down, top_left], dim=1)
        out = self.conv3(out)
        out = self.relu2(out)

        return out
    

class LayerConv(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride, padding, relu):
        super(LayerConv, self).__init__()
        self.conv = nn.Conv2d(in_channels=in_planes, out_channels=out_planes, kernel_size=kernel_size,
                              stride=stride, padding=padding)
        self.relu = nn.ReLU() if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.relu is not None:
            x = self.relu(x)

        return x
    

class Decoder(nn.Module):
    def __init__(self, num_classes, backbone, BatchNorm):
        super(Decoder, self).__init__()

        self.last_conv = nn.Sequential(nn.Conv2d(312, 256, kernel_size=3, stride=1, padding=2, bias=False,dilation=2),
                                       BatchNorm(256),
                                       nn.ReLU(),
                                       # nn.Dropout(0.5),
                                       nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False,dilation=1),
                                       BatchNorm(256),
                                       nn.ReLU())
        self.class_conv = nn.Sequential(nn.Conv2d(256, num_classes, kernel_size=1, stride=1))


        self.reduce_low_dsc = nn.Sequential(nn.Conv2d(256, 24, kernel_size=1, stride=1))

        self.beta_low = nn.Parameter(torch.tensor([[[1.0]]]*24),requires_grad=True)

        self._init_weight()





    def forward(self, dsc, low_level_feat, middle_level_feat,high_level_feat):


        low_dsc = self.reduce_low_dsc(dsc)
        low_dsc = F.interpolate(low_dsc, size=low_level_feat.size()[2:], mode='bilinear', align_corners=True)


        low_dis = torch.pow(low_dsc-low_level_feat,2)

        low_distance = torch.log(1+low_dis)

        local_gated_coeffeicent_low = low_distance

        low_level_feat = local_gated_coeffeicent_low * self.beta_low * low_level_feat


        x = F.interpolate(high_level_feat, size=low_level_feat.size()[2:], mode='bilinear', align_corners=True)
        middle_level_feat = F.interpolate(middle_level_feat, size=low_level_feat.size()[2:], mode='bilinear', align_corners=True)
        final_feat = torch.cat((x,middle_level_feat, low_level_feat), dim=1)


        x = self.last_conv(final_feat)
        x = self.class_conv(x)


        return x


    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, SynchronizedBatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
    
    
class FSDNet(nn.Module):
    def __init__(self, backbone='mobilenet', output_stride=8, num_classes=1,
                 sync_bn=True, freeze_bn=False):
        super(FSDNet, self).__init__()
        if backbone == 'drn':
            output_stride = 8

        if sync_bn == True:
            BatchNorm = SynchronizedBatchNorm2d
        else:
            BatchNorm = nn.BatchNorm2d

        self.backbone = build_backbone(backbone, output_stride, BatchNorm)

        self.reduce1 = LayerConv(320, 256, 1, 1, 0, False)

        self.dsc = DSC_Module(256, 256)

        self.reduce2 = LayerConv(576, 256, 1, 1, 0, False)

        self.decoder = Decoder(num_classes, backbone, BatchNorm)


        if freeze_bn:
            self.freeze_bn()

    def forward(self, input):
        high_level_feat, low_level_feat, middle_level_feat = self.backbone(input)

        x = self.reduce1(high_level_feat)
        dsc = self.dsc(x)
        x = self.reduce2(torch.cat((high_level_feat, dsc), 1))

        x = self.decoder(dsc, low_level_feat, middle_level_feat,x) # 256,256,256,256
        x = F.interpolate(x, size=input.size()[2:], mode='bilinear', align_corners=True)
        x = torch.sigmoid(x)

        
        return x


    def freeze_bn(self):
        for m in self.modules():
            if isinstance(m, SynchronizedBatchNorm2d):
                m.eval()
            elif isinstance(m, nn.BatchNorm2d):
                m.eval()

    def get_1x_lr_params(self):
        modules = [self.backbone]
        for i in range(len(modules)):
            for m in modules[i].named_modules():
                if isinstance(m[1], nn.Conv2d) or isinstance(m[1], SynchronizedBatchNorm2d) \
                        or isinstance(m[1], nn.BatchNorm2d):
                    for p in m[1].parameters():
                        if p.requires_grad:
                            yield p

    def get_10x_lr_params(self):
        modules = [self.reduce1, self.reduce2, self.dsc, self.decoder]
        for i in range(len(modules)):
            for m in modules[i].named_modules():
                if isinstance(m[1], nn.Conv2d) or isinstance(m[1], SynchronizedBatchNorm2d) \
                        or isinstance(m[1], nn.BatchNorm2d):
                    for p in m[1].parameters():
                        if p.requires_grad:
                            yield p
