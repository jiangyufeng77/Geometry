import torch
import torch.nn as nn
from torch.nn import init
import functools
from torch.optim import lr_scheduler
from torch.nn.parameter import Parameter
import numpy as np
import torch.nn.functional as f

###############################################################################
# Helper Functions
###############################################################################


def get_norm_layer(norm_type='instance'):
    if norm_type == 'batch':
        norm_layer = functools.partial(nn.BatchNorm2d, affine=True)
    elif norm_type == 'instance':
        norm_layer = functools.partial(nn.InstanceNorm2d, affine=False, track_running_stats=False)
    elif norm_type == 'none':
        norm_layer = None
    else:
        raise NotImplementedError('normalization layer [%s] is not found' % norm_type)
    return norm_layer


def get_scheduler(optimizer, opt):
    if opt.lr_policy == 'lambda':
        def lambda_rule(epoch):
            lr_l = 1.0 - max(0, epoch + opt.epoch_count - opt.niter) / float(opt.niter_decay + 1)
            return lr_l
        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
    elif opt.lr_policy == 'step':
        scheduler = lr_scheduler.StepLR(optimizer, step_size=opt.lr_decay_iters, gamma=0.1)
    elif opt.lr_policy == 'plateau':
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, threshold=0.01, patience=5)
    elif opt.lr_policy == 'cosine':
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=opt.niter, eta_min=0)
    else:
        return NotImplementedError('learning rate policy [%s] is not implemented', opt.lr_policy)
    return scheduler


def init_weights(net, init_type='normal', gain=0.02):
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:
            init.normal_(m.weight.data, 1.0, gain)
            init.constant_(m.bias.data, 0.0)

    print('initialize network with %s' % init_type)
    net.apply(init_func)


def init_net(net, init_type='normal', init_gain=0.02, gpu_ids=[]):
    if len(gpu_ids) > 0:
        assert(torch.cuda.is_available())
        net.to(gpu_ids[0])
        net = torch.nn.DataParallel(net, gpu_ids)
    init_weights(net, init_type, gain=init_gain)
    return net


def define_G(input_nc, output_nc, ngf, netG, norm='batch', use_dropout=False, init_type='normal', init_gain=0.02, gpu_ids=[]):
    net = None
    norm_layer = get_norm_layer(norm_type=norm)

    if netG == 'basic':
        net = ResnetSetGenerator(input_nc, output_nc, ngf, norm_layer=norm_layer, use_dropout=use_dropout, n_blocks=9)
    elif netG == 'transformation':
        net = TransGenerator(input_nc, output_nc, ngf, img_size=256, norm_layer=norm_layer, use_dropout=use_dropout, n_blocks=4)
    else:
        raise NotImplementedError('Generator model name [%s] is not recognized' % netG)
    return init_net(net, init_type, init_gain, gpu_ids)


def define_D(input_nc, ndf, netD, n_layers_D=3, norm='batch', use_sigmoid=False, init_type='normal', init_gain=0.02, gpu_ids=[]):
    net = None
    norm_layer = get_norm_layer(norm_type=norm)

    if netD == 'basic':
        net = NLayerDiscriminator(input_nc, ndf, n_layers=3, norm_layer=norm_layer, use_sigmoid=use_sigmoid)
    elif netD == 'instagan':
        net = NLayerSetDiscriminator(input_nc, ndf, n_layers=3, norm_layer=norm_layer, use_sigmoid=use_sigmoid)
    else:
        raise NotImplementedError('Discriminator model name [%s] is not recognized' % net)
    return init_net(net, init_type, init_gain, gpu_ids)


##############################################################################
# Classes
##############################################################################


# Defines the GAN loss which uses either LSGAN or the regular GAN.
# When LSGAN is used, it is basically same as MSELoss,
# but it abstracts away the need to create the target label tensor
# that has the same size as the input
class GANLoss(nn.Module):
    def __init__(self, use_lsgan=True, target_real_label=1.0, target_fake_label=0.0):
        super(GANLoss, self).__init__()
        self.register_buffer('real_label', torch.tensor(target_real_label))
        self.register_buffer('fake_label', torch.tensor(target_fake_label))
        if use_lsgan:
            self.loss = nn.MSELoss()
        else:
            self.loss = nn.BCELoss()

    def get_target_tensor(self, input, target_is_real):
        if target_is_real:
            target_tensor = self.real_label
        else:
            target_tensor = self.fake_label
        return target_tensor.expand_as(input)

    def __call__(self, input, target_is_real):
        target_tensor = self.get_target_tensor(input, target_is_real)
        return self.loss(input, target_tensor)


# Define spectral normalization layer
# Code from Christian Cosgrove's repository
# https://github.com/christiancosgrove/pytorch-spectral-normalization-gan/blob/master/spectral_normalization.py
def l2normalize(v, eps=1e-12):
    return v / (v.norm() + eps)

class SpectralNorm(nn.Module):
    def __init__(self, module, name='weight', power_iterations=1):
        super(SpectralNorm, self).__init__()
        self.module = module
        self.name = name
        self.power_iterations = power_iterations
        if not self._made_params():
            self._make_params()

    def _update_u_v(self):
        u = getattr(self.module, self.name + "_u")
        v = getattr(self.module, self.name + "_v")
        w = getattr(self.module, self.name + "_bar")

        height = w.data.shape[0]
        for _ in range(self.power_iterations):
            v.data = l2normalize(torch.mv(torch.t(w.view(height, -1).data), u.data))
            u.data = l2normalize(torch.mv(w.view(height, -1).data, v.data))

        sigma = u.dot(w.view(height, -1).mv(v))
        setattr(self.module, self.name, w / sigma.expand_as(w))

    def _made_params(self):
        try:
            u = getattr(self.module, self.name + "_u")
            v = getattr(self.module, self.name + "_v")
            w = getattr(self.module, self.name + "_bar")
            return True
        except AttributeError:
            return False

    def _make_params(self):
        w = getattr(self.module, self.name)

        height = w.data.shape[0]
        width = w.view(height, -1).data.shape[1]

        u = nn.Parameter(w.data.new(height).normal_(0, 1), requires_grad=False)
        v = nn.Parameter(w.data.new(width).normal_(0, 1), requires_grad=False)
        u.data = l2normalize(u.data)
        v.data = l2normalize(v.data)
        w_bar = nn.Parameter(w.data)

        del self.module._parameters[self.name]

        self.module.register_parameter(self.name + "_u", u)
        self.module.register_parameter(self.name + "_v", v)
        self.module.register_parameter(self.name + "_bar", w_bar)

    def forward(self, *args):
        self._update_u_v()
        return self.module.forward(*args)


class TransGenerator(nn.Module):
    def __init__(self, input_nc, output_nc, ngf=64, img_size=256, norm_layer=nn.BatchNorm2d, use_dropout=False, n_blocks=4,
                 padding_type='reflect'):
        assert (n_blocks >= 0)
        super(TransGenerator, self).__init__()
        self.input_nc = input_nc
        self.output_nc = output_nc
        self.ngf = ngf
        self.light = True

        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        n_downsampling = 2
        self.enc = Encoder(input_nc+1, n_downsampling, ngf, norm_layer, use_dropout, n_blocks,
                                   padding_type, use_bias)
        # self.encoder_seg = Encoder(1, n_downsampling, ngf, norm_layer, use_dropout, n_blocks,
        #                            padding_type, use_bias)
        self.dec = Decoder_img(output_nc+1, ngf, norm_layer, use_bias)  # 2*ngf
        # self.decoder_seg = Decoder_seg(1, 3*ngf, n_blocks, norm_layer, use_dropout, padding_type, use_bias)  # 3*ngf
        self.geo = GeoTransform()


    def forward(self, img, segs, output_size, random_affine):
        # run image encoder
        # print(inp.shape)
        img_T = self.geo.forward(img, output_size, random_affine)
        segs_T = self.geo.forward(segs, output_size, random_affine)
        enc_input = torch.cat([img_T, segs_T], dim=1)
        enc_img = self.enc(enc_input)

        enc_img_T = self.geo.forward(enc_img, enc_img.shape[2:], -random_affine)

        # run decoder
        out = self.dec(enc_img_T)

        return out


class Encoder(nn.Module):
    def __init__(self, inp_nc, n_downsampling, ngf, norm_layer, use_dropout, n_blocks, padding_type, use_bias):
        super(Encoder, self).__init__()

        model = [nn.ReflectionPad2d(3),
                 nn.Conv2d(inp_nc, ngf, kernel_size=7, padding=0, bias=use_bias),
                 norm_layer(ngf),
                 nn.ReLU(True)]

        for i in range(n_downsampling):
            mult = 2 ** i
            model += [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=2, padding=1, bias=use_bias),
                      norm_layer(ngf * mult * 2),
                      nn.ReLU(True)]

        mult = 2 ** n_downsampling
        for i in range(n_blocks):
            model += [ResnetBlock(ngf * mult, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout,
                                  use_bias=use_bias)]

        self.model = nn.Sequential(*model)

    def forward(self, x):
        return self.model(x)

class Decoder_img(nn.Module):
    def __init__(self, output_nc, ngf, norm_layer, use_bias):
        super(Decoder_img, self).__init__()

        model = []
        n_downsampling = 2
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

        self.up = nn.Sequential(*model)

    def forward(self, input):
        return self.up(input)

class GeoTransform(nn.Module):
    def __init__(self):
        super(GeoTransform, self).__init__()

    def forward(self, input_tensor, target_size, shifts):
        sz = input_tensor.shape
        theta = homography_based_on_top_corners_x_shift(shifts)

        pad = f.pad(input_tensor, (np.abs(np.int(np.ceil(sz[3] * shifts[0]))), np.abs(np.int(np.ceil(-sz[3] * shifts[1]))), 0, 0), 'reflect')
        target_size4d = torch.Size([pad.shape[0], pad.shape[1], target_size[0], target_size[1]])

        grid = homography_grid(theta.expand(pad.shape[0], -1, -1), target_size4d)

        return f.grid_sample(pad, grid, mode='bilinear', padding_mode='border')

def homography_based_on_top_corners_x_shift(rand_h):
    p = np.array([[1., 1., -1, 0, 0, 0, -(-1. + rand_h[0]), -(-1. + rand_h[0]), -1. + rand_h[0]],
                  [0, 0, 0, 1., 1., -1., 1., 1., -1.],
                  [-1., -1., -1, 0, 0, 0, 1 + rand_h[1], 1 + rand_h[1], 1 + rand_h[1]],
                  [0, 0, 0, -1, -1, -1, 1, 1, 1],
                  [1, 0, -1, 0, 0, 0, 1, 0, -1],
                  [0, 0, 0, 1, 0, -1, 0, 0, 0],
                  [-1, 0, -1, 0, 0, 0, 1, 0, 1],
                  [0, 0, 0, -1, 0, -1, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0, 0, 0, 1]], dtype=np.float32)
    b = np.zeros((9, 1), dtype=np.float32)
    b[8, 0] = 1.
    h = np.dot(np.linalg.inv(p), b)
    return torch.from_numpy(h).view(3, 3).cuda()

def homography_grid(theta, size):
    r"""Generates a 2d flow field, given a batch of homography matrices :attr:`theta`
    Generally used in conjunction with :func:`grid_sample` to
    implement Spatial Transformer Networks.

    Args:
        theta (Tensor): input batch of homography matrices (:math:`N \times 3 \times 3`)
        size (torch.Size): the target output image size (:math:`N \times C \times H \times W`)
                           Example: torch.Size((32, 3, 24, 24))

    Returns:
        output (Tensor): output Tensor of size (:math:`N \times H \times W \times 2`)
    """
    a = 1
    b = 1
    y, x = torch.meshgrid((torch.linspace(-b, b, np.int(size[-2]*a)), torch.linspace(-b, b, np.int(size[-1]*a))))
    n = np.int(size[-2] * a) * np.int(size[-1] * a)
    hxy = torch.ones(n, 3, dtype=torch.float)
    hxy[:, 0] = x.contiguous().view(-1)
    hxy[:, 1] = y.contiguous().view(-1)
    out = hxy[None, ...].cuda().matmul(theta.transpose(1, 2))
    # normalize
    out = out[:, :, :2] / out[:, :, 2:]
    return out.view(theta.shape[0], np.int(size[-2]*a), np.int(size[-1]*a), 2)


def random_size(orig_size, curriculum=True, i=None, iter_for_max_range=None, must_divide=8.0,
                min_scale=0.25, max_scale=2.0, max_transform_magniutude=0.3):
    cur_max_scale = 1.0 + (max_scale - 1.0) * np.clip(1.0 * i / iter_for_max_range, 0, 1) if curriculum else max_scale
    cur_min_scale = 1.0 + (min_scale - 1.0) * np.clip(1.0 * i / iter_for_max_range, 0, 1) if curriculum else min_scale
    cur_max_transform_magnitude = (max_transform_magniutude * np.clip(1.0 * i / iter_for_max_range, 0, 1)
                                   if curriculum else max_transform_magniutude)

    # set random transformation magnitude. scalar = affine, pair = homography.
    random_affine = -cur_max_transform_magnitude + 2 * cur_max_transform_magnitude * np.random.rand(2)

    # set new size for the output image
    new_size = np.array(orig_size) * (cur_min_scale + (cur_max_scale - cur_min_scale) * np.random.rand(2))

    return tuple(np.uint32(np.ceil(new_size * 1.0 / must_divide) * must_divide)), random_affine


# ResNet generator for "set" of instance attributes
# See https://openreview.net/forum?id=ryxwJhC9YX for details
class ResnetSetGenerator(nn.Module):
    def __init__(self, input_nc, output_nc, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False, n_blocks=6, padding_type='reflect'):
        assert (n_blocks >= 0)
        super(ResnetSetGenerator, self).__init__()
        self.input_nc = input_nc
        self.output_nc = output_nc
        self.ngf = ngf
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        n_downsampling = 2
        self.encoder_img = self.get_encoder(input_nc, n_downsampling, ngf, norm_layer, use_dropout, n_blocks, padding_type, use_bias)
        self.encoder_seg = self.get_encoder(1, n_downsampling, ngf, norm_layer, use_dropout, n_blocks, padding_type, use_bias)
        self.decoder_img = self.get_decoder(output_nc, n_downsampling, 2 * ngf, norm_layer, use_bias)  # 2*ngf
        self.decoder_seg = self.get_decoder(1, n_downsampling, 3 * ngf, norm_layer, use_bias)  # 3*ngf

    def get_encoder(self, input_nc, n_downsampling, ngf, norm_layer, use_dropout, n_blocks, padding_type, use_bias):
        model = [nn.ReflectionPad2d(3),
                 nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0, bias=use_bias),
                 norm_layer(ngf),
                 nn.ReLU(True)]

        for i in range(n_downsampling):
            mult = 2 ** i
            model += [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=2, padding=1, bias=use_bias),
                      norm_layer(ngf * mult * 2),
                      nn.ReLU(True)]

        mult = 2 ** n_downsampling
        for i in range(n_blocks):
            model += [ResnetBlock(ngf * mult, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout, use_bias=use_bias)]

        return nn.Sequential(*model)

    def get_decoder(self, output_nc, n_downsampling, ngf, norm_layer, use_bias):
        model = []
        for i in range(n_downsampling):
            mult = 2**(n_downsampling - i)
            model += [nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2), kernel_size=3, stride=2, padding=1, output_padding=1, bias=use_bias),
                      norm_layer(int(ngf * mult / 2)),
                      nn.ReLU(True)]
        model += [nn.ReflectionPad2d(3)]
        model += [nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0)]
        model += [nn.Tanh()]
        return nn.Sequential(*model)

    def forward(self, inp):
        # split data
        img = inp[:, :self.input_nc, :, :]  # (B, CX, W, H)
        segs = inp[:, self.input_nc:, :, :]  # (B, CA, W, H)
        mean = (segs + 1).mean(0).mean(-1).mean(-1)
        if mean.sum() == 0:
            mean[0] = 1  # forward at least one segmentation

        # run encoder
        enc_img = self.encoder_img(img)
        enc_segs = list()
        for i in range(segs.size(1)):
            if mean[i] > 0:  # skip empty segmentation
                seg = segs[:, i, :, :].unsqueeze(1)
                enc_segs.append(self.encoder_seg(seg))
        enc_segs = torch.cat(enc_segs)
        enc_segs_sum = torch.sum(enc_segs, dim=0, keepdim=True)  # aggregated set feature

        # run decoder
        feat = torch.cat([enc_img, enc_segs_sum], dim=1)
        out = [self.decoder_img(feat)]
        idx = 0
        for i in range(segs.size(1)):
            if mean[i] > 0:
                enc_seg = enc_segs[idx].unsqueeze(0)  # (1, ngf, w, h)
                idx += 1  # move to next index
                feat = torch.cat([enc_seg, enc_img, enc_segs_sum], dim=1)
                out += [self.decoder_seg(feat)]
            else:
                out += [segs[:, i, :, :].unsqueeze(1)]  # skip empty segmentation
        return torch.cat(out, dim=1)


# Define a resnet block
class ResnetBlock(nn.Module):
    def __init__(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        super(ResnetBlock, self).__init__()
        self.conv_block = self.build_conv_block(dim, padding_type, norm_layer, use_dropout, use_bias)

    def build_conv_block(self, dim, padding_type, norm_layer, use_dropout, use_bias):
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

        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias),
                       norm_layer(dim),
                       nn.ReLU(True)]
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
        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias),
                       norm_layer(dim)]

        return nn.Sequential(*conv_block)

    def forward(self, x):
        out = x + self.conv_block(x)
        return out


class NLayerDiscriminator(nn.Module):
    """Defines a PatchGAN discriminator"""

    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d, use_sigmoid=False):
        """Construct a PatchGAN discriminator

        Parameters:
            input_nc (int)  -- the number of channels in input images
            ndf (int)       -- the number of filters in the last conv layer
            n_layers (int)  -- the number of conv layers in the discriminator
            norm_layer      -- normalization layer
        """
        super(NLayerDiscriminator, self).__init__()
        self.input_nc = input_nc
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        kw = 4
        padw = 1
        self.feature_img = self.get_feature_extractor(input_nc, ndf, n_layers, kw, padw, norm_layer, use_bias)
        self.feature_seg = self.get_feature_extractor(1, ndf, n_layers, kw, padw, norm_layer, use_bias)
        self.classifier = self.get_classifier(2 * ndf, n_layers, kw, padw, norm_layer, use_sigmoid)  # 2*ndf

    def get_feature_extractor(self, input_nc, ndf, n_layers, kw, padw, norm_layer, use_bias):
        model = [
            # Use spectral normalization
            SpectralNorm(nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw)),
            nn.LeakyReLU(0.2, True)
        ]
        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            model += [
                # Use spectral normalization
                SpectralNorm(nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=2, padding=padw,
                                       bias=use_bias)),
                norm_layer(ndf * nf_mult),
                nn.LeakyReLU(0.2, True)
            ]
        return nn.Sequential(*model)

    def get_classifier(self, ndf, n_layers, kw, padw, norm_layer, use_sigmoid):
        nf_mult_prev = min(2 ** (n_layers - 1), 8)
        nf_mult = min(2 ** n_layers, 8)
        model = [
            # Use spectral normalization
            SpectralNorm(nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=1, padding=padw)),
            norm_layer(ndf * nf_mult),
            nn.LeakyReLU(0.2, True)
        ]
        # Use spectral normalization
        model += [SpectralNorm(nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw))]
        if use_sigmoid:
            model += [nn.Sigmoid()]
        return nn.Sequential(*model)

    def forward(self, inp):
        """Standard forward."""
        # split data
        img = inp[:, :self.input_nc, :, :]  # (B, CX, W, H)
        segs = inp[:, self.input_nc:, :, :]  # (B, CA, W, H)

        feat_img = self.feature_img(img)
        feat_seg = self.feature_seg(segs)

        feat = torch.cat([feat_img, feat_seg], 1)
        out = self.classifier(feat)

        return out


# PatchGAN discriminator for "set" of instance attributes
# See https://openreview.net/forum?id=ryxwJhC9YX for details
class NLayerSetDiscriminator(nn.Module):
    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d, use_sigmoid=False):
        super(NLayerSetDiscriminator, self).__init__()
        self.input_nc = input_nc
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        kw = 4
        padw = 1
        self.feature_img = self.get_feature_extractor(input_nc, ndf, n_layers, kw, padw, norm_layer, use_bias)
        self.feature_seg = self.get_feature_extractor(1, ndf, n_layers, kw, padw, norm_layer, use_bias)
        self.classifier = self.get_classifier(2 * ndf, n_layers, kw, padw, norm_layer, use_sigmoid)  # 2*ndf

    def get_feature_extractor(self, input_nc, ndf, n_layers, kw, padw, norm_layer, use_bias):
        model = [
            # Use spectral normalization
            SpectralNorm(nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw)),
            nn.LeakyReLU(0.2, True)
        ]
        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            model += [
                # Use spectral normalization
                SpectralNorm(nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=2, padding=padw, bias=use_bias)),
                norm_layer(ndf * nf_mult),
                nn.LeakyReLU(0.2, True)
            ]
        return nn.Sequential(*model)

    def get_classifier(self, ndf, n_layers, kw, padw, norm_layer, use_sigmoid):
        nf_mult_prev = min(2 ** (n_layers-1), 8)
        nf_mult = min(2 ** n_layers, 8)
        model = [
            # Use spectral normalization
            SpectralNorm(nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=1, padding=padw)),
            norm_layer(ndf * nf_mult),
            nn.LeakyReLU(0.2, True)
        ]
        # Use spectral normalization
        model += [SpectralNorm(nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw))]
        if use_sigmoid:
            model += [nn.Sigmoid()]
        return nn.Sequential(*model)

    def forward(self, inp):
        # split data
        img = inp[:, :self.input_nc, :, :]  # (B, CX, W, H)
        segs = inp[:, self.input_nc:, :, :]  # (B, CA, W, H)
        mean = (segs + 1).mean(0).mean(-1).mean(-1)
        if mean.sum() == 0:
            mean[0] = 1  # forward at least one segmentation

        # run feature extractor
        feat_img = self.feature_img(img)
        feat_segs = list()
        for i in range(segs.size(1)):
            if mean[i] > 0:  # skip empty segmentation
                seg = segs[:, i, :, :].unsqueeze(1)
                feat_segs.append(self.feature_seg(seg))
        feat_segs_sum = torch.sum(torch.stack(feat_segs), dim=0)  # aggregated set feature

        # run classifier
        feat = torch.cat([feat_img, feat_segs_sum], dim=1)
        out = self.classifier(feat)
        return out


