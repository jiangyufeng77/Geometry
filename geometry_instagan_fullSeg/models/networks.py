import torch
import torch.nn as nn
from torch.nn import init
import functools
from torch.optim import lr_scheduler
from torch.nn.parameter import Parameter

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
    elif netG == 'attention':
        net = UGATGenerator(input_nc, output_nc, ngf, img_size=256, norm_layer=norm_layer, use_dropout=use_dropout, n_blocks=4)
    else:
        raise NotImplementedError('Generator model name [%s] is not recognized' % netG)
    return init_net(net, init_type, init_gain, gpu_ids)


def define_D(input_nc, ndf, netD, n_layers_D=3, norm='batch', use_sigmoid=False, init_type='normal', init_gain=0.02, gpu_ids=[]):
    net = None
    norm_layer = get_norm_layer(norm_type=norm)

    if netD == 'basic':
        net = NLayerSetDiscriminator(input_nc, ndf, n_layers=3, norm_layer=norm_layer, use_sigmoid=use_sigmoid)
    elif netD == 'attention':
        net = AttDiscriminator(input_nc, ndf, n_layers=3, norm_layer=norm_layer, use_sigmoid=use_sigmoid)
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


class UGATGenerator(nn.Module):
    def __init__(self, input_nc, output_nc, ngf=64, img_size=256, norm_layer=nn.BatchNorm2d, use_dropout=False, n_blocks=4,
                 padding_type='reflect'):
        assert (n_blocks >= 0)
        super(UGATGenerator, self).__init__()
        self.input_nc = input_nc
        self.output_nc = output_nc
        self.ngf = ngf
        self.light = True

        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        n_downsampling = 2
        self.encoder_img = Encoder(input_nc*2, n_downsampling, ngf, norm_layer, use_dropout, n_blocks,
                                   padding_type, use_bias)
        # self.encoder_seg = Encoder(input_nc, n_downsampling, ngf, norm_layer, use_dropout, n_blocks,
        #                            padding_type, use_bias)
        self.dec = Decoder_img(output_nc, ngf, n_blocks, use_bias)  # 2*ngf
        # self.decoder_seg = Decoder_seg(1, 3*ngf, n_blocks, norm_layer, use_dropout, padding_type, use_bias)  # 3*ngf

        mult = 2 ** n_downsampling
        # Class Activation Map
        self.gap_fc = nn.Linear(ngf * mult, 1, bias=False)
        self.gmp_fc = nn.Linear(ngf * mult, 1, bias=False)
        self.conv1x1 = nn.Conv2d(ngf * mult * 2, ngf * mult, kernel_size=1, stride=1, bias=True)
        self.relu = nn.ReLU(True)

        # Gamma, Beta block
        if self.light:
            FC = [nn.Linear(ngf * mult, ngf * mult, bias=False),
                  nn.ReLU(True),
                  nn.Linear(ngf * mult, ngf * mult, bias=False),
                  nn.ReLU(True)]
        else:
            FC = [nn.Linear(img_size // mult * img_size // mult * ngf * mult, ngf * mult, bias=False),
                  nn.ReLU(True),
                  nn.Linear(ngf * mult, ngf * mult, bias=False),
                  nn.ReLU(True)]
        self.gamma = nn.Linear(ngf * mult, ngf * mult, bias=False)
        self.beta = nn.Linear(ngf * mult, ngf * mult, bias=False)

        self.FC = nn.Sequential(*FC)

    def forward(self, inp):
        # split data
        # img = inp[:, :self.input_nc, :, :]  # (B, CX, W, H)
        # segs = inp[:, self.input_nc:, :, :]  # (B, CA, W, H)

        # run image encoder
        enc_img = self.encoder_img(inp)
        # add attention to image
        gap = torch.nn.functional.adaptive_avg_pool2d(enc_img, 1)
        gap_logit = self.gap_fc(gap.view(enc_img.shape[0], -1))
        gap_weight = list(self.gap_fc.parameters())[0]
        gap = enc_img * gap_weight.unsqueeze(2).unsqueeze(3)

        gmp = torch.nn.functional.adaptive_max_pool2d(enc_img, 1)
        gmp_logit = self.gmp_fc(gmp.view(enc_img.shape[0], -1))
        gmp_weight = list(self.gmp_fc.parameters())[0]
        gmp = enc_img * gmp_weight.unsqueeze(2).unsqueeze(3)

        cam_logit = torch.cat([gap_logit, gmp_logit], 1)
        enc_img = torch.cat([gap, gmp], 1)
        enc_img = self.relu(self.conv1x1(enc_img))

        heatmap = torch.sum(enc_img, dim=1, keepdim=True)

        if self.light:
            x_ = torch.nn.functional.adaptive_avg_pool2d(enc_img, 1)
            x_ = self.FC(x_.view(x_.shape[0], -1))
        else:
            x_ = self.FC(enc_img.view(enc_img.shape[0], -1))
        gamma, beta = self.gamma(x_), self.beta(x_)

        # run segmentation encoder
        # enc_segs = self.encoder_seg(segs)

        # run decoder
        feat = enc_img
        # feat = torch.cat([enc_img, enc_segs], dim=1)
        out = self.dec(feat, gamma, beta)
        return out, cam_logit, heatmap


class Encoder(nn.Module):
    def __init__(self, input_nc, n_downsampling, ngf, norm_layer, use_dropout, n_blocks, padding_type, use_bias):
        super(Encoder, self).__init__()

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
            model += [ResnetBlock(ngf * mult, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout,
                                  use_bias=use_bias)]

        self.model = nn.Sequential(*model)

    def forward(self, x):
        return self.model(x)

class Decoder_img(nn.Module):
    def __init__(self, output_nc, ngf, n_blocks, use_bias):
        super(Decoder_img, self).__init__()

        self.n_blocks = n_blocks
        # Up-Sampling Bottleneck
        n_downsampling = 2
        mult = 2 ** n_downsampling
        for i in range(n_blocks):
            setattr(self, 'UpBlock1_' + str(i + 1), ResnetAdaILNBlock(ngf * mult, use_bias=False))

        # Up-Sampling
        UpBlock2 = []
        for i in range(n_downsampling):
            mult = 2 ** (n_downsampling - i)
            UpBlock2 += [nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2), kernel_size=3, stride=2, padding=1, output_padding=1, bias=use_bias),
                         ILN(int(ngf * mult / 2)),
                         nn.ReLU(True)]

        UpBlock2 += [nn.ReflectionPad2d(3),
                     nn.Conv2d(ngf, output_nc, kernel_size=7, stride=1, padding=0, bias=False),
                     nn.Tanh()]

        self.UpBlock2 = nn.Sequential(*UpBlock2)

    def forward(self, input, gamma, beta):
        for i in range(self.n_blocks):
            x = getattr(self, 'UpBlock1_' + str(i + 1))(input, gamma, beta)
        out = self.UpBlock2(x)
        return out


class ResnetAdaILNBlock(nn.Module):
    def __init__(self, dim, use_bias):
        super(ResnetAdaILNBlock, self).__init__()
        self.pad1 = nn.ReflectionPad2d(1)
        self.conv1 = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=0, bias=use_bias)
        self.norm1 = adaILN(dim)
        self.relu1 = nn.ReLU(True)

        self.pad2 = nn.ReflectionPad2d(1)
        self.conv2 = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=0, bias=use_bias)
        self.norm2 = adaILN(dim)

    def forward(self, x, gamma, beta):
        out = self.pad1(x)
        out = self.conv1(out)
        out = self.norm1(out, gamma, beta)
        out = self.relu1(out)
        out = self.pad2(out)
        out = self.conv2(out)
        out = self.norm2(out, gamma, beta)

        return out + x


class adaILN(nn.Module):
    def __init__(self, num_features, eps=1e-5):
        super(adaILN, self).__init__()
        self.eps = eps
        self.rho = Parameter(torch.Tensor(1, num_features, 1, 1))
        self.rho.data.fill_(0.9)

    def forward(self, input, gamma, beta):
        in_mean, in_var = torch.mean(input, dim=[2, 3], keepdim=True), torch.var(input, dim=[2, 3], keepdim=True)
        out_in = (input - in_mean) / torch.sqrt(in_var + self.eps)
        ln_mean, ln_var = torch.mean(input, dim=[1, 2, 3], keepdim=True), torch.var(input, dim=[1, 2, 3], keepdim=True)
        out_ln = (input - ln_mean) / torch.sqrt(ln_var + self.eps)
        out = self.rho.expand(input.shape[0], -1, -1, -1) * out_in + (1-self.rho.expand(input.shape[0], -1, -1, -1)) * out_ln
        out = out * gamma.unsqueeze(2).unsqueeze(3) + beta.unsqueeze(2).unsqueeze(3)

        return out


class ILN(nn.Module):
    def __init__(self, num_features, eps=1e-5):
        super(ILN, self).__init__()
        self.eps = eps
        self.rho = Parameter(torch.Tensor(1, num_features, 1, 1))
        self.gamma = Parameter(torch.Tensor(1, num_features, 1, 1))
        self.beta = Parameter(torch.Tensor(1, num_features, 1, 1))
        self.rho.data.fill_(0.0)
        self.gamma.data.fill_(1.0)
        self.beta.data.fill_(0.0)

    def forward(self, input):
        in_mean, in_var = torch.mean(input, dim=[2, 3], keepdim=True), torch.var(input, dim=[2, 3], keepdim=True)
        out_in = (input - in_mean) / torch.sqrt(in_var + self.eps)
        ln_mean, ln_var = torch.mean(input, dim=[1, 2, 3], keepdim=True), torch.var(input, dim=[1, 2, 3], keepdim=True)
        out_ln = (input - ln_mean) / torch.sqrt(ln_var + self.eps)
        out = self.rho.expand(input.shape[0], -1, -1, -1) * out_in + (1-self.rho.expand(input.shape[0], -1, -1, -1)) * out_ln
        out = out * self.gamma.expand(input.shape[0], -1, -1, -1) + self.beta.expand(input.shape[0], -1, -1, -1)

        return out


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


class AttDiscriminator(nn.Module):
    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d, use_sigmoid=False):
        super(AttDiscriminator, self).__init__()

        self.input_nc = input_nc
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        kw = 4
        padw = 1
        self.feature_img = Feature_extractor(2*input_nc, ndf, n_layers, kw, padw, norm_layer, use_bias)
        # self.feature_seg = Feature_extractor(input_nc, ndf, n_layers, kw, padw, norm_layer, use_bias)
        self.classifier = Classifier(ndf, n_layers, kw, padw, norm_layer, use_sigmoid)  # 2*ndf

        # Class Activation Map
        mult = 4
        self.gap_fc = nn.utils.spectral_norm(nn.Linear(ndf * mult, 1, bias=False))
        self.gmp_fc = nn.utils.spectral_norm(nn.Linear(ndf * mult, 1, bias=False))
        self.conv1x1 = nn.Conv2d(ndf * mult * 2, ndf * mult, kernel_size=1, stride=1, bias=True)
        self.leaky_relu = nn.LeakyReLU(0.2, True)

    def forward(self, inp):
        # split data
        # img = inp[:, :self.input_nc, :, :]  # (B, CX, W, H)
        # segs = inp[:, self.input_nc:, :, :]  # (B, CA, W, H)

        # run feature extractor for image
        feat_img = self.feature_img(inp)
        # add attention
        gap = torch.nn.functional.adaptive_avg_pool2d(feat_img, 1)
        gap_logit = self.gap_fc(gap.view(feat_img.shape[0], -1))
        gap_weight = list(self.gap_fc.parameters())[0]
        gap = feat_img * gap_weight.unsqueeze(2).unsqueeze(3)

        gmp = torch.nn.functional.adaptive_max_pool2d(feat_img, 1)
        gmp_logit = self.gmp_fc(gmp.view(feat_img.shape[0], -1))
        gmp_weight = list(self.gmp_fc.parameters())[0]
        gmp = feat_img * gmp_weight.unsqueeze(2).unsqueeze(3)

        cam_logit = torch.cat([gap_logit, gmp_logit], 1)
        feat_img = torch.cat([gap, gmp], 1)
        feat_img = self.leaky_relu(self.conv1x1(feat_img))

        heatmap = torch.sum(feat_img, dim=1, keepdim=True)

        # feature exactor for segmentation
        # feat_segs = self.feature_seg(segs)

        # run classifier
        feat = feat_img
        # feat = torch.cat([feat_img, feat_segs], dim=1)
        out = self.classifier(feat)
        return out, cam_logit, heatmap

class Feature_extractor(nn.Module):
    def __init__(self, input_nc, ndf, n_layers, kw, padw, norm_layer, use_bias):
        super(Feature_extractor, self).__init__()

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
        self.model = nn.Sequential(*model)

    def forward(self, input):
        return self.model(input)
    
class Classifier(nn.Module):
    def __init__(self, ndf, n_layers, kw, padw, norm_layer, use_sigmoid):
        super(Classifier, self).__init__()

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
        self.model = nn.Sequential(*model)

    def forward(self, input):
        return self.model(input)

class RhoClipper(object):

    def __init__(self, min, max):
        self.clip_min = min
        self.clip_max = max
        assert min < max

    def __call__(self, module):

        if hasattr(module, 'rho'):
            w = module.rho.data
            w = w.clamp(self.clip_min, self.clip_max)
            module.rho.data = w