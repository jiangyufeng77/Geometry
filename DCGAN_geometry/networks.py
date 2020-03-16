import torch.nn as nn
import os
import torch
import numpy as np
import torch.nn.functional as f


class autoencoder(nn.Module):
    def __init__(self):
        super(autoencoder, self).__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, 3, 1, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # channels * 64 * 64->64*32*32
            nn.Conv2d(64, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            # 64*32*32->128*16*16
            nn.Conv2d(128, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            # 128*16*16->256*8*8
            nn.Conv2d(256, 512, 4, 2, 1, bias=False),
            nn.BatchNorm2d(512, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            # 256*8*8->512*4*4
            nn.Conv2d(512, 1024, 4, 2, 1, bias=False),
            nn.BatchNorm2d(1024, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            # 512 * 4 * 4 -> 1*1*1
            nn.Conv2d(1024, 100, 4, 1, 0, bias=False)
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(100*2, 1024, kernel_size=4, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(1024, 0.8),
            nn.ReLU(),

            nn.ConvTranspose2d(1024, 512, 4, 2, 1, bias=False),
            nn.BatchNorm2d(512, 0.8),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(512, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256, 0.8),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128, 0.8),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64, 0.8),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(64, 3, 3, 1, 1, bias=False),
            nn.Tanh()
        )

        self.geo = GeoTransform()

    def forward(self, x, output_size, random_affine):
        geo_out = self.geo.forward(x, output_size, random_affine)
        enc_outT = self.encoder(geo_out)
        geo_T_out = self.geo.forward(enc_outT, output_size, -random_affine)
        print(geo_T_out.shape)
        enc_out = self.encoder(x)
        print(enc_out.shape)
        dec_in = torch.cat((geo_T_out, enc_out), dim=1)
        dec_out = self.decoder(dec_in)
        return dec_out


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