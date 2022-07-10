import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import util.functional as FE
from PWCNet import PWCDCNet
from util.sampler import shift, warp


class SPyBlock(nn.Sequential):

    def __init__(self):
        super().__init__(
            nn.Conv2d(8, 32, kernel_size=7, padding=3),
            nn.ReLU(inplace=False),
            nn.Conv2d(32, 64, kernel_size=7, padding=3),
            nn.ReLU(inplace=False),
            nn.Conv2d(64, 32, kernel_size=7, padding=3),
            nn.ReLU(inplace=False),
            nn.Conv2d(32, 16, kernel_size=7, padding=3),
            nn.ReLU(inplace=False),
            nn.Conv2d(16, 2, kernel_size=7, padding=3)
        )

    def forward(self, flow_course, im1, im2):
        flow = FE.interpolate_as(flow_course, im2, mode='bilinear') * 2.0
        res = super().forward(torch.cat([im1, warp(im2, flow), flow], dim=1))
        flow_fine = res + flow

        return flow_fine


class SPyNet(nn.Module):
    """SPyNet"""

    def __init__(self, level=4, path='./models/spy_net-sintel-final.pytorch', trainable=False):
        super(SPyNet, self).__init__()
        self.level = level
        self.Blocks = nn.ModuleList([SPyBlock() for _ in range(level+1)])

        if path is not None:
            data = torch.load(path, map_location='cpu')
            if 'state_dict' in data.keys():
                self.load_state_dict(data['state_dict'])
            else:
                self.load_state_dict(data)

        self.register_buffer('mean', torch.Tensor(
            [.406, .456, .485]).view(-1, 1, 1))
        self.register_buffer('std', torch.Tensor(
            [.225, .224, .229]).view(-1, 1, 1))

        if not trainable:
            self.requires_grad_(False)

    def norm(self, input):
        return input.sub(self.mean).div(self.std)

    def forward(self, im2, im1):  # backwarp
        # B, 6, H, W
        volume = [torch.cat([self.norm(im1), self.norm(im2)], dim=1)]
        for _ in range(self.level):
            volume.append(F.avg_pool2d(volume[-1], kernel_size=2))

        flows = [torch.zeros_like(volume[-1][:, :2])]  # B, 2, H//16, W//16
        for l, layer in enumerate(self.Blocks):
            flows.append(layer(flows[-1], *volume[self.level-l].chunk(2, 1)))

        return flows[-1]


def norm(input, mean=[.406, .456, .485], std=[.225, .224, .229]):
    return input.sub(input.new_tensor(mean).view(-1, 1, 1)).div(input.new_tensor(std).view(-1, 1, 1))


class SPyNet0(nn.Module):
    """SPyNet"""

    def __init__(self, level=4, path='./models/spy_net-sintel-final.pytorch', trainable=False):
        super(SPyNet0, self).__init__()
        self.level = level
        self.Blocks = nn.ModuleList([SPyBlock() for _ in range(level+1)])

        if path is not None:
            data = torch.load(path, map_location='cpu')
            if 'state_dict' in data.keys():
                self.load_state_dict(data['state_dict'])
            else:
                self.load_state_dict(data)

        if not trainable:
            self.requires_grad_(False)

    def forward(self, im2, im1):  # backwarp
        volume = [torch.cat([norm(im1), norm(im2)], dim=1)]  # B, 6, H, W
        for _ in range(self.level):
            volume.append(F.avg_pool2d(volume[-1], kernel_size=2))

        flows = [torch.zeros_like(volume[-1][:, :2])]  # B, 2, H//16, W//16
        for l, layer in enumerate(self.Blocks):
            flows.append(layer(flows[-1], *volume[self.level-l].chunk(2, 1)))

        return flows[-1]


class PWCNet(PWCDCNet):
    """PWCNet rewrite"""

    def __init__(self, path='./models/pwc_net.pth.tar', trainable=False):
        super().__init__()
        if path is not None:
            data = torch.load(path, map_location='cpu')
            if 'state_dict' in data.keys():
                self.load_state_dict(data['state_dict'])
            else:
                self.load_state_dict(data)

        if not trainable:
            self.requires_grad_(False)

    @staticmethod
    def resize(input, size):
        return F.interpolate(input, size, mode='bilinear', align_corners=False)

    def forward(self, im1, im2, scale=1):
        inputs = torch.cat([im2, im1], dim=1)  # B, 6, H, W

        divisor = 64.
        H, W = inputs.size()[-2:]
        H_ = int(np.ceil(H/divisor) * divisor)
        W_ = int(np.ceil(W/divisor) * divisor)
        if H != H_ or W != W_:
            inputs = self.resize(inputs, size=(H_, W_))  # B, 6, H_, W_

        # print(inputs.shape)
        pwcret = super().forward(inputs)
        flow = pwcret[0] if self.training else pwcret  # B, 2, H_/4, W_/4
        # print(flow.shape)

        flow = self.resize(flow*20.0, size=(H//scale, W//scale))  # B, 2, H, W
        scale = inputs.new_tensor([W/float(W_), H/float(H_)])
        flow = flow * scale.view(2, 1, 1)  # B, 2, H, W
        return flow
