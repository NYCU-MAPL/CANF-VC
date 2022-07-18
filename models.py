import torch
import torch.nn as nn
import torch.nn.functional as F

import util.functional as FE
from util.sampler import shift, warp, Resampler


class ResidualBlock(nn.Sequential):
    """Builds the residual block"""

    def __init__(self, num_filters):
        super(ResidualBlock, self).__init__(
            nn.ReLU(inplace=False),
            nn.Conv2d(num_filters, num_filters, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(num_filters, num_filters, 3, padding=1)
        )

    def forward(self, input):
        return input + super().forward(input)


class Refinement(nn.Module):
    """Refinement UNet"""

    def __init__(self, in_channels, num_filters, out_channels=3):
        super().__init__()
        self.l1 = nn.Sequential(
            nn.Conv2d(in_channels, num_filters, 3, padding=1),
            ResidualBlock(num_filters)
        )
        self.l2 = nn.Sequential(
            nn.AvgPool2d(kernel_size=2),
            ResidualBlock(num_filters)
        )
        self.l3 = nn.Sequential(
            nn.AvgPool2d(kernel_size=2),
            ResidualBlock(num_filters)
        )

        self.d3 = nn.Sequential(
            ResidualBlock(num_filters),
            nn.UpsamplingBilinear2d(scale_factor=2)
        )
        self.d2 = nn.Sequential(
            ResidualBlock(num_filters),
            nn.UpsamplingBilinear2d(scale_factor=2)
        )
        self.d1 = nn.Sequential(
            ResidualBlock(num_filters),
            nn.Conv2d(num_filters, num_filters, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(num_filters, out_channels, 3, padding=1)
        )

    def forward(self, *input):
        if len(input) == 1:
            input = input[0]
        else:
            input = torch.cat(input, dim=1)

        conv1 = self.l1(input)
        conv2 = self.l2(conv1)
        conv3 = self.l3(conv2)

        deconv3 = self.d3(conv3)
        deconv2 = self.d2(deconv3 + conv2)
        deconv1 = self.d1(deconv2 + conv1)

        return deconv1


class Post(nn.Module):
    def __init__(self, in_channels, num_filters):
        super().__init__()
        self.l1 = nn.Sequential(
            nn.Conv2d(in_channels, num_filters, 3, padding=1),
            ResidualBlock(num_filters)
        )
        self.l2 = nn.Sequential(
            nn.AvgPool2d(kernel_size=2),
            ResidualBlock(num_filters)
        )
        self.l3 = nn.Sequential(
            nn.AvgPool2d(kernel_size=2),
            ResidualBlock(num_filters)
        )

        self.d3 = nn.Sequential(
            ResidualBlock(num_filters),
            nn.UpsamplingBilinear2d(scale_factor=2)
        )
        self.d2 = nn.Sequential(
            ResidualBlock(num_filters*2),
            nn.UpsamplingBilinear2d(scale_factor=2)
        )
        self.d1 = nn.Sequential(
            ResidualBlock(num_filters*3),
            nn.Conv2d(num_filters*3, num_filters*2, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(num_filters*2, num_filters, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(num_filters, 3, 3, padding=1)
        )

    def forward(self, inputs):
        conv1 = self.l1(inputs)
        conv2 = self.l2(conv1)
        conv3 = self.l3(conv2)

        deconv3 = self.d3(conv3)
        deconv2 = self.d2(torch.cat([deconv3, conv2], dim=1))
        deconv1 = self.d1(torch.cat([deconv2, conv1], dim=1))

        return deconv1 + inputs


class MC_Net(Refinement):
    """MC_Net"""

    def __init__(self, num_filters=64, use_flow=False, stop_flow=False, obmc=False, path=None):
        self.use_flow = use_flow
        self.stop_flow = stop_flow
        in_channels = 6
        if use_flow:
            in_channels += 2
        if obmc:
            in_channels += OPMC.aux_channel[obmc]

        super(MC_Net, self).__init__(in_channels, num_filters)
        self.warp = Resampler() if obmc == 0 else OPMC(obmc)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, previous_frame, Flow):
        warped = self.warp(previous_frame, Flow)

        mc_input = [previous_frame, Flow.detach() if self.stop_flow else Flow, warped] if self.use_flow else [
            previous_frame, warped]

        return super().forward(*mc_input)

