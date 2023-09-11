### Copyright (C) 2017 NVIDIA Corporation. All rights reserved. 
### Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
from typing import List
import torch
import torch.nn as nn
from DCVC_utils import DepthConvBlock, DepthConvBlockUpsample


# Residual Block
def ResidualBlock(in_channels, out_channels, stride=1, use_depthconv=False):
    if use_depthconv:
        return torch.nn.Sequential(
            nn.PReLU(),
            DepthConvBlock(in_channels, out_channels),
            nn.PReLU(),
            DepthConvBlock(out_channels, out_channels)
        )
    else:
        return torch.nn.Sequential(
            nn.PReLU(),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=True),
            nn.PReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=True)
        )


# downsample block
def DownsampleBlock(in_channels, out_channels, stride=2, use_depthconv=False):
    if use_depthconv:
        return torch.nn.Sequential(
            nn.PReLU(),
            DepthConvBlock(in_channels, out_channels, stride=2),
            nn.PReLU(),
            DepthConvBlock(out_channels, out_channels),
        )
    else:
        return torch.nn.Sequential(
            nn.PReLU(),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=True),
            nn.PReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=True)
        )


# upsample block
def UpsampleBlock(in_channels, out_channels, stride=1, use_depthconv=False):
    if use_depthconv:
        return torch.nn.Sequential(
            nn.PReLU(),
            DepthConvBlockUpsample(in_channels, out_channels),
            nn.PReLU(),
            DepthConvBlock(out_channels, out_channels),
        )
    else:
        return torch.nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.PReLU(),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=True),
            nn.PReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=True),
        )


def FeatBlock(in_channels, out_channels, stride=2):
	return torch.nn.Sequential(
		nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=True),
        nn.PReLU(),
		nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=True),
        nn.PReLU()
	)


class ColumnBlock(nn.Module):
    def __init__(self, channels: List, down: bool, use_depthconv=False) -> None:
        super(ColumnBlock, self).__init__()
        self.down = down
        
        if down:
            bridge = DownsampleBlock 
        else:
            bridge = UpsampleBlock
            channels = channels[::-1]
        
        self.resblocks = nn.ModuleList([ResidualBlock(c, c, stride=1, use_depthconv=use_depthconv) 
                                        for c in channels])
        self.bridge = nn.ModuleList([bridge(cin, cout, use_depthconv=use_depthconv) 
                                        for cin, cout in zip(channels[:-1], channels[1:])])

    def forward(self, inputs) -> List:
        outputs = []

        if not self.down:
            inputs = inputs[::-1]

        for i, x in enumerate(inputs):
            out = self.resblocks[i](x)

            if i > 0:
                out += self.bridge[i-1](outputs[-1])

            outputs.append(out)

        if not self.down:
            outputs = outputs[::-1]

        return outputs


class Backbone(nn.Module):
    def __init__(self, hidden_channels: List) -> None:
        super().__init__()
        self.backbone = nn.ModuleList([FeatBlock(cin, cout, stride=1 if cin == 3 else 2)
                                       for cin, cout in zip(hidden_channels[:-1], hidden_channels[1:])])

    def forward(self, x):
        feats = []
        for m in self.backbone:
            feats.append(m(x))
            x = feats[-1]

        return feats


class GridNet(nn.Module): # GridNet([6, 64, 128, 192], [32, 64, 96], 6, 3)
    def __init__(self, in_channels: List, hidden_channels: List, columns, out_channels: int, use_depthconv=False):
        super(GridNet, self).__init__()
        self.heads = nn.ModuleList([ResidualBlock(i, c, stride=1, use_depthconv=use_depthconv)
                                    for i, c in zip(in_channels, [hidden_channels[0]] + hidden_channels)])
        
        self.downs = nn.ModuleList([nn.Identity()])
        self.downs.extend([DownsampleBlock(cin, cout, use_depthconv=use_depthconv) 
                           for cin, cout in zip(hidden_channels[:-1], hidden_channels[1:])])
    
        columns -= 1 # minus 1 for heads
        self.columns = nn.Sequential(*[ColumnBlock(hidden_channels, n < columns//2, use_depthconv=use_depthconv) for n in range(columns)])
        self.tail = ResidualBlock(hidden_channels[0], out_channels, stride=1)

    def forward(self, inputs):        
        feats = []
        for i, x in enumerate(inputs):
            feat = self.heads[i](x)
            if i > 0:
                feat += self.downs[i-1](feats[-1]) 
            
            feats.append(feat)

        feats.pop(0) # head feat of image has been added into feat-1
        feats = self.columns(feats)
        output = self.tail(feats[0])

        return output, feats


if __name__ == "__main__":
    """
    Test GridNet & Feature extractor
    """

    def get_size(net):
        sum = 0.
        for p, v in net.named_parameters():
            m = v.numel() / 1000000
            print("Key: {0}, param shape: {1}, params: {2:.3f}M".format(p, list(v.size()), m))
            sum += m

        print("======== Total =======")
        print(f"{sum:.3f}M params")

    device = "cuda"
    gridnet = GridNet([6, 64, 128, 192], [32, 64, 96], 6, 3).to(device) # 2.6M params
    featnet = Backbone([3, 32, 64, 96]).to(device) # 0.2M params

    # print("Number of parater of GridNet: ")
    # get_size(gridnet)
    
    # print("Number of parater of feature extractor: ")
    # get_size(featnet)

    # img0 = torch.randn(1, 3, 64, 64).to(device)
    # img1 = torch.randn(1, 3, 64, 64).to(device)
    # print("input image size: ", img0.size())

    # feats0 = featnet(img0)
    # feats1 = featnet(img1)

    # print("hierarchical feature size: ")
    # for f0, f1 in zip(feats0, feats1):
    #     print(f0.size(), f1.size())
    
    # inputs = [torch.cat([img0, img1], dim=1)]
    # for f0, f1 in zip(feats0, feats1):
    #     inputs.append(torch.cat([f0, f1], dim=1))

    # synth = gridnet(inputs)
    # print("synthesized image size: ", synth.size())

    print(featnet)
    print(gridnet)
